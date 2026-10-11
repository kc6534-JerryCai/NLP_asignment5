#!/usr/bin/env python3
"""Collect Guardian articles for a US midterm-election text corpus.

The program deliberately keeps raw query hits and creates a second, lightly
filtered dataset. It does not perform topic modelling or sentiment analysis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


API_URL = "https://content.guardianapis.com/search"
SOURCE = "The Guardian"
DEFAULT_FROM = "2026-07-10"
DEFAULT_TO = "2026-10-09"

# Queries intentionally describe the election/institution/party context rather
# than preselecting policy topics that should later be discovered by LDA.
QUERIES: list[tuple[str, str]] = [
    ("midterm_elections", '"midterm election" OR "midterm elections" OR midterms'),
    (
        "us_congressional_elections",
        '("congressional election" OR "congressional elections") AND (US OR American OR America)',
    ),
    (
        "party_competition",
        '(Democrat OR Democrats OR Democratic) AND (Republican OR Republicans) '
        'AND (election OR elections OR midterm OR midterms OR Congress)',
    ),
    (
        "party_and_institutions",
        '(Democrat OR Democrats OR Democratic OR Republican OR Republicans) '
        'AND (Congress OR congressional OR Senate OR senator OR senators OR "House of Representatives")',
    ),
    (
        "congress_elections",
        '(Congress OR congressional) AND (election OR elections OR midterm OR midterms) '
        'AND (US OR American OR Democrat OR Republican)',
    ),
    (
        "house_elections",
        '("House of Representatives" OR "US House") AND '
        '(election OR elections OR midterm OR midterms OR Democrat OR Republican)',
    ),
    (
        "senate_elections",
        '(Senate OR senator OR senators) AND (election OR elections OR midterm OR midterms) '
        'AND (US OR American OR Democrat OR Republican)',
    ),
]

OUTPUT_FIELDS = [
    "article_id",
    "title",
    "published_at_utc",
    "published_at_new_york",
    "url",
    "normalized_url",
    "section",
    "source",
    "author",
    "summary",
    "body_text",
    "matched_queries",
    "retrieved_at_utc",
    "body_missing",
    "relevance_flag",
    "relevance_reason",
    "duplicate_body",
    "duplicate_of_article_id",
]


class TextExtractor(HTMLParser):
    """Conservative HTML-to-text converter that preserves punctuation."""

    BLOCK_TAGS = {
        "p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4",
        "h5", "h6", "blockquote", "section", "article", "tr", "td",
    }
    SKIP_TAGS = {"script", "style", "noscript", "svg"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.SKIP_TAGS:
            self.skip_depth += 1
        elif not self.skip_depth and tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP_TAGS and self.skip_depth:
            self.skip_depth -= 1
        elif not self.skip_depth and tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skip_depth:
            self.parts.append(data)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_dotenv(path: Path) -> None:
    """Minimal .env loader; existing environment variables take precedence."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and value[0:1] == value[-1:] and value.startswith(("'", '"')):
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if "<" in text and ">" in text:
        parser = TextExtractor()
        try:
            parser.feed(text)
            text = "".join(parser.parts)
        except Exception:
            text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text).replace("\u00a0", " ")
    text = re.sub(r"[\t\f\v ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_url(value: str) -> str:
    if not value:
        return ""
    parts = urlsplit(value.strip())
    host = parts.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    path = re.sub(r"/{2,}", "/", parts.path).rstrip("/") or "/"
    tracking = {"CMP", "cmp", "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content"}
    query = urlencode(sorted((k, v) for k, v in parse_qsl(parts.query) if k not in tracking))
    return urlunsplit((parts.scheme.lower() or "https", host, path, query, ""))


def parse_guardian_time(value: str) -> tuple[str, str]:
    if not value:
        return "", ""
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        utc_dt = dt.astimezone(timezone.utc)
        ny_dt = to_new_york(utc_dt)
        return (
            utc_dt.isoformat().replace("+00:00", "Z"),
            ny_dt.isoformat(),
        )
    except ValueError:
        return value, ""


def to_new_york(utc_dt: datetime) -> datetime:
    """Convert UTC to New York, including a no-tzdata Windows fallback.

    The fallback implements the US rules in force since 2007: daylight time
    begins at 07:00 UTC on the second Sunday in March and ends at 06:00 UTC on
    the first Sunday in November. The target assignment range is in 2026.
    """
    try:
        return utc_dt.astimezone(ZoneInfo("America/New_York"))
    except ZoneInfoNotFoundError:
        year = utc_dt.year
        march_1 = datetime(year, 3, 1, tzinfo=timezone.utc)
        first_sunday_march = march_1 + timedelta(days=(6 - march_1.weekday()) % 7)
        dst_start_utc = first_sunday_march + timedelta(days=7, hours=7)
        november_1 = datetime(year, 11, 1, tzinfo=timezone.utc)
        first_sunday_november = november_1 + timedelta(days=(6 - november_1.weekday()) % 7)
        dst_end_utc = first_sunday_november + timedelta(hours=6)
        is_dst = dst_start_utc <= utc_dt < dst_end_utc
        offset = timedelta(hours=-4 if is_dst else -5)
        return utc_dt.astimezone(timezone(offset, "EDT" if is_dst else "EST"))


def relevance(title: str, summary: str, body: str) -> tuple[str, str]:
    text = " ".join((title, summary, body[:12000])).lower()
    explicit_midterm = bool(re.search(r"\bmid[ -]?term(?: election)?s?\b", text))
    election = bool(re.search(r"\b(election|elections|electoral|campaign|ballot|vote|voting|race|races)\b", text))
    party = bool(re.search(r"\b(democrat|democrats|democratic party|republican|republicans|gop)\b", text))
    institution = bool(re.search(r"\b(congress|congressional|senate|senator|senators|us house|house of representatives)\b", text))
    us_context = bool(re.search(r"\b(united states|u\.?s\.?|american|america|washington dc|capitol hill)\b", text))

    if explicit_midterm:
        return "relevant", "explicit US midterm term found"
    if election and institution and (party or us_context):
        return "relevant", "election term plus US party/institution context"
    if election and party and us_context:
        return "relevant", "election term plus US party and country context"
    if institution and party:
        return "uncertain", "US party and federal institution context; no explicit election anchor"
    if election and (institution or party or us_context):
        return "uncertain", "partial election and US political context; manual review recommended"
    return "likely_irrelevant", "insufficient US election context after basic screening"


def record_from_result(result: dict[str, Any], query_name: str, query_text: str, retrieved: str) -> dict[str, Any]:
    fields = result.get("fields") or {}
    title = clean_text(fields.get("headline") or result.get("webTitle"))
    summary = clean_text(fields.get("trailText") or fields.get("standfirst"))
    body = clean_text(fields.get("bodyText"))
    published_utc, published_ny = parse_guardian_time(result.get("webPublicationDate", ""))
    flag, reason = relevance(title, summary, body)
    return {
        "article_id": str(result.get("id") or ""),
        "title": title,
        "published_at_utc": published_utc,
        "published_at_new_york": published_ny,
        "url": str(result.get("webUrl") or ""),
        "normalized_url": normalize_url(str(result.get("webUrl") or "")),
        "section": clean_text(result.get("sectionName") or result.get("sectionId")),
        "source": SOURCE,
        "author": clean_text(fields.get("byline")),
        "summary": summary,
        "body_text": body,
        "matched_queries": [query_text],
        "matched_query_names": [query_name],
        "retrieved_at_utc": retrieved,
        "body_missing": not bool(body),
        "relevance_flag": flag,
        "relevance_reason": reason,
        "duplicate_body": False,
        "duplicate_of_article_id": "",
    }


def merge_record(primary: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    primary["matched_queries"] = sorted(set(primary.get("matched_queries", [])) | set(other.get("matched_queries", [])))
    primary["matched_query_names"] = sorted(set(primary.get("matched_query_names", [])) | set(other.get("matched_query_names", [])))
    if primary.get("body_missing") and not other.get("body_missing"):
        for key in ("body_text", "summary", "author", "title"):
            if other.get(key):
                primary[key] = other[key]
        primary["body_missing"] = False
    elif len(str(other.get("body_text", ""))) > len(str(primary.get("body_text", ""))):
        primary["body_text"] = other["body_text"]
    primary["retrieved_at_utc"] = min(filter(None, [primary.get("retrieved_at_utc", ""), other.get("retrieved_at_utc", "")]), default="")
    flag, reason = relevance(primary.get("title", ""), primary.get("summary", ""), primary.get("body_text", ""))
    primary["relevance_flag"] = flag
    primary["relevance_reason"] = reason
    return primary


def body_fingerprint(body: str) -> set[str]:
    normalized = re.sub(r"[^a-z0-9']+", " ", body.lower())
    words = normalized.split()
    if len(words) < 40:
        return set()
    return {" ".join(words[i:i + 5]) for i in range(0, len(words) - 4, 3)}


def deduplicate(records: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_id: dict[str, dict[str, Any]] = {}
    id_merges = 0
    for record in records:
        key = record.get("article_id") or f"missing:{record.get('normalized_url')}"
        if key in by_id:
            merge_record(by_id[key], record)
            id_merges += 1
        else:
            by_id[key] = record

    by_url: dict[str, dict[str, Any]] = {}
    url_merges = 0
    no_url: list[dict[str, Any]] = []
    for record in by_id.values():
        url = record.get("normalized_url", "")
        if not url:
            no_url.append(record)
        elif url in by_url:
            merge_record(by_url[url], record)
            url_merges += 1
        else:
            by_url[url] = record
    unique = list(by_url.values()) + no_url
    unique.sort(key=lambda row: (row.get("published_at_utc", ""), row.get("article_id", "")))

    # Flag exact or very high-overlap bodies, but keep both articles. Candidate
    # comparison is restricted by publication day to avoid quadratic work and
    # to avoid conflating distinct reporting on the same long-running event.
    exact_seen: dict[str, str] = {}
    daily: dict[str, list[tuple[dict[str, Any], set[str]]]] = defaultdict(list)
    duplicate_flags = 0
    for row in unique:
        body = row.get("body_text", "")
        if not body:
            continue
        normalized_body = re.sub(r"\s+", " ", body.strip().lower())
        digest = hashlib.sha256(normalized_body.encode("utf-8")).hexdigest()
        if digest in exact_seen:
            row["duplicate_body"] = True
            row["duplicate_of_article_id"] = exact_seen[digest]
            duplicate_flags += 1
            continue
        exact_seen[digest] = row.get("article_id", "")
        fp = body_fingerprint(body)
        day = row.get("published_at_utc", "")[:10]
        if not fp:
            continue
        for prior, prior_fp in daily[day]:
            overlap = len(fp & prior_fp) / max(1, min(len(fp), len(prior_fp)))
            if overlap >= 0.90:
                row["duplicate_body"] = True
                row["duplicate_of_article_id"] = prior.get("article_id", "")
                duplicate_flags += 1
                break
        daily[day].append((row, fp))
    return unique, {"article_id_merges": id_merges, "normalized_url_merges": url_merges, "duplicate_body_flags": duplicate_flags}


def date_chunks(start: date, end: date, days: int) -> list[tuple[date, date]]:
    chunks: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(end, cursor + timedelta(days=days - 1))
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return chunks


@dataclass
class Budget:
    ledger_path: Path
    daily_limit: int
    run_limit: int
    run_count: int = 0
    last_request_monotonic: float = 0.0

    def _ledger(self) -> dict[str, Any]:
        return read_json(self.ledger_path, {"days": {}})

    def remaining(self) -> int:
        ledger = self._ledger()
        today = datetime.now(timezone.utc).date().isoformat()
        used = int(ledger.get("days", {}).get(today, 0))
        return max(0, min(self.daily_limit - used, self.run_limit - self.run_count))

    def consume(self) -> None:
        if self.remaining() <= 0:
            raise RequestBudgetExhausted("local request budget exhausted")
        elapsed = time.monotonic() - self.last_request_monotonic
        if elapsed < 1.05:
            time.sleep(1.05 - elapsed)
        ledger = self._ledger()
        today = datetime.now(timezone.utc).date().isoformat()
        days = ledger.setdefault("days", {})
        days[today] = int(days.get(today, 0)) + 1
        # Retain only a short history.
        for old in sorted(days)[:-14]:
            del days[old]
        atomic_json(self.ledger_path, ledger)
        self.run_count += 1
        self.last_request_monotonic = time.monotonic()


class RequestBudgetExhausted(RuntimeError):
    pass


class APIRequestError(RuntimeError):
    pass


def retry_delay(attempt: int, retry_after: str | None = None) -> float:
    if retry_after:
        try:
            return min(60.0, max(1.0, float(retry_after)))
        except ValueError:
            try:
                dt = parsedate_to_datetime(retry_after)
                return min(60.0, max(1.0, (dt - datetime.now(timezone.utc)).total_seconds()))
            except Exception:
                pass
    return min(30.0, (2 ** attempt) + random.random())


def fetch_json(api_key: str, params: dict[str, Any], budget: Budget, max_retries: int) -> tuple[dict[str, Any], int]:
    safe_params = dict(params)
    safe_params["api-key"] = api_key
    payload = urlencode(safe_params)
    last_error = "request failed"
    for attempt in range(max_retries + 1):
        budget.consume()
        request = Request(
            f"{API_URL}?{payload}",
            headers={"User-Agent": "Assignment5-GuardianCollector/1.0", "Accept": "application/json"},
        )
        try:
            with urlopen(request, timeout=30) as response:
                raw = response.read()
                return json.loads(raw.decode("utf-8")), int(response.status)
        except HTTPError as exc:
            status = int(exc.code)
            last_error = f"HTTP {status}"
            if status == 429 or 500 <= status <= 599:
                if attempt < max_retries:
                    time.sleep(retry_delay(attempt, exc.headers.get("Retry-After")))
                    continue
            raise APIRequestError(last_error) from None
        except (URLError, TimeoutError, json.JSONDecodeError):
            last_error = "network timeout/transport error or invalid JSON"
            if attempt < max_retries:
                time.sleep(retry_delay(attempt))
                continue
            raise APIRequestError(last_error) from None
    raise APIRequestError(last_error)


def redact(value: Any, api_key: str) -> Any:
    if isinstance(value, dict):
        return {
            key: ("[REDACTED]" if key.lower().replace("_", "-") == "api-key" else redact(item, api_key))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item, api_key) for item in value]
    if isinstance(value, str) and api_key:
        return value.replace(api_key, "[REDACTED]")
    return value


def append_records(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: skipped malformed state line {line_no}", file=sys.stderr)
    return rows


def write_outputs(data_dir: Path, rows: list[dict[str, Any]]) -> None:
    def public(row: dict[str, Any]) -> dict[str, Any]:
        return {field: row.get(field, "") for field in OUTPUT_FIELDS}

    def write_pair(stem: str, selected: list[dict[str, Any]]) -> None:
        jsonl_path = data_dir / f"{stem}.jsonl"
        with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in selected:
                handle.write(json.dumps(public(row), ensure_ascii=False) + "\n")
        csv_path = data_dir / f"{stem}.csv"
        # utf-8-sig emits a BOM so Excel opens multilingual text correctly.
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in selected:
                item = public(row)
                item["matched_queries"] = json.dumps(item["matched_queries"], ensure_ascii=False)
                writer.writerow(item)

    data_dir.mkdir(parents=True, exist_ok=True)
    write_pair("guardian_news", rows)
    filtered = [row for row in rows if row.get("relevance_flag") in {"relevant", "uncertain"}]
    write_pair("guardian_news_filtered", filtered)


def daily_quality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_day[row.get("published_at_utc", "")[:10] or "unknown"].append(row)
    return {
        day: {
            "articles": len(items),
            "body_missing": sum(bool(item.get("body_missing")) for item in items),
            "body_missing_rate": round(sum(bool(item.get("body_missing")) for item in items) / len(items), 4),
        }
        for day, items in sorted(by_day.items())
    }


def safe_task_id(query_name: str, start: str, end: str) -> str:
    return f"{start}__{end}__{query_name}"


def run_smoke_test(args: argparse.Namespace, api_key: str, budget: Budget) -> int:
    data_dir = Path(args.data_dir)
    raw_dir = data_dir / "raw" / "smoke_test"
    raw_dir.mkdir(parents=True, exist_ok=True)
    query_name, query_text = QUERIES[0]
    results: list[dict[str, Any]] = []
    pages_checked: list[int] = []
    errors: list[str] = []
    for page in (1, 2):
        params = {
            "q": query_text,
            "from-date": args.from_date,
            "to-date": args.to_date,
            "page-size": 2,
            "page": page,
            "order-by": "oldest",
            "type": "article",
            "show-fields": "headline,trailText,standfirst,byline,bodyText",
        }
        try:
            payload, _ = fetch_json(api_key, params, budget, args.max_retries)
        except (APIRequestError, RequestBudgetExhausted) as exc:
            errors.append(str(exc))
            break
        retrieved = utc_now()
        atomic_json(raw_dir / f"page_{page}.json", {
            "request": {key: value for key, value in params.items()},
            "retrieved_at_utc": retrieved,
            "api_response": redact(payload, api_key),
        })
        response = payload.get("response") or {}
        pages_checked.append(page)
        page_results = response.get("results") or []
        results.extend(record_from_result(item, query_name, query_text, retrieved) for item in page_results)
        if page >= int(response.get("pages") or 0):
            break

    report = {
        "tested_at_utc": utc_now(),
        "date_range": {"from": args.from_date, "to": args.to_date},
        "query": {"name": query_name, "q": query_text},
        "pages_checked": pages_checked,
        "records_checked": len(results),
        "body_present": sum(not row["body_missing"] for row in results),
        "body_missing": sum(row["body_missing"] for row in results),
        "timestamps_with_new_york_conversion": sum(bool(row["published_at_new_york"]) for row in results),
        "body_contains_html_tags": sum(bool(re.search(r"<[^>]+>", row["body_text"])) for row in results),
        "pagination_confirmed": len(pages_checked) >= 2,
        "sample": [
            {
                "title": row["title"],
                "published_at_utc": row["published_at_utc"],
                "published_at_new_york": row["published_at_new_york"],
                "url": row["url"],
                "body_excerpt": row["body_text"][:240],
            }
            for row in results[:4]
        ],
        "errors": errors,
        "requests_this_run": budget.run_count,
    }
    atomic_json(data_dir / "smoke_test_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if results and not errors else 2


def collect(args: argparse.Namespace, api_key: str, budget: Budget) -> int:
    data_dir = Path(args.data_dir)
    raw_dir = data_dir / "raw"
    state_dir = data_dir / ".state"
    state_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    progress_path = data_dir / "collection_progress.json"
    records_path = state_dir / "records.jsonl"

    start = date.fromisoformat(args.from_date)
    end = date.fromisoformat(args.to_date)
    if start > end:
        raise ValueError("--from-date must be on or before --to-date")
    chunks = date_chunks(start, end, args.chunk_days)
    tasks = [
        (safe_task_id(name, str(chunk_start), str(chunk_end)), name, query, str(chunk_start), str(chunk_end))
        for chunk_start, chunk_end in chunks
        for name, query in QUERIES
    ]
    signature = hashlib.sha256(json.dumps({
        "from": args.from_date, "to": args.to_date, "chunk_days": args.chunk_days,
        "page_size": args.page_size, "queries": QUERIES,
    }, sort_keys=True).encode("utf-8")).hexdigest()
    progress = read_json(progress_path, {})
    if progress and progress.get("collection_signature") != signature:
        raise ValueError(
            "Existing progress belongs to different dates/query settings. "
            "Use --fresh to archive/reset collection state before changing them."
        )
    if not progress:
        progress = {
            "collection_signature": signature,
            "requested_range": {"from": args.from_date, "to": args.to_date},
            "created_at_utc": utc_now(),
            "updated_at_utc": utc_now(),
            "queries": [{"name": name, "q": query} for name, query in QUERIES],
            "tasks": {},
            "article_ids": [],
        }
    # Migrate progress created by v1 of this script, which only retained the
    # most recent invocation's count. Keeping this cumulative makes a resumed
    # run's final report truthful even when the last invocation needs 0 calls.
    prior_request_attempts = int(
        progress.get("request_attempts_total", progress.get("requests_this_invocation", 0))
    )

    budget_stop = False
    for task_index, (task_id, query_name, query_text, chunk_start, chunk_end) in enumerate(tasks, 1):
        task_state = progress["tasks"].setdefault(task_id, {
            "query_name": query_name,
            "query": query_text,
            "from_date": chunk_start,
            "to_date": chunk_end,
            "status": "pending",
            "next_page": 1,
            "pages_total": None,
            "results_reported": None,
            "records_received": 0,
            "errors": [],
        })
        if task_state.get("status") == "complete":
            continue
        print(f"[{task_index}/{len(tasks)}] {chunk_start}..{chunk_end} {query_name}")
        while True:
            page = int(task_state.get("next_page", 1))
            params = {
                "q": query_text,
                "from-date": chunk_start,
                "to-date": chunk_end,
                "page-size": args.page_size,
                "page": page,
                "order-by": "oldest",
                "type": "article",
                "show-fields": "headline,trailText,standfirst,byline,bodyText",
            }
            try:
                payload, status = fetch_json(api_key, params, budget, args.max_retries)
            except RequestBudgetExhausted as exc:
                task_state["status"] = "incomplete_budget"
                task_state["errors"].append({"at_utc": utc_now(), "page": page, "error": str(exc)})
                progress["request_attempts_total"] = prior_request_attempts + budget.run_count
                budget_stop = True
                break
            except APIRequestError as exc:
                task_state["status"] = "failed"
                task_state["errors"].append({"at_utc": utc_now(), "page": page, "error": str(exc)})
                progress["request_attempts_total"] = prior_request_attempts + budget.run_count
                break

            retrieved = utc_now()
            response = payload.get("response") or {}
            page_results = response.get("results") or []
            raw_task_dir = raw_dir / task_id
            atomic_json(raw_task_dir / f"page_{page:04d}.json", {
                "request": {key: value for key, value in params.items()},
                "http_status": status,
                "retrieved_at_utc": retrieved,
                "api_response": redact(payload, api_key),
            })
            new_rows = [record_from_result(item, query_name, query_text, retrieved) for item in page_results]
            append_records(records_path, new_rows)
            task_state["pages_total"] = int(response.get("pages") or 0)
            task_state["results_reported"] = int(response.get("total") or 0)
            task_state["records_received"] = int(task_state.get("records_received", 0)) + len(new_rows)
            task_state["next_page"] = page + 1
            task_state["status"] = "in_progress"
            current_ids = set(progress.get("article_ids", []))
            current_ids.update(row["article_id"] for row in new_rows if row["article_id"])
            progress["article_ids"] = sorted(current_ids)
            progress["updated_at_utc"] = utc_now()
            progress["requests_this_invocation"] = budget.run_count
            progress["request_attempts_total"] = prior_request_attempts + budget.run_count
            atomic_json(progress_path, progress)

            pages = int(response.get("pages") or 0)
            if page >= pages or not page_results:
                task_state["status"] = "complete"
                task_state["completed_at_utc"] = utc_now()
                atomic_json(progress_path, progress)
                break
        progress["updated_at_utc"] = utc_now()
        atomic_json(progress_path, progress)
        if budget_stop:
            break

    progress["updated_at_utc"] = utc_now()
    progress["requests_this_invocation"] = budget.run_count
    progress["request_attempts_total"] = prior_request_attempts + budget.run_count
    atomic_json(progress_path, progress)
    state_rows = load_records(records_path)
    unique_rows, dedup_stats = deduplicate(state_rows)
    write_outputs(data_dir, unique_rows)
    completed = [task for task in progress["tasks"].values() if task.get("status") == "complete"]
    incomplete = [
        {
            "query_name": task.get("query_name"),
            "from_date": task.get("from_date"),
            "to_date": task.get("to_date"),
            "status": task.get("status"),
            "next_page": task.get("next_page"),
            "errors": task.get("errors", []),
        }
        for _, name, query, chunk_start, chunk_end in tasks
        for task in [progress["tasks"].get(safe_task_id(name, chunk_start, chunk_end), {
            "query_name": name, "from_date": chunk_start, "to_date": chunk_end,
            "status": "pending", "next_page": 1, "errors": [],
        })]
        if task.get("status") != "complete"
    ]
    raw_count = len(state_rows)
    filtered_count = sum(row.get("relevance_flag") in {"relevant", "uncertain"} for row in unique_rows)
    relevance_counts = Counter(row.get("relevance_flag", "unknown") for row in unique_rows)
    completed_dates = sorted({
        day
        for task in completed
        for day in (
            date.fromisoformat(task["from_date"]) + timedelta(days=offset)
            for offset in range((date.fromisoformat(task["to_date"]) - date.fromisoformat(task["from_date"])).days + 1)
        )
    })
    # A date is fully covered only if every configured query task for its chunk completed.
    fully_covered_chunks = []
    for chunk_start, chunk_end in chunks:
        chunk_tasks = [progress["tasks"].get(safe_task_id(name, str(chunk_start), str(chunk_end)), {}) for name, _ in QUERIES]
        if all(item.get("status") == "complete" for item in chunk_tasks):
            fully_covered_chunks.append({"from": str(chunk_start), "to": str(chunk_end)})

    samples = []
    candidates = [row for row in unique_rows if row.get("body_text") and row.get("relevance_flag") != "likely_irrelevant"]
    sample_rng = random.Random(20261010)
    for row in sample_rng.sample(candidates, min(10, len(candidates))):
        samples.append({
            "title": row["title"],
            "published_at_utc": row["published_at_utc"],
            "url": row["url"],
            "body_excerpt": row["body_text"][:300],
        })
    summary = {
        "generated_at_utc": utc_now(),
        "requested_date_range_inclusive": {"from": args.from_date, "to": args.to_date},
        "actual_completed_date_range_inclusive": (
            {"from": args.from_date, "to": args.to_date} if not incomplete else None
        ),
        "actual_fully_completed_chunks": fully_covered_chunks,
        "collection_complete": not incomplete,
        "queries": [{"name": name, "q": query} for name, query in QUERIES],
        "request_statistics": {
            "requests_this_invocation": budget.run_count,
            "collection_request_attempts_total": prior_request_attempts + budget.run_count,
            "configured_daily_guard": budget.daily_limit,
            "configured_run_budget": budget.run_limit,
            "tasks_total": len(tasks),
            "tasks_complete": len(completed),
        },
        "record_statistics": {
            "raw_records_across_query_hits": raw_count,
            "unique_after_id_and_url_deduplication": len(unique_rows),
            "filtered_relevant_or_uncertain": filtered_count,
            "body_missing": sum(row.get("body_missing", False) for row in unique_rows),
            "body_missing_rate": round(sum(row.get("body_missing", False) for row in unique_rows) / max(1, len(unique_rows)), 4),
            "relevance_flags": dict(sorted(relevance_counts.items())),
            **dedup_stats,
        },
        "daily_quality": daily_quality(unique_rows),
        "incomplete_tasks": incomplete,
        "sample_articles": samples,
        "notes": [
            "guardian_news files retain every deduplicated query hit.",
            "guardian_news_filtered files retain relevant and uncertain rows; uncertain rows require manual review.",
            "Summary text is never substituted for a missing body; body_missing remains explicit.",
        ],
    }
    atomic_json(data_dir / "collection_summary.json", summary)
    print(json.dumps({
        "collection_complete": summary["collection_complete"],
        "requests_this_invocation": budget.run_count,
        **summary["record_statistics"],
        "incomplete_tasks": len(incomplete),
    }, ensure_ascii=False, indent=2))
    return 0 if not incomplete else 3


def archive_fresh_state(data_dir: Path) -> None:
    """Move resumable state aside without deleting user data."""
    progress = data_dir / "collection_progress.json"
    records = data_dir / ".state" / "records.jsonl"
    if not progress.exists() and not records.exists():
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = data_dir / ".state" / "archive" / stamp
    archive.mkdir(parents=True, exist_ok=False)
    if progress.exists():
        progress.replace(archive / progress.name)
    if records.exists():
        records.replace(archive / records.name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-date", default=DEFAULT_FROM, help="inclusive YYYY-MM-DD")
    parser.add_argument("--to-date", default=DEFAULT_TO, help="inclusive YYYY-MM-DD")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--chunk-days", type=int, default=7)
    parser.add_argument("--page-size", type=int, default=200)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--daily-request-limit", type=int, default=475, help="local safety guard below the documented 500/day")
    parser.add_argument("--run-request-budget", type=int, default=450)
    parser.add_argument("--smoke-test", action="store_true", help="fetch at most two 2-record pages without changing collection state")
    parser.add_argument("--fresh", action="store_true", help="archive prior resumable state before a changed collection")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.chunk_days < 1 or not 1 <= args.page_size <= 200:
        print("Error: --chunk-days must be >= 1 and --page-size must be 1..200", file=sys.stderr)
        return 2
    try:
        date.fromisoformat(args.from_date)
        date.fromisoformat(args.to_date)
    except ValueError:
        print("Error: dates must use YYYY-MM-DD", file=sys.stderr)
        return 2
    load_dotenv(Path(".env"))
    api_key = os.environ.get("GUARDIAN_API_KEY", "").strip()
    if not api_key or api_key == "replace_with_your_guardian_api_key":
        print(
            "GUARDIAN_API_KEY is not configured. Copy .env.example to .env, set the key locally, "
            "and do not paste it into chat or commit .env.",
            file=sys.stderr,
        )
        return 2
    data_dir = Path(args.data_dir)
    if args.fresh and not args.smoke_test:
        archive_fresh_state(data_dir)
    budget = Budget(
        ledger_path=data_dir / ".state" / "request_ledger.json",
        daily_limit=args.daily_request_limit,
        run_limit=args.run_request_budget,
    )
    try:
        if args.smoke_test:
            return run_smoke_test(args, api_key, budget)
        return collect(args, api_key, budget)
    except (ValueError, OSError) as exc:
        # Never include request URLs or credentials in error output.
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
