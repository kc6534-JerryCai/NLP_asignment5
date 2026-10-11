#!/usr/bin/env python
"""Verify and download Polymarket Democratic Sweep price history.

Raw API responses are immutable cache files. Cleaning always applies the exact
requested segment bounds, which removes the current-price observation that the
history endpoint can append beyond an historical request's endTs.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yaml


CSV_ENCODING = "utf-8-sig"
USER_AGENT = "Assignment5-polymarket-research/1.0"


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    raise ValueError(f"Expected JSON list, received {type(value).__name__}")


class ApiClient:
    def __init__(self, timeout: int, max_retries: int, backoff: float) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})

    def get_json(self, url: str, params: dict[str, str] | None = None) -> tuple[Any, dict[str, Any]]:
        attempts: list[dict[str, Any]] = []
        last_error = "unknown error"
        for attempt in range(1, self.max_retries + 1):
            started = datetime.now(timezone.utc)
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                record = {
                    "attempt": attempt,
                    "requested_at_utc": started.isoformat(),
                    "http_status": response.status_code,
                    "url": response.url,
                }
                attempts.append(record)
                if response.status_code == 429 or 500 <= response.status_code <= 599:
                    last_error = f"HTTP {response.status_code}"
                    if attempt < self.max_retries:
                        retry_after = response.headers.get("Retry-After")
                        delay = float(retry_after) if retry_after and retry_after.isdigit() else self.backoff * 2 ** (attempt - 1)
                        time.sleep(min(delay, 30.0))
                        continue
                response.raise_for_status()
                data = response.json()
                return data, {"attempts": attempts, "final_status": "success"}
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempts:
                    attempts[-1]["error"] = last_error
                if attempt < self.max_retries:
                    time.sleep(min(self.backoff * 2 ** (attempt - 1), 30.0))
        raise RuntimeError(json.dumps({"error": last_error, "attempts": attempts}, ensure_ascii=False))


def find_target_market(event: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    candidates = []
    for market in event.get("markets", []):
        if (
            str(market.get("id")) == str(cfg["market_id"])
            or market.get("question") == cfg["question"]
            or market.get("groupItemTitle") == cfg["group_item_title"]
        ):
            candidates.append(market)
    exact = [
        m
        for m in candidates
        if str(m.get("id")) == str(cfg["market_id"])
        and m.get("question") == cfg["question"]
        and m.get("groupItemTitle") == cfg["group_item_title"]
    ]
    if len(exact) != 1:
        raise RuntimeError(f"Target market verification failed; exact matches={len(exact)}")
    market = exact[0]
    outcomes = parse_json_list(market.get("outcomes"))
    tokens = parse_json_list(market.get("clobTokenIds"))
    if len(outcomes) != len(tokens):
        raise RuntimeError("outcomes and clobTokenIds lengths differ")
    mapping = dict(zip(outcomes, tokens))
    yes_keys = [key for key in mapping if key.casefold() == "yes"]
    if len(yes_keys) != 1:
        raise RuntimeError(f"Expected one Yes outcome, found {yes_keys}")
    yes_token = mapping[yes_keys[0]]
    if yes_token != str(cfg["expected_yes_token"]):
        raise RuntimeError("Verified Yes token differs from configured expected token")
    return {
        "market_id": str(market["id"]),
        "question": market["question"],
        "group_item_title": market["groupItemTitle"],
        "outcomes": outcomes,
        "clob_token_ids": tokens,
        "outcome_token_mapping": mapping,
        "yes_token": yes_token,
        "market_slug": market.get("slug"),
        "condition_id": market.get("conditionId"),
        "rules": market.get("rules"),
        "market_description_and_resolution_rules": market.get("description"),
        "resolution_source": market.get("resolutionSource"),
        "end_date": market.get("endDate"),
        "active": market.get("active"),
        "closed": market.get("closed"),
    }


def segment_bounds(start_ts: int, end_ts: int, segment_days: int) -> list[tuple[int, int]]:
    step = segment_days * 86400
    return [(s, min(s + step, end_ts)) for s in range(start_ts, end_ts, step)]


def raw_history_path(raw_dir: Path, start_ts: int, end_ts: int, fidelity: int) -> Path:
    return raw_dir / f"history_{start_ts}_{end_ts}_f{fidelity}.json"


def download_segment(
    client: ApiClient,
    history_url: str,
    token: str,
    start_ts: int,
    end_ts: int,
    fidelity: int,
    raw_dir: Path,
    minimum_seconds: int,
) -> list[dict[str, Any]]:
    path = raw_history_path(raw_dir, start_ts, end_ts, fidelity)
    base = {
        "start_ts": start_ts,
        "end_ts": end_ts,
        "duration_hours": (end_ts - start_ts) / 3600,
        "raw_path": str(path.as_posix()),
    }
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return [{**base, "status": "cached", "response": data, "request_meta": {"cache_hit": True}}]
        except (OSError, json.JSONDecodeError) as exc:
            return [{**base, "status": "failed", "error": f"Invalid immutable cache: {exc}"}]
    params = {"market": token, "startTs": str(start_ts), "endTs": str(end_ts), "fidelity": str(fidelity)}
    try:
        data, request_meta = client.get_json(history_url, params=params)
        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(f"API error: {data['error']}")
        if not isinstance(data, dict) or not isinstance(data.get("history"), list):
            raise RuntimeError("Response does not contain a history list")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return [{**base, "status": "downloaded", "response": data, "request_meta": request_meta}]
    except Exception as exc:
        if end_ts - start_ts > minimum_seconds:
            midpoint = start_ts + (end_ts - start_ts) // 2
            midpoint -= midpoint % 3600
            if midpoint <= start_ts or midpoint >= end_ts:
                return [{**base, "status": "failed", "error": str(exc)}]
            children = download_segment(client, history_url, token, start_ts, midpoint, fidelity, raw_dir, minimum_seconds)
            children += download_segment(client, history_url, token, midpoint, end_ts, fidelity, raw_dir, minimum_seconds)
            return [{**base, "status": "split_after_failure", "error": str(exc), "children": len(children)}] + children
        return [{**base, "status": "failed", "error": str(exc)}]


def clean_history(segment_records: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    outside: list[dict[str, Any]] = []
    segment_summaries = []
    for seg in segment_records:
        if seg.get("status") not in {"downloaded", "cached"}:
            continue
        history = seg["response"].get("history", [])
        kept = 0
        for item in history:
            try:
                t_raw = item.get("t")
                p_raw = item.get("p")
                t_float = float(t_raw)
                p = float(p_raw)
                if not math.isfinite(t_float) or not math.isfinite(p) or not t_float.is_integer() or not 0 <= p <= 1:
                    raise ValueError("non-finite, non-integer timestamp, or probability outside [0,1]")
                t = int(t_float)
            except (TypeError, ValueError, AttributeError) as exc:
                invalid.append({"segment_start": seg["start_ts"], "segment_end": seg["end_ts"], "item": item, "reason": str(exc)})
                continue
            if not (seg["start_ts"] <= t < seg["end_ts"]):
                outside.append({"segment_start": seg["start_ts"], "segment_end": seg["end_ts"], "t": t, "p": p})
                continue
            rows.append({"t": t, "p": p, "segment_start_ts": seg["start_ts"], "segment_end_ts": seg["end_ts"]})
            kept += 1
        segment_summaries.append({
            "start_ts": seg["start_ts"], "end_ts": seg["end_ts"], "status": seg["status"],
            "returned": len(history), "kept": kept, "discarded_outside": len(history) - kept,
            "raw_path": seg["raw_path"], "request_meta": seg.get("request_meta", {}),
        })
    frame = pd.DataFrame(rows)
    conflicts: list[dict[str, Any]] = []
    if not frame.empty:
        for t, group in frame.groupby("t"):
            values = sorted(set(float(x) for x in group["p"]))
            if len(values) > 1:
                conflicts.append({"t": int(t), "prices": values, "records": int(len(group))})
        frame = frame.sort_values(["t", "segment_start_ts"]).drop_duplicates("t", keep="last")
        frame["timestamp_utc"] = pd.to_datetime(frame["t"], unit="s", utc=True)
        frame["timestamp_new_york"] = frame["timestamp_utc"].dt.tz_convert("America/New_York")
        frame = frame[["t", "timestamp_utc", "timestamp_new_york", "p", "segment_start_ts", "segment_end_ts"]]
    report = {
        "segment_summaries": segment_summaries,
        "invalid_record_count": len(invalid),
        "invalid_records": invalid[:100],
        "outside_segment_count": len(outside),
        "outside_segment_records": outside[:200],
        "duplicate_timestamp_conflict_count": len(conflicts),
        "duplicate_timestamp_conflicts": conflicts,
    }
    return frame, report


def make_daily(hourly: pd.DataFrame, start_local: str, end_local_exclusive: str, stale_hours: float) -> pd.DataFrame:
    tz = ZoneInfo("America/New_York")
    start = pd.Timestamp(start_local, tz=tz)
    end = pd.Timestamp(end_local_exclusive, tz=tz)
    dates = pd.date_range(start.normalize(), end.normalize() - pd.Timedelta(days=1), freq="D")
    base = pd.DataFrame({"date": dates.date.astype(str)})
    if hourly.empty:
        grouped = pd.DataFrame(columns=["date", "p_close", "p_mean", "n_observations", "last_timestamp_utc", "last_timestamp_new_york"])
    else:
        work = hourly.copy()
        work["date"] = pd.to_datetime(work["timestamp_new_york"], utc=True).dt.tz_convert(tz).dt.date.astype(str)
        groups = []
        for date, group in work.groupby("date", sort=True):
            group = group.sort_values("t")
            last = group.iloc[-1]
            groups.append({
                "date": date,
                "p_close": float(last["p"]),
                "p_mean": float(group["p"].mean()),
                "n_observations": int(len(group)),
                "last_timestamp_utc": pd.Timestamp(last["timestamp_utc"]).isoformat(),
                "last_timestamp_new_york": pd.Timestamp(last["timestamp_new_york"]).isoformat(),
            })
        grouped = pd.DataFrame(groups)
    daily = base.merge(grouped, on="date", how="left")
    hours = []
    flags = []
    for row in daily.itertuples(index=False):
        if pd.isna(row.p_close):
            hours.append(float("nan"))
            flags.append("missing")
            continue
        day_end = pd.Timestamp(row.date, tz=tz) + pd.Timedelta(days=1)
        last = pd.Timestamp(row.last_timestamp_new_york)
        gap = (day_end - last).total_seconds() / 3600
        hours.append(gap)
        flags.append("stale" if gap > stale_hours else "ok")
    daily["hours_from_day_end"] = hours
    daily["quality_flag"] = flags
    daily["stale"] = daily["quality_flag"].eq("stale")
    daily["analysis_p_close"] = daily["p_close"].where(daily["quality_flag"].eq("ok"))
    daily["analysis_p_mean"] = daily["p_mean"].where(daily["quality_flag"].eq("ok"))
    return daily


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="sweep_analysis_config.yaml")
    parser.add_argument("--offline", action="store_true", help="Use existing raw cache only; do not call APIs")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config = load_config(root / args.config)
    project = config["project"]
    market_cfg = config["market"]
    poly_dir = root / project["polymarket_dir"]
    raw_dir = poly_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    client = ApiClient(int(market_cfg["timeout_seconds"]), int(market_cfg["max_retries"]), float(market_cfg["backoff_seconds"]))

    event_cache_files = sorted(raw_dir.glob("event_*.json"))
    event: dict[str, Any]
    event_source: str
    if args.offline:
        if not event_cache_files:
            raise RuntimeError("Offline mode requested but no cached event response exists")
        event_path = event_cache_files[-1]
        event = json.loads(event_path.read_text(encoding="utf-8"))
        event_source = "cache"
    else:
        event, _ = client.get_json(market_cfg["event_url"])
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        event_path = raw_dir / f"event_{stamp}.json"
        if event_path.exists():
            event_path = raw_dir / f"event_{stamp}_{time.time_ns()}.json"
        event_path.write_text(json.dumps(event, ensure_ascii=False), encoding="utf-8")
        event_source = "network"
    verified = find_target_market(event, market_cfg)
    retrieved_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        **verified,
        "verified_at_utc": retrieved_at,
        "event_source": event_source,
        "event_raw_path": str(event_path.relative_to(root).as_posix()),
        "event_url": market_cfg["event_url"],
        "history_url": market_cfg["history_url"],
        "research_window": {
            "timezone": project["timezone"], "start_local": project["start_local"],
            "end_local_exclusive": project["end_local_exclusive"], "start_ts": int(project["start_ts"]),
            "end_ts": int(project["end_ts"]),
        },
        "interpretation": "The Yes price is a market-implied probability proxy, not an objective probability.",
    }
    write_json(poly_dir / "market_metadata.json", metadata)

    records: list[dict[str, Any]] = []
    bounds = segment_bounds(int(project["start_ts"]), int(project["end_ts"]), int(market_cfg["segment_days"]))
    for start_ts, end_ts in bounds:
        path = raw_history_path(raw_dir, start_ts, end_ts, int(market_cfg["fidelity_minutes"]))
        if args.offline and not path.exists():
            records.append({"start_ts": start_ts, "end_ts": end_ts, "status": "failed", "error": "missing cache in offline mode", "raw_path": str(path.as_posix())})
            continue
        records.extend(download_segment(
            client, market_cfg["history_url"], verified["yes_token"], start_ts, end_ts,
            int(market_cfg["fidelity_minutes"]), raw_dir, int(market_cfg["minimum_segment_hours"]) * 3600,
        ))
    hourly, clean_report = clean_history(records)
    if not hourly.empty:
        hourly.to_csv(poly_dir / "hourly_sweep_prices.csv", index=False, encoding=CSV_ENCODING)
    else:
        pd.DataFrame(columns=["t", "timestamp_utc", "timestamp_new_york", "p", "segment_start_ts", "segment_end_ts"]).to_csv(
            poly_dir / "hourly_sweep_prices.csv", index=False, encoding=CSV_ENCODING
        )
    daily = make_daily(hourly, project["start_local"], project["end_local_exclusive"], float(market_cfg["stale_after_hours"]))
    daily.to_csv(poly_dir / "daily_sweep_probability.csv", index=False, encoding=CSV_ENCODING)

    leaf_records = [r for r in records if r.get("status") != "split_after_failure"]
    failed = [r for r in leaf_records if r.get("status") == "failed"]
    quality = {
        "generated_at_utc": retrieved_at,
        "market_verification": {"status": "pass", **verified},
        "requested_segments": len(bounds),
        "leaf_segments": len(leaf_records),
        "successful_or_cached_segments": sum(r.get("status") in {"downloaded", "cached"} for r in leaf_records),
        "failed_segments": [{k: v for k, v in r.items() if k != "response"} for r in failed],
        "all_requested_segments_complete": len(failed) == 0,
        "hourly_rows": int(len(hourly)),
        "hourly_min_ts": int(hourly["t"].min()) if not hourly.empty else None,
        "hourly_max_ts": int(hourly["t"].max()) if not hourly.empty else None,
        "daily_rows": int(len(daily)),
        "daily_quality_counts": daily["quality_flag"].value_counts(dropna=False).to_dict(),
        "daily_observed_from": daily.loc[daily["p_close"].notna(), "date"].min() if daily["p_close"].notna().any() else None,
        "daily_observed_to": daily.loc[daily["p_close"].notna(), "date"].max() if daily["p_close"].notna().any() else None,
        "strict_range_validation": bool(hourly.empty or ((hourly["t"] >= int(project["start_ts"])) & (hourly["t"] < int(project["end_ts"]))).all()),
        "probability_range_validation": bool(hourly.empty or hourly["p"].between(0, 1).all()),
        "timestamp_unique_validation": bool(hourly["t"].is_unique) if not hourly.empty else True,
        **clean_report,
        "sources": {
            "gamma_event": market_cfg["event_url"],
            "clob_price_history": market_cfg["history_url"],
            "official_docs_price_history": "https://docs.polymarket.com/api-reference/pricing/get-price-history-for-a-token",
            "official_docs_event_slug": "https://docs.polymarket.com/api-reference/events/get-event-by-slug",
        },
    }
    write_json(poly_dir / "download_quality_report.json", quality)
    write_json(poly_dir / "segment_manifest.json", [{k: v for k, v in r.items() if k != "response"} for r in records])
    print(json.dumps({
        "market_id": verified["market_id"], "yes_token": verified["yes_token"], "segments_complete": len(failed) == 0,
        "hourly_rows": len(hourly), "outside_segment_removed": clean_report["outside_segment_count"],
        "daily_quality": quality["daily_quality_counts"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
