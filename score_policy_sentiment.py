#!/usr/bin/env python
"""Score K=6 Guardian policy topics with DeepSeek and build daily/weekly indices.

The script never changes the collected Guardian data or retrains LDA. API results
are cached by article, body hash, topic-definition version, prompt version, and
model. Missing or failed annotations remain missing in every aggregate.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import httpx
import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont


CSV_ENCODING = "utf-8-sig"
VALID_SENTIMENT = {"positive": 1, "negative": -1, "neutral": 0, "mixed": 0, "unclear": None}
VALID_STANCE = {"support": 1, "oppose": -1, "neutral": 0, "mixed": 0, "not_expressed": None, "unclear": None}
SPEAKER_TYPES = {"author", "quoted_person", "multiple_voices", "unclear"}
TOPIC_COLUMNS = [f"topic_{i}_probability" for i in range(6)]
SUCCESS_STATUSES = {"success", "cached"}


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding=CSV_ENCODING)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_evidence(text: str) -> str:
    value = str(text or "").strip().strip('"“”‘’')
    value = value.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", value).casefold()


@dataclass
class Inputs:
    probabilities: pd.DataFrame
    articles: dict[str, dict[str, Any]]
    joined: list[dict[str, Any]]
    topics: list[dict[str, Any]]
    audit: dict[str, Any]


def read_inputs(config: dict[str, Any], root: Path) -> Inputs:
    project = config["project"]
    prob_path = root / project["input_probabilities"]
    article_path = root / project["input_articles"]
    definition_path = root / project["topic_definitions"]
    probabilities = pd.read_csv(prob_path, encoding=CSV_ENCODING)
    required = {"article_id", "title", "published_at_new_york", "section", *TOPIC_COLUMNS}
    missing_columns = sorted(required - set(probabilities.columns))
    if missing_columns:
        raise ValueError(f"Probability file is missing columns: {missing_columns}")
    duplicate_probability_ids = probabilities.loc[probabilities["article_id"].duplicated(False), "article_id"].tolist()
    if duplicate_probability_ids:
        raise ValueError(f"Duplicate probability article IDs: {duplicate_probability_ids[:5]}")
    if len(probabilities) != 226:
        raise ValueError(f"Expected 226 policy articles, found {len(probabilities)}")
    probability_sums = probabilities[TOPIC_COLUMNS].astype(float).sum(axis=1)
    if float((probability_sums - 1.0).abs().max()) > 1e-5:
        raise ValueError("LDA probability sums exceed tolerance")

    articles: dict[str, dict[str, Any]] = {}
    duplicate_article_ids: list[str] = []
    with article_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            article_id = str(item.get("article_id", ""))
            if article_id in articles:
                duplicate_article_ids.append(article_id)
            articles[article_id] = item
    if duplicate_article_ids:
        raise ValueError(f"Duplicate raw article IDs: {duplicate_article_ids[:5]}")

    topic_data = load_json(definition_path)
    topics = sorted(topic_data["topics"], key=lambda item: int(item["topic_id"]))
    if [int(item["topic_id"]) for item in topics] != list(range(6)):
        raise ValueError("Topic definition IDs must be exactly 0 through 5")

    joined: list[dict[str, Any]] = []
    unmatched: list[str] = []
    missing_bodies: list[str] = []
    for row in probabilities.to_dict("records"):
        article_id = str(row["article_id"])
        article = articles.get(article_id)
        if article is None:
            unmatched.append(article_id)
            continue
        body = str(article.get("body_text") or "").strip()
        if not body:
            missing_bodies.append(article_id)
        joined.append({**row, "body_text": body, "url": article.get("url", ""), "raw_title": article.get("title", "")})
    if unmatched or missing_bodies:
        raise ValueError(f"Join failed: unmatched={len(unmatched)}, missing bodies={len(missing_bodies)}")

    observed_dates = pd.to_datetime(probabilities["published_at_new_york"], errors="coerce").dt.date
    audit = {
        "probability_rows": len(probabilities),
        "unique_probability_article_ids": int(probabilities["article_id"].nunique()),
        "raw_article_rows": len(articles),
        "joined_articles": len(joined),
        "unmatched_article_ids": unmatched,
        "missing_body_article_ids": missing_bodies,
        "topic_probability_columns": TOPIC_COLUMNS,
        "max_probability_sum_error": float((probability_sums - 1.0).abs().max()),
        "minimum_new_york_date": min(observed_dates).isoformat(),
        "maximum_new_york_date": max(observed_dates).isoformat(),
        "section_counts": probabilities["section"].fillna("").value_counts().to_dict(),
        "topic_definition_version": topic_data["version"],
    }
    return Inputs(probabilities, articles, joined, topics, audit)


def cache_key(article: dict[str, Any], config: dict[str, Any]) -> tuple[str, str]:
    body_hash = hashlib.sha256(article["body_text"].encode("utf-8")).hexdigest()
    api = config["api"]
    payload = "|".join([
        str(article["article_id"]), body_hash, str(api["topic_definition_version"]),
        str(api["prompt_version"]), str(api["model"]),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), body_hash


def user_message(article: dict[str, Any], topics: list[dict[str, Any]]) -> str:
    probabilities = {f"T{i}": float(article[f"topic_{i}_probability"]) for i in range(6)}
    topic_context = [{
        "topic_id": item["topic_id"], "topic_name": item["topic_name"],
        "english_name": item["english_name"], "definition": item["definition"],
        "subissue_guidance": item.get("subissue_guidance", []),
        "special_instruction": item.get("special_instruction", ""),
    } for item in topics]
    return (
        "Return JSON following the required schema.\n\n"
        f"ARTICLE_ID: {article['article_id']}\n"
        f"TITLE: {article['title']}\n"
        f"SECTION: {article['section']}\n"
        f"PUBLISHED_AT_NEW_YORK: {article['published_at_new_york']}\n"
        f"LDA_PROBABILITIES_CONTEXT_ONLY: {json.dumps(probabilities)}\n"
        f"TOPIC_DEFINITIONS: {json.dumps(topic_context, ensure_ascii=False)}\n\n"
        "ARTICLE_BODY_START\n" + article["body_text"] + "\nARTICLE_BODY_END"
    )


def normalize_annotation(value: dict[str, Any]) -> dict[str, Any]:
    for topic in value.get("topics", []):
        for field in ("topic_sentiment_label", "topic_sentiment_score", "democrat_stance_score", "republican_stance_score"):
            if topic.get(field) == "":
                topic[field] = None
        for field in ("subissue", "sentiment_target", "topic_sentiment_evidence", "democrat_evidence",
                      "republican_evidence", "speaker_name", "ambiguity_note"):
            if topic.get(field) is None:
                topic[field] = ""
    return value


def validate_annotation(annotation: dict[str, Any], article: dict[str, Any], topics: list[dict[str, Any]]) -> tuple[list[str], dict[str, bool]]:
    errors: list[str] = []
    evidence_checks: dict[str, bool] = {}
    if annotation.get("article_id") != article["article_id"]:
        errors.append("article_id mismatch")
    values = annotation.get("topics")
    if not isinstance(values, list) or len(values) != 6:
        return errors + ["topics must contain exactly six objects"], evidence_checks
    ids = [item.get("topic_id") for item in values if isinstance(item, dict)]
    if ids != list(range(6)):
        errors.append(f"topic IDs/order invalid: {ids}")
    combined_text = normalize_evidence(str(article["title"]) + "\n" + article["body_text"])
    for expected, item in zip(topics, values):
        if not isinstance(item, dict):
            errors.append("topic item is not an object")
            continue
        topic_id = int(expected["topic_id"])
        prefix = f"T{topic_id}"
        if item.get("topic_name") != expected["topic_name"]:
            errors.append(f"{prefix} topic_name mismatch")
        relevant = item.get("relevant")
        if not isinstance(relevant, bool):
            errors.append(f"{prefix} relevant is not boolean")
        sentiment_label = item.get("topic_sentiment_label")
        sentiment_score = item.get("topic_sentiment_score")
        if relevant is False:
            if sentiment_label is not None or sentiment_score is not None:
                errors.append(f"{prefix} irrelevant topic has sentiment")
            for party in ("democrat", "republican"):
                if item.get(f"{party}_stance_label") != "not_expressed" or item.get(f"{party}_stance_score") is not None:
                    errors.append(f"{prefix} irrelevant topic has {party} stance")
        elif relevant is True:
            if sentiment_label not in VALID_SENTIMENT or VALID_SENTIMENT.get(sentiment_label) != sentiment_score:
                errors.append(f"{prefix} invalid sentiment label/score")
            if topic_id in {2, 4} and not str(item.get("subissue", "")).strip():
                errors.append(f"{prefix} requires a specific subissue")
        for party in ("democrat", "republican"):
            label, score = item.get(f"{party}_stance_label"), item.get(f"{party}_stance_score")
            if label not in VALID_STANCE or VALID_STANCE.get(label) != score:
                errors.append(f"{prefix} invalid {party} label/score")
        if item.get("speaker_type") not in SPEAKER_TYPES:
            errors.append(f"{prefix} invalid speaker_type")
        evidence_fields = {
            "sentiment": (sentiment_score, item.get("topic_sentiment_evidence", "")),
            "democrat": (item.get("democrat_stance_score"), item.get("democrat_evidence", "")),
            "republican": (item.get("republican_stance_score"), item.get("republican_evidence", "")),
        }
        for dimension, (score, evidence) in evidence_fields.items():
            normalized = normalize_evidence(evidence)
            key = f"{prefix}_{dimension}"
            evidence_checks[key] = (normalized in combined_text) if normalized else score is None
            if score is not None and not normalized:
                errors.append(f"{prefix} {dimension} non-null score lacks evidence")
            elif normalized and normalized not in combined_text:
                errors.append(f"{prefix} {dimension} evidence is not an exact contiguous source quotation")
    return errors, evidence_checks


def request_cost(usage: dict[str, Any], config: dict[str, Any], request_time: datetime) -> tuple[float, str]:
    pricing = config["pricing_usd_per_million_tokens"]
    is_weekday = request_time.weekday() < 5
    minutes = request_time.hour * 60 + request_time.minute
    is_peak = is_weekday and ((60 <= minutes < 240) or (360 <= minutes < 600))
    period = "peak" if is_peak else "off_peak"
    hit = int(usage.get("prompt_cache_hit_tokens", 0) or usage.get("prompt_tokens_details", {}).get("cached_tokens", 0) or 0)
    miss = int(usage.get("prompt_cache_miss_tokens", 0) or max(0, int(usage.get("prompt_tokens", 0) or 0) - hit))
    output = int(usage.get("completion_tokens", 0) or 0)
    cost = (
        hit * float(pricing[f"cache_hit_{period}"]) +
        miss * float(pricing[f"cache_miss_{period}"]) +
        output * float(pricing[f"output_{period}"])
    ) / 1_000_000
    return cost, period


class RunBudget:
    def __init__(self, max_requests: int, max_cost: float):
        self.max_requests = max_requests
        self.max_cost = max_cost
        self.requests = 0
        self.cost = 0.0
        self.lock = asyncio.Lock()

    async def reserve(self) -> None:
        async with self.lock:
            if self.requests >= self.max_requests:
                raise RuntimeError("request budget exhausted")
            if self.cost >= self.max_cost:
                raise RuntimeError("cost budget exhausted")
            self.requests += 1

    async def add_cost(self, value: float) -> None:
        async with self.lock:
            self.cost += value


async def annotate_one(
    client: httpx.AsyncClient, article: dict[str, Any], topics: list[dict[str, Any]],
    prompt: str, config: dict[str, Any], output_dir: Path, budget: RunBudget,
    log_lock: asyncio.Lock,
) -> dict[str, Any]:
    api = config["api"]
    key, body_hash = cache_key(article, config)
    cache_path = output_dir / "cache" / f"{key}.json"
    if cache_path.exists():
        cached = load_json(cache_path)
        annotation = normalize_annotation(cached["annotation"])
        errors, evidence = validate_annotation(annotation, article, topics)
        if not errors:
            return {"status": "cached", "cache_key": key, "body_sha256": body_hash,
                    "annotation": annotation, "usage": cached.get("usage", {}),
                    "cost_usd": cached.get("cost_usd", 0.0), "evidence_checks": evidence,
                    "processed_at_utc": cached.get("processed_at_utc", "")}

    payload = {
        "model": api["model"],
        "messages": [{"role": "system", "content": prompt},
                     {"role": "user", "content": user_message(article, topics)}],
        "thinking": {"type": "disabled"},
        "reasoning_effort": "none",
        "response_format": {"type": "json_object"},
        "max_tokens": int(api["max_output_tokens"]),
        "temperature": 0,
        "stream": False,
    }
    retries = int(api["max_retries"])
    last_error = ""
    for attempt in range(retries + 1):
        request_time = datetime.now(timezone.utc)
        await budget.reserve()
        started = time.perf_counter()
        status_code: int | None = None
        attempt_usage: dict[str, Any] = {}
        attempt_cost = 0.0
        price_period = ""
        response_model = api["model"]
        try:
            request_payload = dict(payload)
            if last_error:
                request_payload["messages"] = [dict(message) for message in payload["messages"]]
                request_payload["messages"][-1]["content"] += (
                    "\n\nREPAIR_INSTRUCTION: The previous JSON failed validation: " + last_error +
                    ". Re-read the original article and return the entire JSON again. Correct only by using exact, "
                    "contiguous source quotations for every cited evidence field; do not paraphrase or join fragments. "
                    "If no exact quotation supports a party stance, set that party label to not_expressed, its score "
                    "to null, and its evidence to an empty string rather than inventing or repairing a quotation."
                )
            response = await client.post("/chat/completions", json=request_payload)
            status_code = response.status_code
            if status_code == 429 or status_code >= 500:
                raise httpx.HTTPStatusError(f"retryable status {status_code}", request=response.request, response=response)
            response.raise_for_status()
            data = response.json()
            attempt_usage = data.get("usage", {}) or {}
            attempt_cost, price_period = request_cost(attempt_usage, config, request_time)
            response_model = data.get("model", api["model"])
            await budget.add_cost(attempt_cost)
            content = data["choices"][0]["message"]["content"]
            annotation = normalize_annotation(json.loads(content))
            errors, evidence = validate_annotation(annotation, article, topics)
            if errors:
                raise ValueError("; ".join(errors))
            record = {
                "cache_key": key, "article_id": article["article_id"], "body_sha256": body_hash,
                "topic_definition_version": api["topic_definition_version"],
                "prompt_version": api["prompt_version"], "model": response_model,
                "thinking": "disabled", "annotation": annotation, "usage": attempt_usage,
                "cost_usd": attempt_cost, "price_period": price_period, "processed_at_utc": iso_utc(),
            }
            write_json(cache_path, record)
            ledger = {"article_id": article["article_id"], "attempt": attempt + 1, "status": "success",
                      "http_status": status_code, "elapsed_seconds": time.perf_counter() - started,
                      "processed_at_utc": record["processed_at_utc"], "usage": attempt_usage, "cost_usd": attempt_cost,
                      "cache_key": key, "model": record["model"], "prompt_version": api["prompt_version"]}
            async with log_lock:
                append_jsonl(output_dir / "api_request_ledger.jsonl", ledger)
            return {"status": "success", "cache_key": key, "body_sha256": body_hash,
                    "annotation": annotation, "usage": attempt_usage, "cost_usd": attempt_cost,
                    "evidence_checks": evidence, "processed_at_utc": record["processed_at_utc"]}
        except Exception as exc:  # bounded retry path includes transport, HTTP, JSON and schema errors
            last_error = f"{type(exc).__name__}: {exc}"
            ledger = {"article_id": article["article_id"], "attempt": attempt + 1, "status": "retry" if attempt < retries else "failed",
                      "http_status": status_code, "elapsed_seconds": time.perf_counter() - started,
                      "processed_at_utc": iso_utc(), "error": last_error, "cache_key": key,
                      "model": response_model, "prompt_version": api["prompt_version"],
                      "usage": attempt_usage, "cost_usd": attempt_cost, "price_period": price_period}
            async with log_lock:
                append_jsonl(output_dir / "api_request_ledger.jsonl", ledger)
            if attempt < retries:
                await asyncio.sleep(float(api["backoff_seconds"]) * (2 ** attempt))
    return {"status": "failed", "cache_key": key, "body_sha256": body_hash,
            "annotation": None, "usage": {}, "cost_usd": 0.0,
            "evidence_checks": {}, "processed_at_utc": iso_utc(), "error": last_error}


def trial_articles(joined: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    frame = pd.DataFrame(joined).copy()
    frame["dominant_topic"] = frame[TOPIC_COLUMNS].astype(float).values.argmax(axis=1)
    frame["date"] = pd.to_datetime(frame["published_at_new_york"]).dt.date
    chosen: list[int] = []
    per_topic = max(1, size // 6)
    for topic_id in range(6):
        group = frame.loc[frame["dominant_topic"] == topic_id].sort_values("date")
        opinion = group.loc[group["section"] == "Opinion"]
        if len(opinion):
            chosen.append(int(opinion.index[len(opinion) // 2]))
        remaining = group.loc[~group.index.isin(chosen)]
        need = min(per_topic - (1 if len(opinion) else 0), len(remaining))
        if need > 0:
            positions = np.linspace(0, len(remaining) - 1, need, dtype=int)
            chosen.extend(int(remaining.index[pos]) for pos in positions)
    if len(chosen) < size:
        remaining = frame.loc[~frame.index.isin(chosen)].sort_values("date")
        positions = np.linspace(0, len(remaining) - 1, min(size - len(chosen), len(remaining)), dtype=int)
        chosen.extend(int(remaining.index[pos]) for pos in positions)
    selected_ids = frame.loc[chosen[:size], "article_id"].tolist()
    by_id = {row["article_id"]: row for row in joined}
    return [by_id[item] for item in selected_ids]


def failure_topics(article: dict[str, Any], topics: list[dict[str, Any]], result: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    return [{
        "article_id": article["article_id"], "title": article["title"],
        "published_at_new_york": article["published_at_new_york"], "new_york_date": article["published_at_new_york"][:10],
        "section": article["section"], "url": article.get("url", ""), "topic_id": i,
        "topic_name": topics[i]["topic_name"], "LDA_probability": float(article[f"topic_{i}_probability"]),
        "relevant": None, "subissue": "", "sentiment_target": "",
        "topic_sentiment_label": None, "topic_sentiment_score": None, "topic_sentiment_evidence": "",
        "democrat_stance_label": None, "democrat_stance_score": None, "democrat_evidence": "",
        "republican_stance_label": None, "republican_stance_score": None, "republican_evidence": "",
        "speaker_type": "", "speaker_name": "", "ambiguity_note": "",
        "model": config["api"]["model"], "prompt_version": config["api"]["prompt_version"],
        "topic_definition_version": config["api"]["topic_definition_version"],
        "processing_status": "failed", "evidence_validation_status": None,
        "topic_sentiment_evidence_valid": None, "democrat_evidence_valid": None, "republican_evidence_valid": None,
        "cache_key": result["cache_key"], "processed_at_utc": result["processed_at_utc"],
        "processing_error": result.get("error", ""),
    } for i in range(6)]


def flatten_result(article: dict[str, Any], topics: list[dict[str, Any]], result: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    if result["status"] == "failed" or not result.get("annotation"):
        return failure_topics(article, topics, result, config)
    rows: list[dict[str, Any]] = []
    for item in result["annotation"]["topics"]:
        topic_id = int(item["topic_id"])
        checks = result.get("evidence_checks", {})
        relevant_checks = [checks.get(f"T{topic_id}_{dimension}", True) for dimension in ("sentiment", "democrat", "republican")]
        rows.append({
            "article_id": article["article_id"], "title": article["title"],
            "published_at_new_york": article["published_at_new_york"], "new_york_date": article["published_at_new_york"][:10],
            "section": article["section"], "url": article.get("url", ""), "topic_id": topic_id,
            "topic_name": item["topic_name"], "LDA_probability": float(article[f"topic_{topic_id}_probability"]),
            "relevant": item["relevant"], "subissue": item["subissue"], "sentiment_target": item["sentiment_target"],
            "topic_sentiment_label": item["topic_sentiment_label"], "topic_sentiment_score": item["topic_sentiment_score"],
            "topic_sentiment_evidence": item["topic_sentiment_evidence"],
            "democrat_stance_label": item["democrat_stance_label"], "democrat_stance_score": item["democrat_stance_score"],
            "democrat_evidence": item["democrat_evidence"],
            "republican_stance_label": item["republican_stance_label"], "republican_stance_score": item["republican_stance_score"],
            "republican_evidence": item["republican_evidence"], "speaker_type": item["speaker_type"],
            "speaker_name": item["speaker_name"], "ambiguity_note": item["ambiguity_note"],
            "model": config["api"]["model"], "prompt_version": config["api"]["prompt_version"],
            "topic_definition_version": config["api"]["topic_definition_version"],
            "processing_status": result["status"], "evidence_validation_status": all(relevant_checks),
            "topic_sentiment_evidence_valid": checks.get(f"T{topic_id}_sentiment", True),
            "democrat_evidence_valid": checks.get(f"T{topic_id}_democrat", True),
            "republican_evidence_valid": checks.get(f"T{topic_id}_republican", True),
            "cache_key": result["cache_key"], "processed_at_utc": result["processed_at_utc"], "processing_error": "",
        })
    return rows


async def score_articles(selected: list[dict[str, Any]], inputs: Inputs, config: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    output_dir = root / config["project"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    load_dotenv(root / ".env")
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured in the environment or .env")
    api = config["api"]
    prompt = (root / config["project"]["prompt_file"]).read_text(encoding="utf-8")
    addendum = config["project"].get("prompt_addendum_file")
    if addendum:
        prompt += "\n\n" + (root / addendum).read_text(encoding="utf-8")
    budget = RunBudget(int(api["max_requests"]), float(api["max_cost_usd"]))
    semaphore = asyncio.Semaphore(int(api["concurrency"]))
    log_lock = asyncio.Lock()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    timeout = httpx.Timeout(float(api["timeout_seconds"]))
    async with httpx.AsyncClient(base_url=api["base_url"], headers=headers, timeout=timeout) as client:
        async def limited(article: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            async with semaphore:
                return article, await annotate_one(client, article, inputs.topics, prompt, config, output_dir, budget, log_lock)
        tasks = [asyncio.create_task(limited(article)) for article in selected]
        results: list[dict[str, Any]] = []
        completed = 0
        for task in asyncio.as_completed(tasks):
            article, result = await task
            results.extend(flatten_result(article, inputs.topics, result, config))
            completed += 1
            if completed == 1 or completed % 10 == 0 or completed == len(tasks):
                print(f"Scored {completed}/{len(tasks)} articles; new requests={budget.requests}; current-run cost=${budget.cost:.4f}", flush=True)
    return sorted(results, key=lambda row: (row["article_id"], int(row["topic_id"])))


def dry_run(inputs: Inputs, config: dict[str, Any], root: Path) -> dict[str, Any]:
    output_dir = root / config["project"]["output_dir"]
    frame = inputs.probabilities.copy()
    frame["dominant_topic"] = frame[TOPIC_COLUMNS].astype(float).values.argmax(axis=1)
    trial = trial_articles(inputs.joined, int(config["api"]["trial_size"]))
    estimated_chars = sum(len(user_message(row, inputs.topics)) for row in inputs.joined)
    selection_report = load_json(root / "data/lda_policy/policy_sample_selection_report.json")
    report = {
        "status": "dry_run_no_api_calls",
        "generated_at_utc": iso_utc(),
        "input_audit": inputs.audit,
        "scope_counts_from_existing_report": selection_report["policy_scope_counts"],
        "scope_count_sum": sum(selection_report["policy_scope_counts"].values()),
        "raw_input_articles": selection_report["total_input_articles"],
        "scope_count_difference": selection_report["total_input_articles"] - sum(selection_report["policy_scope_counts"].values()),
        "planned_api_requests": len(inputs.joined),
        "planned_article_topic_rows": len(inputs.joined) * 6,
        "trial_articles": len(trial),
        "trial_article_ids": [row["article_id"] for row in trial],
        "dominant_topic_counts": {f"T{int(k)}": int(v) for k, v in frame["dominant_topic"].value_counts().sort_index().items()},
        "trial_section_counts": dict(Counter(row["section"] for row in trial)),
        "approximate_total_input_tokens_chars_div_4": round(estimated_chars / 4),
        "model": config["api"]["model"], "thinking": "disabled",
        "prompt_version": config["api"]["prompt_version"],
        "topic_definition_version": config["api"]["topic_definition_version"],
        "request_limit": config["api"]["max_requests"], "cost_limit_usd": config["api"]["max_cost_usd"],
    }
    write_json(output_dir / "dry_run_report.json", report)
    plan_rows = [{
        "article_id": row["article_id"], "title": row["title"], "published_at_new_york": row["published_at_new_york"],
        "section": row["section"], "dominant_topic": int(np.argmax([float(row[col]) for col in TOPIC_COLUMNS])),
        "body_characters": len(row["body_text"]), "selected_for_trial": row["article_id"] in set(report["trial_article_ids"]),
    } for row in inputs.joined]
    write_csv(output_dir / "request_plan.csv", plan_rows)
    return report


def sample_review(rows: list[dict[str, Any]], root: Path, config: dict[str, Any]) -> None:
    output_dir = root / config["project"]["output_dir"]
    review_rows = []
    for row in rows:
        review_rows.append({
            **row,
            "human_relevant": "", "human_topic_sentiment_label": "",
            "human_democrat_stance_label": "", "human_republican_stance_label": "",
            "evidence_verified_by_human": "", "human_review_note": "",
        })
    write_csv(output_dir / "sample_review.csv", review_rows)
    template_fields = [
        "article_id", "topic_id", "topic_name", "human_relevant", "human_topic_sentiment_label",
        "human_democrat_stance_label", "human_republican_stance_label", "evidence_verified_by_human",
        "human_review_note",
    ]
    write_csv(output_dir / "human_validation_template.csv", [{field: item.get(field, "") for field in template_fields} for item in review_rows])


def read_scored(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding=CSV_ENCODING)
    for column in ("topic_sentiment_score", "democrat_stance_score", "republican_stance_score", "LDA_probability"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["topic_id"] = pd.to_numeric(frame["topic_id"], errors="raise").astype(int)
    frame["relevant"] = frame["relevant"].map(lambda value: str(value).strip().lower() == "true" if pd.notna(value) else False)
    return frame


def section_groups(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    return [
        ("All", frame),
        ("Opinion", frame.loc[frame["section"] == "Opinion"]),
        ("Non-Opinion", frame.loc[frame["section"] != "Opinion"]),
    ]


def attention_rows(probabilities: pd.DataFrame, topics: list[dict[str, Any]], config: dict[str, Any], frequency: str) -> list[dict[str, Any]]:
    frame = probabilities.copy()
    frame["new_york_date"] = pd.to_datetime(frame["published_at_new_york"]).dt.date
    if frequency == "daily":
        start = date.fromisoformat(config["project"]["start_new_york"])
        end = date.fromisoformat(config["project"]["end_new_york"])
        periods = [(start + timedelta(days=i), start + timedelta(days=i), start + timedelta(days=i), False)
                   for i in range((end - start).days + 1)]
    else:
        observed_start = date.fromisoformat(config["project"]["start_new_york"])
        observed_end = date.fromisoformat(config["project"]["end_new_york"])
        first = observed_start - timedelta(days=observed_start.weekday())
        periods = []
        current = first
        while current <= observed_end:
            week_end = current + timedelta(days=6)
            periods.append((current, max(current, observed_start), min(week_end, observed_end), current < observed_start or week_end > observed_end))
            current += timedelta(days=7)
    rows: list[dict[str, Any]] = []
    for period_key, observed_from, observed_to, partial in periods:
        period_frame = frame.loc[(frame["new_york_date"] >= observed_from) & (frame["new_york_date"] <= observed_to)]
        for section_group, group in section_groups(period_frame):
            for topic_id, topic in enumerate(topics):
                count = len(group)
                weight = float(group[f"topic_{topic_id}_probability"].sum()) if count else 0.0
                rows.append({
                    "date" if frequency == "daily" else "week_start": period_key.isoformat(),
                    **({} if frequency == "daily" else {"week_end": (period_key + timedelta(days=6)).isoformat(),
                       "observed_from": observed_from.isoformat(), "observed_to": observed_to.isoformat(), "partial_week": partial}),
                    "topic_id": topic_id, "topic_name": topic["topic_name"], "section_group": section_group,
                    "article_count": count, "topic_probability_sum": weight,
                    "attention_share": weight / count if count else np.nan,
                    "low_sample": count < int(config["aggregation"]["low_sample_threshold"]),
                })
    return rows


def period_spec(config: dict[str, Any], frequency: str) -> list[tuple[date, date, date, bool]]:
    start = date.fromisoformat(config["project"]["start_new_york"])
    end = date.fromisoformat(config["project"]["end_new_york"])
    if frequency == "daily":
        return [(start + timedelta(days=i), start + timedelta(days=i), start + timedelta(days=i), False)
                for i in range((end - start).days + 1)]
    first = start - timedelta(days=start.weekday())
    result = []
    current = first
    while current <= end:
        week_end = current + timedelta(days=6)
        result.append((current, max(current, start), min(week_end, end), current < start or week_end > end))
        current += timedelta(days=7)
    return result


def sentiment_aggregate(scored: pd.DataFrame, topics: list[dict[str, Any]], config: dict[str, Any], frequency: str) -> list[dict[str, Any]]:
    frame = scored.copy()
    frame["date_value"] = pd.to_datetime(frame["new_york_date"]).dt.date
    rows: list[dict[str, Any]] = []
    threshold = int(config["aggregation"]["low_sample_threshold"])
    for key, observed_from, observed_to, partial in period_spec(config, frequency):
        period = frame.loc[(frame["date_value"] >= observed_from) & (frame["date_value"] <= observed_to)]
        for section_group, section_frame in section_groups(period):
            for topic_id, topic in enumerate(topics):
                subset = section_frame.loc[section_frame["topic_id"] == topic_id]
                total_articles = len(subset)
                total_weight = float(subset["LDA_probability"].sum()) if total_articles else 0.0
                valid_api = subset["processing_status"].isin(SUCCESS_STATUSES)
                eligible = valid_api & subset["relevant"] & subset["topic_sentiment_score"].notna() & subset["topic_sentiment_evidence_valid"].astype(bool)
                effective = subset.loc[eligible]
                effective_weight = float(effective["LDA_probability"].sum())
                numerator = float((effective["LDA_probability"] * effective["topic_sentiment_score"]).sum())
                label_counts = effective["topic_sentiment_label"].value_counts()
                rows.append({
                    "date" if frequency == "daily" else "week_start": key.isoformat(),
                    **({} if frequency == "daily" else {"week_end": (key + timedelta(days=6)).isoformat(),
                       "observed_from": observed_from.isoformat(), "observed_to": observed_to.isoformat(), "partial_week": partial}),
                    "topic_id": topic_id, "topic_name": topic["topic_name"], "section_group": section_group,
                    "total_lda_articles": total_articles, "total_topic_weight": total_weight,
                    "topic_sentiment_index": numerator / effective_weight if effective_weight > 0 else np.nan,
                    "effective_articles": len(effective), "effective_weight": effective_weight,
                    "positive_count": int(label_counts.get("positive", 0)), "negative_count": int(label_counts.get("negative", 0)),
                    "neutral_count": int(label_counts.get("neutral", 0)), "mixed_count": int(label_counts.get("mixed", 0)),
                    "unclear_count": int((valid_api & subset["relevant"] & subset["topic_sentiment_label"].eq("unclear")).sum()),
                    "irrelevant_count": int((valid_api & ~subset["relevant"]).sum()),
                    "api_failed_count": int((~valid_api).sum()), "missing_count": total_articles - len(effective),
                    "article_coverage": len(effective) / total_articles if total_articles else np.nan,
                    "weight_coverage": effective_weight / total_weight if total_weight > 0 else np.nan,
                    "low_sample": len(effective) < threshold,
                })
    return rows


def party_aggregate(scored: pd.DataFrame, topics: list[dict[str, Any]], config: dict[str, Any], frequency: str) -> list[dict[str, Any]]:
    frame = scored.copy()
    frame["date_value"] = pd.to_datetime(frame["new_york_date"]).dt.date
    rows: list[dict[str, Any]] = []
    threshold = int(config["aggregation"]["low_sample_threshold"])
    for key, observed_from, observed_to, partial in period_spec(config, frequency):
        period = frame.loc[(frame["date_value"] >= observed_from) & (frame["date_value"] <= observed_to)]
        for section_group, section_frame in section_groups(period):
            for topic_id, topic in enumerate(topics):
                subset = section_frame.loc[section_frame["topic_id"] == topic_id]
                total_articles = len(subset); total_weight = float(subset["LDA_probability"].sum()) if total_articles else 0.0
                valid_api = subset["processing_status"].isin(SUCCESS_STATUSES)
                row: dict[str, Any] = {
                    "date" if frequency == "daily" else "week_start": key.isoformat(),
                    **({} if frequency == "daily" else {"week_end": (key + timedelta(days=6)).isoformat(),
                       "observed_from": observed_from.isoformat(), "observed_to": observed_to.isoformat(), "partial_week": partial}),
                    "topic_id": topic_id, "topic_name": topic["topic_name"], "section_group": section_group,
                    "total_lda_articles": total_articles, "total_topic_weight": total_weight,
                }
                for party in ("democrat", "republican"):
                    score_col, label_col = f"{party}_stance_score", f"{party}_stance_label"
                    evidence_col = f"{party}_evidence_valid"
                    eligible = valid_api & subset["relevant"] & subset[score_col].notna() & subset[evidence_col].astype(bool)
                    effective = subset.loc[eligible]
                    weight = float(effective["LDA_probability"].sum())
                    numerator = float((effective["LDA_probability"] * effective[score_col]).sum())
                    counts = effective[label_col].value_counts()
                    row.update({
                        f"{party}_stance_index": numerator / weight if weight > 0 else np.nan,
                        f"{party}_effective_articles": len(effective), f"{party}_effective_weight": weight,
                        f"{party}_support_count": int(counts.get("support", 0)), f"{party}_oppose_count": int(counts.get("oppose", 0)),
                        f"{party}_neutral_count": int(counts.get("neutral", 0)), f"{party}_mixed_count": int(counts.get("mixed", 0)),
                        f"{party}_not_expressed_count": int((valid_api & subset["relevant"] & subset[label_col].eq("not_expressed")).sum()),
                        f"{party}_unclear_count": int((valid_api & subset["relevant"] & subset[label_col].eq("unclear")).sum()),
                        f"{party}_missing_count": total_articles - len(effective),
                        f"{party}_article_coverage": len(effective) / total_articles if total_articles else np.nan,
                        f"{party}_weight_coverage": weight / total_weight if total_weight > 0 else np.nan,
                        f"{party}_low_sample": len(effective) < threshold,
                    })
                row["api_failed_count"] = int((~valid_api).sum())
                rows.append(row)
    return rows


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arialbd.ttf" if bold else "arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_panel_series(
    path: Path, title: str, topics: list[dict[str, Any]], dates: list[str],
    series_by_topic: dict[int, list[tuple[str, list[float | None], str]]],
    low_by_topic: dict[int, list[bool]], note: str, y_min: float, y_max: float,
) -> None:
    width, panel_height = 1900, 230
    height = 170 + panel_height * 6 + 110
    image = Image.new("RGB", (width, height), "white"); draw = ImageDraw.Draw(image)
    title_font, regular, small = load_font(27, True), load_font(17), load_font(14)
    draw.text((45, 25), title, font=title_font, fill="#111111")
    left, right = 145, 1835
    colors = ["#24527A", "#B24C3D", "#5A7D3A"]
    for topic_id, topic in enumerate(topics):
        top = 105 + topic_id * panel_height; bottom = top + 155
        draw.text((45, top - 25), f"T{topic_id}: {topic['english_name']}", font=regular, fill="#222222")
        zero_y = bottom - (0 - y_min) / (y_max - y_min) * (bottom - top)
        draw.line((left, top, left, bottom), fill="#333333", width=2)
        draw.line((left, bottom, right, bottom), fill="#333333", width=2)
        if y_min < 0 < y_max:
            draw.line((left, zero_y, right, zero_y), fill="#BBBBBB", width=1)
        draw.text((55, top - 5), f"{y_max:.1f}", font=small, fill="#666666")
        draw.text((55, bottom - 15), f"{y_min:.1f}", font=small, fill="#666666")
        for series_index, (label, values, color) in enumerate(series_by_topic.get(topic_id, [])):
            segments: list[list[tuple[float, float]]] = [[]]
            for idx, value in enumerate(values):
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    if segments[-1]: segments.append([])
                    continue
                x = left + idx * (right - left) / max(1, len(dates) - 1)
                y = bottom - (float(value) - y_min) / (y_max - y_min) * (bottom - top)
                segments[-1].append((x, y))
                if series_index == 0 and low_by_topic.get(topic_id, [False] * len(dates))[idx]:
                    draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="#D97706")
            for segment in segments:
                if len(segment) >= 2: draw.line(segment, fill=color or colors[series_index], width=3)
                elif len(segment) == 1: draw.ellipse((segment[0][0] - 2, segment[0][1] - 2, segment[0][0] + 2, segment[0][1] + 2), fill=color)
        if topic_id == 0:
            x = right - 420
            for label, _, color in series_by_topic.get(topic_id, []):
                draw.line((x, top - 18, x + 35, top - 18), fill=color, width=4)
                draw.text((x + 43, top - 28), label, font=small, fill="#333333"); x += 190
    for idx in (0, len(dates) // 2, len(dates) - 1):
        x = left + idx * (right - left) / max(1, len(dates) - 1)
        draw.text((x - 45, 105 + 6 * panel_height), dates[idx], font=small, fill="#333333")
    draw.text((45, height - 62), note, font=small, fill="#555555")
    path.parent.mkdir(parents=True, exist_ok=True); image.save(path)


def make_figures(output_dir: Path, figure_dir: Path, topics: list[dict[str, Any]]) -> None:
    attention = pd.read_csv(output_dir / "daily_policy_attention.csv", encoding=CSV_ENCODING)
    sentiment = pd.read_csv(output_dir / "daily_policy_sentiment.csv", encoding=CSV_ENCODING)
    party = pd.read_csv(output_dir / "daily_policy_party_stance.csv", encoding=CSV_ENCODING)
    attention = attention.loc[attention["section_group"] == "All"]
    sentiment = sentiment.loc[sentiment["section_group"] == "All"]
    party = party.loc[party["section_group"] == "All"]
    dates = sorted(attention["date"].unique().tolist())
    attention_series, sentiment_series, party_series = {}, {}, {}
    attention_low, sentiment_low, party_low = {}, {}, {}
    for topic_id in range(6):
        a = attention.loc[attention["topic_id"] == topic_id].set_index("date").reindex(dates)
        s = sentiment.loc[sentiment["topic_id"] == topic_id].set_index("date").reindex(dates)
        p = party.loc[party["topic_id"] == topic_id].set_index("date").reindex(dates)
        attention_series[topic_id] = [("Attention", a["attention_share"].tolist(), "#24527A")]
        sentiment_series[topic_id] = [("Topic sentiment", s["topic_sentiment_index"].tolist(), "#24527A")]
        party_series[topic_id] = [("Democrat", p["democrat_stance_index"].tolist(), "#2878B5"),
                                  ("Republican", p["republican_stance_index"].tolist(), "#C44E52")]
        attention_low[topic_id] = a["low_sample"].fillna(True).astype(bool).tolist()
        sentiment_low[topic_id] = s["low_sample"].fillna(True).astype(bool).tolist()
        party_low[topic_id] = (p["democrat_low_sample"].fillna(True) | p["republican_low_sample"].fillna(True)).astype(bool).tolist()
    draw_panel_series(figure_dir / "daily_policy_attention.png", "Guardian policy-topic attention (daily)", topics, dates,
                      attention_series, attention_low, "Orange points: fewer than 3 articles. Missing days are not filled with zero.", 0, 1)
    draw_panel_series(figure_dir / "daily_policy_sentiment.png", "Guardian policy-topic sentiment presented in text (daily)", topics, dates,
                      sentiment_series, sentiment_low, "Range -1 to +1. Orange points: fewer than 3 effective articles; gaps are missing.", -1, 1)
    draw_panel_series(figure_dir / "daily_policy_party_stance.png", "Topic-specific party stance presented in Guardian text (daily)", topics, dates,
                      party_series, party_low, "Party series use separate covered samples; they must not be subtracted as a primary index.", -1, 1)


def aggregate_all(inputs: Inputs, config: dict[str, Any], root: Path) -> None:
    output_dir = root / config["project"]["output_dir"]
    figure_dir = root / config["project"]["figure_dir"]
    scored = read_scored(output_dir / "article_policy_sentiment.csv")
    daily_attention = attention_rows(inputs.probabilities, inputs.topics, config, "daily")
    weekly_attention = attention_rows(inputs.probabilities, inputs.topics, config, "weekly")
    daily_sentiment = sentiment_aggregate(scored, inputs.topics, config, "daily")
    weekly_sentiment = sentiment_aggregate(scored, inputs.topics, config, "weekly")
    daily_party = party_aggregate(scored, inputs.topics, config, "daily")
    weekly_party = party_aggregate(scored, inputs.topics, config, "weekly")
    write_csv(output_dir / "daily_policy_attention.csv", daily_attention)
    write_csv(output_dir / "weekly_policy_attention.csv", weekly_attention)
    write_csv(output_dir / "daily_policy_sentiment.csv", daily_sentiment)
    write_csv(output_dir / "weekly_policy_sentiment.csv", weekly_sentiment)
    write_csv(output_dir / "daily_policy_party_stance.csv", daily_party)
    write_csv(output_dir / "weekly_policy_party_stance.csv", weekly_party)
    make_figures(output_dir, figure_dir, inputs.topics)


def usage_summary(inputs: Inputs, config: dict[str, Any], root: Path) -> dict[str, Any]:
    output_dir = root / config["project"]["output_dir"]
    totals = Counter(); total_cost = 0.0; found = 0
    for article in inputs.joined:
        key, _ = cache_key(article, config)
        path = output_dir / "cache" / f"{key}.json"
        if not path.exists(): continue
        record = load_json(path); usage = record.get("usage", {}) or {}; found += 1
        for field in ("prompt_tokens", "completion_tokens", "total_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens"):
            totals[field] += int(usage.get(field, 0) or 0)
        total_cost += float(record.get("cost_usd", 0) or 0)
    all_cache_cost = 0.0; all_cache_files = 0
    for path in (output_dir / "cache").glob("*.json"):
        record = load_json(path); all_cache_files += 1
        all_cache_cost += float(record.get("cost_usd", 0) or 0)
    ledger_rows: list[dict[str, Any]] = []
    ledger_path = output_dir / "api_request_ledger.jsonl"
    if ledger_path.exists():
        with ledger_path.open(encoding="utf-8") as handle:
            ledger_rows = [json.loads(line) for line in handle if line.strip()]
    current_version = config["api"]["prompt_version"]
    current_ledger = [row for row in ledger_rows if row.get("prompt_version") == current_version]
    retry_rows = [row for row in current_ledger if row.get("status") in {"retry", "failed"}]
    metered_retry_cost = sum(float(row.get("cost_usd", 0) or 0) for row in retry_rows)
    unmetered_http200_retries = [row for row in retry_rows if row.get("http_status") == 200 and not row.get("usage")]
    current_cost_by_article: dict[str, float] = {}
    for article in inputs.joined:
        key, _ = cache_key(article, config); path = output_dir / "cache" / f"{key}.json"
        if path.exists(): current_cost_by_article[article["article_id"]] = float(load_json(path).get("cost_usd", 0) or 0)
    fallback_mean = float(np.mean(list(current_cost_by_article.values()))) if current_cost_by_article else 0.0
    estimated_unmetered_retry_cost = sum(current_cost_by_article.get(row.get("article_id", ""), fallback_mean) for row in unmetered_http200_retries)
    summary = {
        "generated_at_utc": iso_utc(), "model": config["api"]["model"], "thinking": "disabled",
        "prompt_version": config["api"]["prompt_version"], "topic_definition_version": config["api"]["topic_definition_version"],
        "successful_cached_articles": found, **dict(totals), "estimated_actual_cost_usd": total_cost,
        "validated_production_cache_cost_usd": total_cost,
        "all_prompt_versions_successful_cache_files": all_cache_files,
        "all_prompt_versions_successful_cache_cost_usd": all_cache_cost,
        "current_prompt_retry_or_failed_attempts": len(retry_rows),
        "metered_retry_cost_usd": metered_retry_cost,
        "unmetered_http200_validation_retries_before_usage_logging_fix": len(unmetered_http200_retries),
        "estimated_unmetered_validation_retry_cost_usd": estimated_unmetered_retry_cost,
        "estimated_total_project_api_cost_usd": all_cache_cost + metered_retry_cost + estimated_unmetered_retry_cost,
        "cost_reporting_note": "Production-cache tokens and cost are exact from API usage. Some early validation-retry responses were not persisted; their cost is estimated from the same article's validated request and is reported separately.",
        "pricing_source": "DeepSeek official Models & Pricing page; per-request peak/off-peak based on UTC weekday/time",
        "holiday_pricing_limitation": "Chinese public holidays are off-peak; the automatic estimate does not include a holiday calendar.",
    }
    write_json(output_dir / "api_usage_summary.json", summary)
    return summary


def validate_outputs(inputs: Inputs, config: dict[str, Any], root: Path) -> dict[str, Any]:
    output_dir = root / config["project"]["output_dir"]
    scored = read_scored(output_dir / "article_policy_sentiment.csv")
    duplicate_keys = int(scored.duplicated(["article_id", "topic_id"]).sum())
    expected_ids = set(inputs.probabilities["article_id"]); actual_ids = set(scored["article_id"])
    scores_valid = all(scored[col].dropna().isin([-1, 0, 1]).all() for col in ("topic_sentiment_score", "democrat_stance_score", "republican_stance_score"))
    failed_articles = scored.loc[scored["processing_status"] == "failed", "article_id"].nunique()
    evidence_invalid = int((scored["processing_status"].isin(SUCCESS_STATUSES) & ~scored["evidence_validation_status"].astype(bool)).sum())
    daily_s = pd.read_csv(output_dir / "daily_policy_sentiment.csv", encoding=CSV_ENCODING)
    weekly_s = pd.read_csv(output_dir / "weekly_policy_sentiment.csv", encoding=CSV_ENCODING)
    daily_p = pd.read_csv(output_dir / "daily_policy_party_stance.csv", encoding=CSV_ENCODING)
    range_ok = all(daily_s["topic_sentiment_index"].dropna().between(-1, 1)) and all(daily_p["democrat_stance_index"].dropna().between(-1, 1)) and all(daily_p["republican_stance_index"].dropna().between(-1, 1))

    candidates = daily_s.loc[(daily_s["section_group"] == "All") & (daily_s["effective_articles"] > 0)]
    manual_daily: dict[str, Any] = {"status": "not_available"}
    manual_weekly: dict[str, Any] = {"status": "not_available"}
    if len(candidates):
        target = candidates.sort_values(["effective_articles", "date"], ascending=[False, True]).iloc[0]
        subset = scored.loc[(scored["new_york_date"] == target["date"]) & (scored["topic_id"] == int(target["topic_id"])) &
                            scored["processing_status"].isin(SUCCESS_STATUSES) & scored["relevant"] & scored["topic_sentiment_score"].notna()]
        weight = float(subset["LDA_probability"].sum()); value = float((subset["LDA_probability"] * subset["topic_sentiment_score"]).sum() / weight)
        manual_daily = {"status": "pass" if abs(value - float(target["topic_sentiment_index"])) < 1e-12 else "fail",
                        "date": target["date"], "topic_id": int(target["topic_id"]), "independent_value": value,
                        "output_value": float(target["topic_sentiment_index"]), "effective_articles": len(subset), "effective_weight": weight}
        week_start = (date.fromisoformat(target["date"]) - timedelta(days=date.fromisoformat(target["date"]).weekday())).isoformat()
        week_target = weekly_s.loc[(weekly_s["week_start"] == week_start) & (weekly_s["topic_id"] == int(target["topic_id"])) & (weekly_s["section_group"] == "All")].iloc[0]
        start, end = date.fromisoformat(week_target["observed_from"]), date.fromisoformat(week_target["observed_to"])
        date_values = pd.to_datetime(scored["new_york_date"]).dt.date
        week_subset = scored.loc[(date_values >= start) & (date_values <= end) & (scored["topic_id"] == int(target["topic_id"])) &
                                 scored["processing_status"].isin(SUCCESS_STATUSES) & scored["relevant"] & scored["topic_sentiment_score"].notna()]
        week_weight = float(week_subset["LDA_probability"].sum())
        week_value = float((week_subset["LDA_probability"] * week_subset["topic_sentiment_score"]).sum() / week_weight)
        manual_weekly = {"status": "pass" if abs(week_value - float(week_target["topic_sentiment_index"])) < 1e-12 else "fail",
                         "week_start": week_start, "topic_id": int(target["topic_id"]), "independent_value": week_value,
                         "output_value": float(week_target["topic_sentiment_index"]), "effective_articles": len(week_subset), "effective_weight": week_weight}

    all_daily = daily_s.loc[daily_s["section_group"] == "All"]
    coverage = []
    party_daily = daily_p.loc[daily_p["section_group"] == "All"]
    for topic_id, topic in enumerate(inputs.topics):
        s_rows = all_daily.loc[all_daily["topic_id"] == topic_id]
        p_rows = party_daily.loc[party_daily["topic_id"] == topic_id]
        topic_rows = scored.loc[scored["topic_id"] == topic_id]
        valid_api = topic_rows["processing_status"].isin(SUCCESS_STATUSES)
        sentiment_valid = valid_api & topic_rows["relevant"] & topic_rows["topic_sentiment_score"].notna() & topic_rows["topic_sentiment_evidence_valid"].astype(bool)
        democrat_valid = valid_api & topic_rows["relevant"] & topic_rows["democrat_stance_score"].notna() & topic_rows["democrat_evidence_valid"].astype(bool)
        republican_valid = valid_api & topic_rows["relevant"] & topic_rows["republican_stance_score"].notna() & topic_rows["republican_evidence_valid"].astype(bool)
        total_weight = float(topic_rows["LDA_probability"].sum())
        coverage.append({
            "topic_id": topic_id, "topic_name": topic["topic_name"],
            "total_policy_articles": len(topic_rows), "total_topic_weight": total_weight,
            "sentiment_effective_articles": int(sentiment_valid.sum()),
            "sentiment_article_coverage": float(sentiment_valid.mean()),
            "sentiment_weight_coverage": float(topic_rows.loc[sentiment_valid, "LDA_probability"].sum() / total_weight),
            "sentiment_mean_article_coverage_on_article_days": float(s_rows.loc[s_rows["total_lda_articles"] > 0, "article_coverage"].mean()),
            "sentiment_mean_weight_coverage_on_article_days": float(s_rows.loc[s_rows["total_topic_weight"] > 0, "weight_coverage"].mean()),
            "sentiment_low_sample_days_with_articles": int(((s_rows["total_lda_articles"] > 0) & s_rows["low_sample"].astype(bool)).sum()),
            "days_with_policy_articles": int((s_rows["total_lda_articles"] > 0).sum()),
            "democrat_effective_articles": int(democrat_valid.sum()),
            "democrat_article_coverage": float(democrat_valid.mean()),
            "democrat_weight_coverage": float(topic_rows.loc[democrat_valid, "LDA_probability"].sum() / total_weight),
            "republican_effective_articles": int(republican_valid.sum()),
            "republican_article_coverage": float(republican_valid.mean()),
            "republican_weight_coverage": float(topic_rows.loc[republican_valid, "LDA_probability"].sum() / total_weight),
        })
    write_csv(output_dir / "topic_coverage_summary.csv", coverage)
    status = "pass" if (len(scored) == 1356 and duplicate_keys == 0 and expected_ids == actual_ids and scores_valid and range_ok and failed_articles == 0 and evidence_invalid == 0 and manual_daily["status"] == "pass" and manual_weekly["status"] == "pass") else "review_required"
    report = {
        "validation_status": status, "article_topic_rows": len(scored), "expected_article_topic_rows": 1356,
        "unique_articles": int(scored["article_id"].nunique()), "unique_key_duplicates": duplicate_keys,
        "article_ids_aligned": expected_ids == actual_ids, "scores_in_allowed_range": scores_valid,
        "aggregate_indices_in_range": range_ok, "failed_articles": int(failed_articles),
        "evidence_validation_failed_rows": evidence_invalid,
        "daily_dates": int(daily_s["date"].nunique()), "october_9_all_missing": bool(all_daily.loc[all_daily["date"] == "2026-10-09", "topic_sentiment_index"].isna().all()),
        "daily_manual_recalculation": manual_daily, "weekly_manual_recalculation": manual_weekly,
        "mixed_sensitivity_analysis": "not_executed; frozen-model inference entry is configured but disabled",
    }
    write_json(output_dir / "policy_sentiment_validation_report.json", report)
    return report


def update_readme(root: Path) -> None:
    path = root / "README.md"
    marker = "## DeepSeek政策情绪与两党态度"
    current = path.read_text(encoding="utf-8")
    if marker in current:
        return
    addition = """

## DeepSeek政策情绪与两党态度

本阶段固定使用K=6实质政策LDA，不重新训练模型。程序通过`article_id`连接226篇政策文章的原始正文与完整主题概率；每篇文章只调用一次DeepSeek，并在同一请求中评价全部六个主题。

```powershell
conda activate assignment5_guardian_lda
python score_policy_sentiment.py --stage dry-run
python score_policy_sentiment.py --stage trial
python score_policy_sentiment.py --stage full
python score_policy_sentiment.py --stage aggregate
python score_policy_sentiment.py --stage validate
```

密钥从`.env`或环境变量`DEEPSEEK_API_KEY`读取，不写入缓存、日志或输出。`trial`默认分层处理30篇；`full`会复用有效缓存，只请求剩余文章。可通过`--concurrency`、`--max-articles`、`--max-requests`和`--max-cost-usd`限制运行。缓存键包含article_id、正文哈希、主题定义版本、提示词版本和模型，修改规则时必须提升版本。

输出位于`data/policy_sentiment/`，图表位于`figures/policy_sentiment/`。日频保留92天，周频按纽约时间周一至周日直接汇集文章重新加权。缺失、not_expressed、unclear和API失败不填零。两党指数的覆盖样本分别保存，不能直接相减解释为相对党派优势。
"""
    path.write_text(current + addition, encoding="utf-8")


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    for name in ("concurrency", "max_requests", "max_cost_usd"):
        value = getattr(args, name)
        if value is not None:
            config["api"][name] = value
    if args.sample_size is not None:
        config["api"]["trial_size"] = args.sample_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="policy_sentiment_config.yaml")
    parser.add_argument("--stage", choices=["dry-run", "trial", "full", "aggregate", "validate", "all"], default="dry-run")
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--max-articles", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--max-cost-usd", type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args(); root = Path(__file__).resolve().parent
    config = yaml.safe_load((root / args.config).read_text(encoding="utf-8")); apply_cli_overrides(config, args)
    inputs = read_inputs(config, root); output_dir = root / config["project"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True); update_readme(root)
    if args.stage in {"dry-run", "all"}:
        report = dry_run(inputs, config, root)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        if args.stage == "dry-run": return 0
    if args.stage in {"trial", "all"}:
        selected = trial_articles(inputs.joined, int(config["api"]["trial_size"]))
        if args.max_articles is not None: selected = selected[:args.max_articles]
        rows = asyncio.run(score_articles(selected, inputs, config, root))
        sample_review(rows, root, config)
        print(f"Trial output: {len(rows)} article-topic rows", flush=True)
        if args.stage == "trial":
            usage_summary(inputs, config, root)
            return 0
    if args.stage in {"full", "all"}:
        selected = inputs.joined[:args.max_articles] if args.max_articles is not None else inputs.joined
        rows = asyncio.run(score_articles(selected, inputs, config, root))
        write_csv(output_dir / "article_policy_sentiment.csv", rows)
        write_jsonl(output_dir / "article_policy_sentiment.jsonl", rows)
        failed = [row for row in rows if row["processing_status"] == "failed"]
        write_jsonl(output_dir / "api_failures.jsonl", failed)
        usage_summary(inputs, config, root)
        print(f"Full output: {len(rows)} article-topic rows; failed article rows={len(failed)}", flush=True)
        if args.stage == "full": return 0
    if args.stage in {"aggregate", "all"}:
        aggregate_all(inputs, config, root)
        print("Daily, weekly, section-group aggregates and figures written", flush=True)
        if args.stage == "aggregate": return 0
    if args.stage in {"validate", "all"}:
        report = validate_outputs(inputs, config, root)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0 if report["validation_status"] == "pass" else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
