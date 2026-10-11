#!/usr/bin/env python3
"""Run reproducible LDA topic analysis on the collected Guardian corpus.

Stages:
  selection  - input audit and deterministic/manual sample screening
  candidates - selection, spaCy/Gensim preprocessing, and K comparison
  final      - final-model outputs from saved candidate artifacts
  all        - candidates followed by final

The analysis is descriptive and retrospective. It is not a real-time forecast
and does not measure US voter preferences.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
import yaml
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import CoherenceModel, LdaModel
from gensim.models.phrases import Phrases, Phraser
from scipy.optimize import linear_sum_assignment


CSV_ENCODING = "utf-8-sig"
EXPECTED_COLUMNS = {
    "article_id", "title", "body_text", "published_at_utc",
    "published_at_new_york", "section", "relevance_flag",
    "body_missing", "duplicate_body", "url",
}

ELECTION_RE = re.compile(
    r"\b(mid[ -]?terms?|elections?|electoral|primar(?:y|ies)|campaigns?|"
    r"candidates?|ballots?|vot(?:e|er|ers|ing)|polling|redistrict(?:ing)?|"
    r"gerrymander(?:ing)?|runoff|races?)\b",
    re.I,
)
MIDTERM_RE = re.compile(r"\bmid[ -]?terms?(?: elections?)?\b", re.I)
US_POLITICAL_RE = re.compile(
    r"\b(democrats?|democratic party|republicans?|republican party|gop|maga|"
    r"donald trump|trump administration|congress|congressional|"
    r"senators?|house of representatives|"
    r"capitol hill|white house|united states|american voters?)\b",
    re.I,
)
US_CORE_RE = re.compile(
    r"\b(donald trump|trump administration|trump|maga|gop|united states|"
    r"american|white house|capitol hill|house of representatives)\b",
    re.I,
)
US_ABBREV_RE = re.compile(r"\bU\.?S\.?(?=\s|$)")
FEDERAL_INSTITUTION_RE = re.compile(
    r"\b(congress|congressional|senate|senator|senators|u\.?s\.? house|"
    r"house of representatives|representative|representatives|white house|"
    r"supreme court|federal government)\b",
    re.I,
)
POLICY_ACTION_RE = re.compile(
    r"\b(bill|legislation|lawmakers?|hearing|committee|administration|policy|"
    r"court|ruling|funding|budget|tax|tariff|immigration|healthcare|medicare|"
    r"climate|abortion|gun|war|economy|jobs|education|rights|oversight)\b",
    re.I,
)
OBVIOUS_NON_NEWS_TEMPLATE_RE = re.compile(
    r"(?i)\b(sign up to|privacy policy|cookie policy|all rights reserved|"
    r"support the guardian|newsletter promotion)\b"
)
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
HTML_RE = re.compile(r"<[^>]+>")


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a YAML mapping")
    return config


def ensure_dirs(config: dict[str, Any]) -> tuple[Path, Path, Path]:
    output = Path(config["output_dir"])
    models = Path(config["model_dir"])
    figures = Path(config["figure_dir"])
    for path in (output, models, figures):
        path.mkdir(parents=True, exist_ok=True)
    return output, models, figures


def load_jsonl(path: Path, *, validate_input_schema: bool = True) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc.msg}") from exc
            rows.append(row)
    if not rows:
        raise ValueError(f"Input JSONL is empty: {path}")
    if validate_input_schema:
        missing = EXPECTED_COLUMNS - set(rows[0])
        if missing:
            raise ValueError(f"Input is missing required columns: {sorted(missing)}")
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding=CSV_ENCODING, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def clean_for_screening(value: str) -> str:
    text = html.unescape(HTML_RE.sub(" ", value or ""))
    text = URL_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def close_context(text: str, left_re: re.Pattern[str], right_re: re.Pattern[str], window: int = 350) -> bool:
    left = [match.start() for match in left_re.finditer(text)]
    right = [match.start() for match in right_re.finditer(text)]
    return any(abs(a - b) <= window for a in left for b in right)


def deterministic_screen(row: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    title = clean_for_screening(row.get("title", ""))
    summary = clean_for_screening(row.get("summary", ""))
    body = clean_for_screening(row.get("body_text", ""))
    headline_context = f"{title} {summary}".strip()
    lead = body[:3000]
    front = f"{headline_context} {lead}".strip()
    full = f"{headline_context} {body}".strip()

    midterm_front = bool(MIDTERM_RE.search(front))
    midterm_headline = bool(MIDTERM_RE.search(headline_context))
    midterm_full = bool(MIDTERM_RE.search(full))
    election_headline = bool(ELECTION_RE.search(headline_context))
    election_front = bool(ELECTION_RE.search(front))
    us_political_front = bool(US_POLITICAL_RE.search(front) or US_ABBREV_RE.search(front))
    us_core_headline = bool(US_CORE_RE.search(headline_context) or US_ABBREV_RE.search(headline_context))
    us_core_front = bool(US_CORE_RE.search(front) or US_ABBREV_RE.search(front))
    federal_front = bool(FEDERAL_INSTITUTION_RE.search(front))
    federal_headline = bool(FEDERAL_INSTITUTION_RE.search(headline_context))
    policy_front = bool(POLICY_ACTION_RE.search(front))
    policy_headline = bool(POLICY_ACTION_RE.search(headline_context))
    election_party_proximity = close_context(full, ELECTION_RE, US_POLITICAL_RE)
    midterm_party_proximity = close_context(full, MIDTERM_RE, US_POLITICAL_RE, window=600)
    section = row.get("section", "")

    signals = {
        "midterm_front": midterm_front,
        "midterm_headline": midterm_headline,
        "midterm_full": midterm_full,
        "election_headline": election_headline,
        "election_front": election_front,
        "us_political_front": us_political_front,
        "us_core_headline": us_core_headline,
        "us_core_front": us_core_front,
        "federal_front": federal_front,
        "federal_headline": federal_headline,
        "policy_front": policy_front,
        "policy_headline": policy_headline,
        "election_party_proximity": election_party_proximity,
        "midterm_party_proximity": midterm_party_proximity,
    }

    if title.casefold() == "corrections and clarifications":
        return "irrelevant", "corrections index is template content, not a substantive election article", signals

    special_interest = section in {"Film", "Sport", "Football", "Culture", "Books", "Television & radio"}
    historical_headline = bool(re.search(r"\b(19\d{2}|200[0-9]|201[0-9]|202[0-4])\b", headline_context))
    roundup = bool(re.search(r"(?i)\b(first thing|morning briefing|monday briefing|wednesday briefing|morning mail|afternoon update|war briefing)\b", title))
    if special_interest:
        if historical_headline:
            return "irrelevant", "historical/special-interest article is not about the 2026 US midterm context", signals
        if re.search(r"(?i)\b(fifa|infantino|uefa|concacaf|world cup)\b", headline_context) and not (
            midterm_headline or (election_headline and us_core_headline)
        ):
            return "irrelevant", "football-governance election/leadership coverage is not a US midterm subject", signals
        if midterm_headline and us_core_headline:
            return "relevant", "special-interest headline explicitly centers current US midterms", signals
        if election_headline and us_core_headline and election_party_proximity:
            return "relevant", "special-interest coverage directly centers a US election-related subject", signals
        if us_political_front and (midterm_full or election_party_proximity or policy_front):
            return "uncertain", "political connection exists but election relevance is not central; manual review required", signals
        return "irrelevant", "special-interest article lacks a substantive US election connection", signals

    if section != "US news" and election_headline and not us_core_headline and not midterm_headline:
        return "irrelevant", "election coverage concerns another country rather than the US midterms", signals
    if roundup and not (midterm_headline or election_headline):
        if section in {"Australia news", "UK news", "Politics"}:
            return "irrelevant", "non-US multi-story roundup has no central US election focus", signals
        if us_core_front or election_party_proximity:
            return "uncertain", "multi-story roundup has US political material but the headline does not center the election", signals
        return "irrelevant", "multi-story roundup lacks a central US election focus", signals
    if section not in {"US news", "Opinion", "Business", "News"} and not us_core_headline and not midterm_headline:
        if us_core_front or election_party_proximity:
            return "uncertain", "non-US section has a secondary US political connection; reserve for manual review", signals
        return "irrelevant", "non-US section lacks a central US election subject", signals
    if midterm_front and us_core_front:
        return "relevant", "midterm language appears in headline/lead with US political context", signals
    if election_headline and us_core_headline and election_party_proximity:
        return "relevant", "headline/summary centers an election process with US political context", signals
    if election_front and federal_front and us_core_front and election_party_proximity:
        return "relevant", "lead contains election process and US federal/party context", signals
    if midterm_full and us_core_front and midterm_party_proximity and (policy_headline or federal_headline or election_headline):
        return "relevant", "article connects its US political subject to the midterms", signals
    if election_party_proximity and us_core_front and (federal_headline or policy_headline):
        return "relevant", "US federal/policy coverage has an explicit nearby election connection", signals
    if not us_core_front and not election_party_proximity:
        return "irrelevant", "no substantive US political/election context in headline or lead", signals
    if us_core_front and (federal_front or policy_front):
        return "uncertain", "US federal or policy discussion lacks a clear election link; retain for sensitivity review", signals
    if row.get("relevance_flag") == "likely_irrelevant":
        return "irrelevant", "weak legacy keyword hit and no stronger contextual evidence", signals
    return "uncertain", "partial political context; manual review required", signals


@dataclass
class SelectionResult:
    input_rows: list[dict[str, Any]]
    relevant: list[dict[str, Any]]
    uncertain: list[dict[str, Any]]
    irrelevant: list[dict[str, Any]]
    exclusions: list[dict[str, Any]]
    audit: dict[str, Any]


def audit_and_select(config: dict[str, Any], output_dir: Path) -> SelectionResult:
    input_path = Path(config["input_jsonl"])
    rows = load_jsonl(input_path)
    start = date.fromisoformat(config["date_range"]["start_new_york"])
    end = date.fromisoformat(config["date_range"]["end_new_york"])
    selection_cfg = config["selection"]
    overrides: dict[str, Any] = selection_cfg.get("manual_overrides", {})

    relevant: list[dict[str, Any]] = []
    uncertain: list[dict[str, Any]] = []
    irrelevant: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    bad_timestamps: list[dict[str, str]] = []
    boundary_records: list[dict[str, str]] = []
    daily_counts: Counter[str] = Counter()

    for source in rows:
        row = dict(source)
        static_exclusion = ""
        local_date = ""
        try:
            parse_timestamp(row["published_at_utc"])
            ny_dt = parse_timestamp(row["published_at_new_york"])
            local_date = ny_dt.date().isoformat()
            daily_counts[local_date] += 1
            if ny_dt.date() < start or ny_dt.date() > end:
                static_exclusion = "outside_new_york_date_range"
                boundary_records.append({
                    "article_id": row["article_id"],
                    "title": row["title"],
                    "published_at_utc": row["published_at_utc"],
                    "published_at_new_york": row["published_at_new_york"],
                    "url": row["url"],
                })
        except (ValueError, TypeError) as exc:
            static_exclusion = "invalid_timestamp"
            bad_timestamps.append({"article_id": row.get("article_id", ""), "error": str(exc)})

        if row.get("body_missing") or not str(row.get("body_text", "")).strip():
            static_exclusion = "body_missing"
        elif row.get("duplicate_body"):
            static_exclusion = "duplicate_body"

        decision, reason, signals = deterministic_screen(row)
        override = overrides.get(row["article_id"])
        if override:
            proposed = override.get("decision", "")
            if proposed not in {"relevant", "uncertain", "irrelevant"}:
                raise ValueError(f"Invalid manual decision for {row['article_id']}: {proposed}")
            decision = proposed
            reason = f"manual override: {override.get('reason', '').strip()}"
        if static_exclusion:
            decision = "irrelevant"
            reason = static_exclusion

        row["new_york_date"] = local_date
        row["screening_decision"] = decision
        row["screening_reason"] = reason
        row["screening_signals"] = signals
        row["manual_override_applied"] = bool(override)
        if decision == "relevant":
            relevant.append(row)
        elif decision == "uncertain":
            uncertain.append(row)
            exclusions.append(exclusion_row(row, "screened_uncertain_reserved_for_sensitivity", reason))
        else:
            irrelevant.append(row)
            exclusions.append(exclusion_row(row, static_exclusion or "screened_irrelevant", reason))

    audit = {
        "input_file": str(input_path),
        "input_format_used": "JSONL only; CSV was not merged",
        "actual_columns": list(rows[0]),
        "total_articles": len(rows),
        "utc_datetime_range": {
            "min": min(row["published_at_utc"] for row in rows),
            "max": max(row["published_at_utc"] for row in rows),
        },
        "new_york_local_date_range_observed": {
            "min": min(daily_counts),
            "max": max(daily_counts),
        },
        "requested_new_york_local_date_range": {"start": str(start), "end": str(end)},
        "daily_counts_new_york": dict(sorted(daily_counts.items())),
        "section_counts": dict(Counter(row.get("section", "") for row in rows).most_common()),
        "legacy_relevance_flag_counts": dict(Counter(row.get("relevance_flag", "") for row in rows)),
        "body_missing_count": sum(bool(row.get("body_missing")) or not str(row.get("body_text", "")).strip() for row in rows),
        "duplicate_body_count": sum(bool(row.get("duplicate_body")) for row in rows),
        "invalid_timestamp_count": len(bad_timestamps),
        "invalid_timestamps": bad_timestamps,
        "boundary_record_count": len(boundary_records),
        "boundary_records": boundary_records,
        "screening_counts": {
            "relevant": len(relevant),
            "uncertain": len(uncertain),
            "irrelevant_or_static_exclusion": len(irrelevant),
        },
        "selection_method": (
            "deterministic contextual screening using headline, summary, lead and full-body proximity; "
            "section never acts as the sole exclusion; configured manual overrides take precedence"
        ),
        "manual_overrides_applied": len(overrides),
    }

    screened_rows = relevant + uncertain + irrelevant
    manual_review = build_manual_review(screened_rows, relevant, uncertain, irrelevant, config)
    audit["manual_review_sample_count"] = len(manual_review)
    audit["manual_review_status_counts"] = dict(Counter(row["review_status"] for row in manual_review))
    audit["manual_review_decision_counts"] = dict(
        Counter(row["manual_review_decision"] for row in manual_review if row["manual_review_decision"])
    )
    prior_report_path = output_dir / "sample_selection_report.json"
    if prior_report_path.exists():
        try:
            prior_report = json.loads(prior_report_path.read_text(encoding="utf-8"))
            if "preprocessing" in prior_report:
                audit["preprocessing"] = prior_report["preprocessing"]
        except (json.JSONDecodeError, OSError):
            pass
    write_csv(output_dir / "manual_review_articles.csv", manual_review)
    write_csv(output_dir / "uncertain_articles.csv", [review_export(row) for row in uncertain])
    write_csv(output_dir / "excluded_articles.csv", exclusions)
    write_json(output_dir / "sample_selection_report.json", audit)
    return SelectionResult(rows, relevant, uncertain, irrelevant, exclusions, audit)


def exclusion_row(row: dict[str, Any], reason_code: str, detail: str) -> dict[str, Any]:
    return {
        "article_id": row.get("article_id", ""),
        "title": row.get("title", ""),
        "published_at_new_york": row.get("published_at_new_york", ""),
        "section": row.get("section", ""),
        "legacy_relevance_flag": row.get("relevance_flag", ""),
        "exclusion_reason": reason_code,
        "exclusion_detail": detail,
        "url": row.get("url", ""),
    }


def review_export(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "article_id": row["article_id"],
        "title": row["title"],
        "published_at_new_york": row["published_at_new_york"],
        "section": row["section"],
        "legacy_relevance_flag": row["relevance_flag"],
        "screening_decision": row["screening_decision"],
        "screening_reason": row["screening_reason"],
        "manual_override_applied": row["manual_override_applied"],
        "url": row["url"],
        "text_excerpt": clean_for_screening(row["body_text"])[:600],
    }


def build_manual_review(
    all_rows: list[dict[str, Any]],
    relevant: list[dict[str, Any]],
    uncertain: list[dict[str, Any]],
    irrelevant: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    cfg = config["selection"]
    rng = random.Random(int(cfg["random_seed"]))
    chosen: dict[str, dict[str, Any]] = {}

    def add(pool: list[dict[str, Any]], n: int | None = None) -> None:
        selected = pool if n is None or n >= len(pool) else rng.sample(pool, n)
        for row in selected:
            chosen[row["article_id"]] = row

    add([row for row in all_rows if row.get("section") in {"Film", "Sport"}])
    add([row for row in all_rows if row.get("section") == "Football"], int(cfg["review_football_sample"]))
    add([row for row in all_rows if row.get("section") == "World news"], int(cfg["review_world_sample"]))
    add(uncertain, int(cfg["review_uncertain_sample"]))
    add(relevant, int(cfg["review_relevant_sample"]))
    reviewed = set(cfg.get("manual_reviewed_ids", []))
    review_all = bool(cfg.get("manual_review_all_generated_sample", False))
    exports: list[dict[str, Any]] = []
    for row in sorted(chosen.values(), key=lambda item: (item.get("section", ""), item["article_id"])):
        exports.append({
            **review_export(row),
            "review_status": "reviewed" if review_all or row["article_id"] in reviewed else "pending_manual_review",
            "manual_review_decision": row["screening_decision"] if review_all or row["article_id"] in reviewed else "",
            "manual_review_note": (
                config["selection"].get("manual_overrides", {}).get(row["article_id"], {}).get("reason", "")
                if review_all or row["article_id"] in reviewed else ""
            ),
        })
    return exports


def remove_templates(text: str) -> str:
    text = html.unescape(HTML_RE.sub(" ", text or ""))
    text = URL_RE.sub(" ", text)
    lines = []
    for line in re.split(r"[\r\n]+", text):
        line = re.sub(r"\s+", " ", line).strip()
        if line and not OBVIOUS_NON_NEWS_TEMPLATE_RE.search(line):
            lines.append(line)
    return "\n".join(lines)


def preprocess_and_build(
    config: dict[str, Any], selection: SelectionResult, output_dir: Path, model_dir: Path
) -> tuple[list[dict[str, Any]], list[list[str]], Dictionary, list[list[tuple[int, int]]]]:
    cfg = config["preprocessing"]
    model_name = cfg["spacy_model"]
    try:
        nlp = spacy.load(model_name, disable=["parser", "ner", "textcat"])
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model {model_name!r} is unavailable. Install it explicitly before running."
        ) from exc
    custom_stop = {word.lower() for word in cfg.get("custom_stopwords", [])}
    min_len = int(cfg["min_token_length"])
    cleaned = [remove_templates(row["body_text"]) for row in selection.relevant]
    base_tokens: list[list[str]] = []
    for doc in nlp.pipe(cleaned, batch_size=int(cfg["spacy_batch_size"]), n_process=1):
        tokens: list[str] = []
        for token in doc:
            lemma = token.lemma_.lower().strip()
            if (
                token.is_alpha
                and not token.is_stop
                and len(lemma) >= min_len
                and lemma not in custom_stop
                and lemma not in {"pron", "-pron-"}
            ):
                tokens.append(lemma)
        base_tokens.append(tokens)

    phrases = Phrases(
        base_tokens,
        min_count=int(cfg["phrase_min_count"]),
        threshold=float(cfg["phrase_threshold"]),
        delimiter="_",
    )
    phraser = Phraser(phrases)
    phraser.save(str(model_dir / "bigram_phraser.phraser"))
    phrased_tokens = [list(phraser[tokens]) for tokens in base_tokens]
    min_tokens = int(cfg["min_document_tokens"])

    token_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    training_tokens: list[list[str]] = []
    modeling_exclusions: list[dict[str, Any]] = []
    for row, original, tokens in zip(selection.relevant, cleaned, phrased_tokens):
        too_short = len(tokens) < min_tokens
        item = {
            "article_id": row["article_id"],
            "title": row["title"],
            "published_at_new_york": row["published_at_new_york"],
            "section": row["section"],
            "url": row["url"],
            "body_text": row["body_text"],
            "cleaned_body_text": original,
            "original_character_count": len(row["body_text"]),
            "original_word_count": len(re.findall(r"\b\w+\b", row["body_text"])),
            "lda_token_count": len(tokens),
            "below_min_token_threshold": too_short,
            "tokens": tokens,
        }
        token_rows.append(item)
        if too_short:
            modeling_exclusions.append(exclusion_row(row, "below_minimum_token_count", f"{len(tokens)} < {min_tokens}"))
        else:
            training_rows.append(row)
            training_tokens.append(tokens)

    dictionary = Dictionary(training_tokens)
    before_vocab = len(dictionary)
    dictionary.filter_extremes(
        no_below=int(cfg["no_below"]),
        no_above=float(cfg["no_above"]),
        keep_n=None,
    )
    dictionary.compactify()
    after_vocab = len(dictionary)
    if after_vocab < 100:
        raise RuntimeError(
            f"Filtered vocabulary is too small ({after_vocab}). Adjust no_below/no_above explicitly."
        )

    bows = [dictionary.doc2bow(tokens) for tokens in training_tokens]
    final_rows: list[dict[str, Any]] = []
    final_tokens: list[list[str]] = []
    final_bows: list[list[tuple[int, int]]] = []
    empty_ids: set[str] = set()
    for row, tokens, bow in zip(training_rows, training_tokens, bows):
        if not bow:
            empty_ids.add(row["article_id"])
            modeling_exclusions.append(exclusion_row(row, "empty_bow_after_dictionary_filter", "no retained dictionary terms"))
        else:
            final_rows.append(row)
            final_tokens.append(tokens)
            final_bows.append(bow)

    for item in token_rows:
        item["empty_bow_after_dictionary_filter"] = item["article_id"] in empty_ids
        item["used_for_lda_training"] = (
            not item["below_min_token_threshold"] and not item["empty_bow_after_dictionary_filter"]
        )
    write_jsonl(output_dir / "tokenized_documents.jsonl", token_rows)
    dictionary.save(str(model_dir / "guardian_lda_dictionary.dict"))
    MmCorpus.serialize(str(model_dir / "guardian_lda_corpus.mm"), final_bows)
    write_json(model_dir / "training_document_index.json", [
        {
            "corpus_index": index,
            "article_id": row["article_id"],
            "title": row["title"],
            "published_at_new_york": row["published_at_new_york"],
        }
        for index, row in enumerate(final_rows)
    ])

    all_exclusions = selection.exclusions + modeling_exclusions
    write_csv(output_dir / "excluded_articles.csv", all_exclusions)
    report = dict(selection.audit)
    report["preprocessing"] = {
        "spacy_model": model_name,
        "document_unit": "one full article body per document; title stored separately and not appended",
        "relevant_before_token_threshold": len(selection.relevant),
        "below_minimum_token_count": sum(item["below_min_token_threshold"] for item in token_rows),
        "minimum_token_count": min_tokens,
        "dictionary_size_before_filter": before_vocab,
        "dictionary_size_after_filter": after_vocab,
        "no_below": int(cfg["no_below"]),
        "no_above": float(cfg["no_above"]),
        "empty_bow_count": len(empty_ids),
        "final_lda_document_count": len(final_rows),
        "token_count_summary": {
            "min": min((item["lda_token_count"] for item in token_rows), default=0),
            "median": float(np.median([item["lda_token_count"] for item in token_rows])) if token_rows else 0,
            "mean": float(np.mean([item["lda_token_count"] for item in token_rows])) if token_rows else 0,
            "max": max((item["lda_token_count"] for item in token_rows), default=0),
        },
        "custom_stopwords": cfg.get("custom_stopwords", []),
        "custom_stopword_log": cfg.get("custom_stopword_log", {}),
    }
    write_json(output_dir / "sample_selection_report.json", report)
    return final_rows, final_tokens, dictionary, final_bows


def topic_term_sets(model: LdaModel, topn: int) -> list[set[str]]:
    return [{term for term, _ in model.show_topic(topic_id, topn=topn)} for topic_id in range(model.num_topics)]


def repetition_metrics(model: LdaModel, topn: int) -> tuple[float, float, float]:
    sets = topic_term_sets(model, topn)
    all_terms = set().union(*sets) if sets else set()
    jaccards = []
    for left in range(len(sets)):
        for right in range(left + 1, len(sets)):
            union = sets[left] | sets[right]
            jaccards.append(len(sets[left] & sets[right]) / len(union) if union else 0.0)
    return (
        len(all_terms) / max(1, len(sets) * topn),
        float(np.mean(jaccards)) if jaccards else 0.0,
        max(jaccards, default=0.0),
    )


def document_topic_matrix(model: LdaModel, corpus: Sequence[list[tuple[int, int]]]) -> np.ndarray:
    matrix = np.zeros((len(corpus), model.num_topics), dtype=float)
    for row_index, bow in enumerate(corpus):
        distribution = model.get_document_topics(bow, minimum_probability=0.0)
        for topic_id, probability in distribution:
            matrix[row_index, topic_id] = probability
    return matrix


def representative_rows(
    model: LdaModel,
    matrix: np.ndarray,
    rows: list[dict[str, Any]],
    per_topic: int,
    k: int,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for topic_id in range(k):
        order = np.argsort(matrix[:, topic_id])[::-1][:per_topic]
        for rank, index in enumerate(order, 1):
            row = rows[int(index)]
            result.append({
                "num_topics": k,
                "topic_id": topic_id,
                "rank": rank,
                "topic_probability": float(matrix[int(index), topic_id]),
                "article_id": row["article_id"],
                "title": row["title"],
                "published_at_new_york": row["published_at_new_york"],
                "section": row["section"],
                "url": row["url"],
                "body_excerpt": clean_for_screening(row["body_text"])[:350],
            })
    return result


def train_candidates(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    tokens: list[list[str]],
    dictionary: Dictionary,
    corpus: list[list[tuple[int, int]]],
    output_dir: Path,
    model_dir: Path,
) -> None:
    cfg = config["lda"]
    topn = int(cfg["topn_keywords"])
    comparison: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    candidate_reps: list[dict[str, Any]] = []
    for k in [int(value) for value in cfg["candidate_topics"]]:
        print(f"Training candidate LDA K={k}...", flush=True)
        model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,
            random_state=int(cfg["random_state"]),
            passes=int(cfg["passes"]),
            iterations=int(cfg["iterations"]),
            alpha=cfg["alpha"],
            eta=cfg["eta"],
            minimum_probability=0.0,
            eval_every=None,
        )
        model_path = model_dir / f"candidate_lda_k{k}.model"
        model.save(str(model_path))
        coherence_model = CoherenceModel(
            model=model,
            texts=tokens,
            dictionary=dictionary,
            coherence="c_v",
            processes=1,
        )
        overall_coherence = float(coherence_model.get_coherence())
        per_topic_coherence = [float(value) for value in coherence_model.get_coherence_per_topic()]
        matrix = document_topic_matrix(model, corpus)
        shares = matrix.mean(axis=0)
        unique_ratio, mean_jaccard, max_jaccard = repetition_metrics(model, topn)
        selection_notes = cfg.get("candidate_selection_notes", {}) or {}
        comparison.append({
            "num_topics": k,
            "overall_c_v_coherence": overall_coherence,
            "mean_topic_c_v_coherence": float(np.mean(per_topic_coherence)),
            "min_topic_c_v_coherence": min(per_topic_coherence),
            "max_topic_c_v_coherence": max(per_topic_coherence),
            "top15_unique_term_ratio": unique_ratio,
            "mean_pairwise_top15_jaccard": mean_jaccard,
            "max_pairwise_top15_jaccard": max_jaccard,
            "mean_dominant_topic_probability": float(matrix.max(axis=1).mean()),
            "training_documents": len(corpus),
            "dictionary_terms": len(dictionary),
            "random_state": int(cfg["random_state"]),
            "passes": int(cfg["passes"]),
            "iterations": int(cfg["iterations"]),
            "recommended": k == int(cfg["selected_k"]),
            "interpretive_selection_note": selection_notes.get(k, selection_notes.get(str(k), "")),
        })
        for topic_id in range(k):
            terms = model.show_topic(topic_id, topn=topn)
            diagnostics.append({
                "num_topics": k,
                "topic_id": topic_id,
                "c_v_coherence": per_topic_coherence[topic_id],
                "average_probability_share": float(shares[topic_id]),
                "top_keywords": "; ".join(term for term, _ in terms),
                "weighted_keywords": "; ".join(f"{term}:{weight:.6f}" for term, weight in terms),
            })
        candidate_reps.extend(representative_rows(
            model,
            matrix,
            rows,
            int(cfg["candidate_representative_articles"]),
            k,
        ))
        print(f"K={k} c_v={overall_coherence:.4f}", flush=True)
    write_csv(output_dir / "model_comparison.csv", comparison)
    write_csv(output_dir / "candidate_topic_diagnostics.csv", diagnostics)
    write_csv(output_dir / "candidate_topic_representative_articles.csv", candidate_reps)


def load_training_artifacts(
    config: dict[str, Any], selection: SelectionResult, model_dir: Path, output_dir: Path
) -> tuple[list[dict[str, Any]], list[list[str]], Dictionary, list[list[tuple[int, int]]]]:
    dictionary = Dictionary.load(str(model_dir / "guardian_lda_dictionary.dict"))
    corpus = list(MmCorpus(str(model_dir / "guardian_lda_corpus.mm")))
    index = json.loads((model_dir / "training_document_index.json").read_text(encoding="utf-8"))
    by_id = {row["article_id"]: row for row in selection.input_rows}
    rows = [by_id[item["article_id"]] for item in index]
    token_rows = load_jsonl(output_dir / "tokenized_documents.jsonl", validate_input_schema=False)
    token_by_id = {
        item["article_id"]: item["tokens"]
        for item in token_rows
        if item.get("used_for_lda_training")
    }
    tokens = [token_by_id[row["article_id"]] for row in rows]
    modeling_exclusions: list[dict[str, Any]] = []
    min_tokens = int(config["preprocessing"]["min_document_tokens"])
    for item in token_rows:
        source = by_id[item["article_id"]]
        if item.get("below_min_token_threshold"):
            modeling_exclusions.append(exclusion_row(
                source,
                "below_minimum_token_count",
                f"{item.get('lda_token_count', 0)} < {min_tokens}",
            ))
        elif item.get("empty_bow_after_dictionary_filter"):
            modeling_exclusions.append(exclusion_row(
                source,
                "empty_bow_after_dictionary_filter",
                "no retained dictionary terms",
            ))
    write_csv(output_dir / "excluded_articles.csv", selection.exclusions + modeling_exclusions)
    token_counts = [int(item.get("lda_token_count", 0)) for item in token_rows]
    prefilter_terms = {
        token
        for item in token_rows
        if not item.get("below_min_token_threshold")
        for token in item.get("tokens", [])
    }
    report = dict(selection.audit)
    report["preprocessing"] = {
        "spacy_model": config["preprocessing"]["spacy_model"],
        "document_unit": "one full article body per document; title stored separately and not appended",
        "relevant_before_token_threshold": len(selection.relevant),
        "below_minimum_token_count": sum(bool(item.get("below_min_token_threshold")) for item in token_rows),
        "minimum_token_count": min_tokens,
        "dictionary_size_before_filter": len(prefilter_terms),
        "dictionary_size_after_filter": len(dictionary),
        "no_below": int(config["preprocessing"]["no_below"]),
        "no_above": float(config["preprocessing"]["no_above"]),
        "empty_bow_count": sum(bool(item.get("empty_bow_after_dictionary_filter")) for item in token_rows),
        "final_lda_document_count": len(rows),
        "token_count_summary": {
            "min": min(token_counts, default=0),
            "median": float(np.median(token_counts)) if token_counts else 0,
            "mean": float(np.mean(token_counts)) if token_counts else 0,
            "max": max(token_counts, default=0),
        },
        "custom_stopwords": config["preprocessing"].get("custom_stopwords", []),
        "custom_stopword_log": config["preprocessing"].get("custom_stopword_log", {}),
    }
    write_json(output_dir / "sample_selection_report.json", report)
    if not (len(rows) == len(tokens) == len(corpus)):
        raise RuntimeError("Saved training artifacts are not aligned")
    return rows, tokens, dictionary, corpus


def labels_for(config: dict[str, Any], k: int) -> dict[int, dict[str, str]]:
    raw = config.get("topic_labels", {}) or {}
    labels: dict[int, dict[str, str]] = {}
    for topic_id in range(k):
        item = raw.get(topic_id, raw.get(str(topic_id), {})) or {}
        labels[topic_id] = {
            "english_name": item.get("english_name", f"Topic {topic_id}"),
            "chinese_name": item.get("chinese_name", f"主题 {topic_id}"),
            "definition": item.get("definition", "Pending human confirmation."),
            "category": item.get("category", "mixed_or_unclear"),
            "naming_basis": item.get("naming_basis", "Candidate keywords and representative articles require review."),
            "naming_status": item.get("naming_status", "pending_human_confirmation"),
        }
    return labels


def cosine_topic_matching(base: LdaModel, other: LdaModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left = base.get_topics()
    right = other.get_topics()
    left_norm = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-12)
    right_norm = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
    similarities = left_norm @ right_norm.T
    row_ind, col_ind = linear_sum_assignment(-similarities)
    return row_ind, col_ind, similarities


def run_stability(
    config: dict[str, Any],
    base: LdaModel,
    dictionary: Dictionary,
    corpus: list[list[tuple[int, int]]],
    model_dir: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    cfg = config["lda"]
    k = base.num_topics
    topn = int(cfg["topn_keywords"])
    rows: list[dict[str, Any]] = []
    base_sets = topic_term_sets(base, topn)
    for seed in [int(value) for value in cfg["stability_seeds"]]:
        print(f"Training stability model seed={seed}...", flush=True)
        model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,
            random_state=seed,
            passes=int(cfg["passes"]),
            iterations=int(cfg["iterations"]),
            alpha=cfg["alpha"],
            eta=cfg["eta"],
            minimum_probability=0.0,
            eval_every=None,
        )
        model.save(str(model_dir / f"stability_lda_k{k}_seed{seed}.model"))
        row_ind, col_ind, similarities = cosine_topic_matching(base, model)
        other_sets = topic_term_sets(model, topn)
        for base_topic, matched_topic in zip(row_ind, col_ind):
            union = base_sets[int(base_topic)] | other_sets[int(matched_topic)]
            rows.append({
                "base_random_state": int(cfg["random_state"]),
                "comparison_random_state": seed,
                "base_topic_id": int(base_topic),
                "matched_topic_id": int(matched_topic),
                "cosine_similarity_full_vocabulary": float(similarities[base_topic, matched_topic]),
                "top15_keyword_jaccard": len(base_sets[int(base_topic)] & other_sets[int(matched_topic)]) / max(1, len(union)),
                "matching_method": "Hungarian assignment maximizing cosine similarity of full topic-word distributions",
            })
    write_csv(output_dir / "topic_stability.csv", rows)
    return rows


def date_sequence(start: date, end: date) -> list[date]:
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def generate_final_outputs(
    config: dict[str, Any],
    selection: SelectionResult,
    rows: list[dict[str, Any]],
    tokens: list[list[str]],
    dictionary: Dictionary,
    corpus: list[list[tuple[int, int]]],
    output_dir: Path,
    model_dir: Path,
    figure_dir: Path,
) -> None:
    cfg = config["lda"]
    k = int(cfg["selected_k"])
    candidate_path = model_dir / f"candidate_lda_k{k}.model"
    if not candidate_path.exists():
        raise FileNotFoundError(f"Selected candidate model not found: {candidate_path}")
    model = LdaModel.load(str(candidate_path))
    final_path = model_dir / f"guardian_lda_final_k{k}.model"
    model.save(str(final_path))
    labels = labels_for(config, k)
    topn = int(cfg["topn_keywords"])
    matrix = document_topic_matrix(model, corpus)

    keyword_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    for topic_id in range(k):
        label = labels[topic_id]
        label_rows.append({"topic_id": topic_id, **label})
        for rank, (term, weight) in enumerate(model.show_topic(topic_id, topn=topn), 1):
            keyword_rows.append({
                "topic_id": topic_id,
                "english_name": label["english_name"],
                "chinese_name": label["chinese_name"],
                "rank": rank,
                "keyword": term,
                "weight": float(weight),
            })
    write_csv(output_dir / "topic_keywords.csv", keyword_rows)
    write_csv(output_dir / "topic_labels.csv", label_rows)

    reps = representative_rows(model, matrix, rows, int(cfg["representative_articles"]), k)
    for item in reps:
        label = labels[int(item["topic_id"])]
        item["english_name"] = label["english_name"]
        item["chinese_name"] = label["chinese_name"]
    write_csv(output_dir / "topic_representative_articles.csv", reps)

    probability_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        probabilities = matrix[index]
        dominant = int(np.argmax(probabilities))
        item: dict[str, Any] = {
            "article_id": row["article_id"],
            "title": row["title"],
            "published_at_new_york": row["published_at_new_york"],
            "section": row["section"],
        }
        for topic_id in range(k):
            item[f"topic_{topic_id}_probability"] = float(probabilities[topic_id])
        item["dominant_topic"] = dominant
        item["dominant_topic_probability"] = float(probabilities[dominant])
        probability_rows.append(item)
    write_csv(output_dir / "article_topic_probabilities.csv", probability_rows)

    overall = matrix.mean(axis=0)
    overall_rows = [
        {
            "topic_id": topic_id,
            "english_name": labels[topic_id]["english_name"],
            "chinese_name": labels[topic_id]["chinese_name"],
            "overall_attention_share": float(overall[topic_id]),
            "effective_article_count": len(rows),
        }
        for topic_id in range(k)
    ]
    write_csv(output_dir / "overall_topic_attention.csv", overall_rows)

    frame = pd.DataFrame({
        "new_york_date": [row["published_at_new_york"][:10] for row in rows],
        **{f"topic_{topic_id}_attention": matrix[:, topic_id] for topic_id in range(k)},
    })
    grouped = frame.groupby("new_york_date", sort=True)
    daily = grouped.mean(numeric_only=True)
    daily.insert(0, "article_count", grouped.size())
    start = date.fromisoformat(config["date_range"]["start_new_york"])
    end = date.fromisoformat(config["date_range"]["end_new_york"])
    all_dates = [day.isoformat() for day in date_sequence(start, end)]
    daily = daily.reindex(all_dates)
    daily.index.name = "new_york_date"
    daily["article_count"] = daily["article_count"].astype("Int64")
    daily.reset_index().to_csv(output_dir / "daily_topic_attention.csv", index=False, encoding=CSV_ENCODING)

    attention_cols = [f"topic_{topic_id}_attention" for topic_id in range(k)]
    smoothed = daily[attention_cols].rolling(window=7, min_periods=1).mean()
    smoothed.insert(0, "article_count", daily["article_count"])
    smoothed.index.name = "new_york_date"
    smoothed.reset_index().to_csv(output_dir / "daily_topic_attention_7d.csv", index=False, encoding=CSV_ENCODING)

    stability_rows = run_stability(config, model, dictionary, corpus, model_dir, output_dir)
    create_figures(config, labels, overall, daily, smoothed, output_dir, figure_dir)
    build_analysis_report(
        config, selection, rows, dictionary, model, overall, labels, stability_rows, output_dir
    )
    validate_outputs(
        config, rows, dictionary, corpus, matrix, model, final_path, output_dir
    )


def chart_note(fig: plt.Figure) -> None:
    fig.text(
        0.01, 0.01,
        "Guardian news sample topic attention; this is not a measure of US voter preferences.",
        fontsize=8,
        color="#555555",
    )


def create_figures(
    config: dict[str, Any],
    labels: dict[int, dict[str, str]],
    overall: np.ndarray,
    daily: pd.DataFrame,
    smoothed: pd.DataFrame,
    output_dir: Path,
    figure_dir: Path,
) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    comparison = pd.read_csv(output_dir / "model_comparison.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(comparison["num_topics"], comparison["overall_c_v_coherence"], marker="o", linewidth=2, color="#24527A")
    ax.set_xlabel("Number of topics (K)")
    ax.set_ylabel("c_v coherence")
    ax.set_title("Candidate LDA coherence")
    ax.set_xticks(comparison["num_topics"])
    ax.grid(axis="y", alpha=0.25)
    chart_note(fig)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(figure_dir / "coherence_by_k.png", dpi=180)
    plt.close(fig)

    order = np.argsort(overall)
    names = [f"T{i}: {labels[i]['english_name']}" for i in order]
    fig, ax = plt.subplots(figsize=(8.5, max(4.5, len(order) * 0.55)))
    ax.barh(names, overall[order], color="#3C78A8")
    ax.set_xlabel("Mean article topic probability")
    ax.set_title("Overall topic attention in the Guardian sample")
    ax.grid(axis="x", alpha=0.2)
    chart_note(fig)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(figure_dir / "overall_topic_attention.png", dpi=180)
    plt.close(fig)

    k = len(labels)
    fig, axes = plt.subplots(k, 1, figsize=(11, max(8, 2.1 * k)), sharex=True)
    if k == 1:
        axes = [axes]
    dates = pd.to_datetime(smoothed.index)
    for topic_id, ax in enumerate(axes):
        column = f"topic_{topic_id}_attention"
        ax.plot(dates, daily[column], color="#AFC8DD", linewidth=0.7, alpha=0.6, label="Daily")
        ax.plot(dates, smoothed[column], color="#24527A", linewidth=1.7, label="7-day mean")
        ax.set_ylabel(f"T{topic_id}")
        ax.set_title(labels[topic_id]["english_name"], loc="left", fontsize=9)
        ax.grid(axis="y", alpha=0.18)
    axes[0].legend(loc="upper right", ncol=2, frameon=False)
    axes[-1].set_xlabel("New York local date")
    fig.suptitle("Daily topic attention in the Guardian sample (7-day mean for display)", y=0.995)
    chart_note(fig)
    fig.tight_layout(rect=(0, 0.025, 1, 0.99))
    fig.savefig(figure_dir / "daily_topic_attention_timeseries.png", dpi=180)
    plt.close(fig)


def build_analysis_report(
    config: dict[str, Any],
    selection: SelectionResult,
    rows: list[dict[str, Any]],
    dictionary: Dictionary,
    model: LdaModel,
    overall: np.ndarray,
    labels: dict[int, dict[str, str]],
    stability_rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    comparison = pd.read_csv(output_dir / "model_comparison.csv", encoding="utf-8-sig")
    selected = comparison.loc[comparison["num_topics"] == model.num_topics].iloc[0]
    lines = [
        "# Guardian US midterm-election LDA analysis",
        "",
        "This is a retrospective descriptive analysis of the full three-month Guardian sample. "
        "It is not a no-look-ahead real-time forecasting model and does not measure US voter preferences.",
        "",
        "## Sample",
        "",
        f"- Input articles: {len(selection.input_rows):,}",
        f"- Contextually screened relevant articles before modeling filters: {len(selection.relevant):,}",
        f"- Uncertain articles reserved for sensitivity analysis: {len(selection.uncertain):,}",
        f"- Final LDA documents: {len(rows):,}",
        f"- Dictionary terms: {len(dictionary):,}",
        "",
        "## Model choice",
        "",
        f"Selected K={model.num_topics}. Its c_v coherence is {selected['overall_c_v_coherence']:.4f}. "
        "Selection must combine coherence, keyword overlap, representative articles and interpretability; "
        "it is not based on election probabilities or market returns.",
        "",
        "## Interpreted topics",
        "",
    ]
    for topic_id in np.argsort(overall)[::-1]:
        label = labels[int(topic_id)]
        terms = ", ".join(term for term, _ in model.show_topic(int(topic_id), topn=10))
        lines.extend([
            f"### Topic {topic_id}: {label['english_name']} / {label['chinese_name']}",
            "",
            f"- Overall attention share: {overall[int(topic_id)]:.4f}",
            f"- Category: {label['category']}",
            f"- Definition: {label['definition']}",
            f"- Naming basis: {label['naming_basis']}",
            f"- Top terms: {terms}",
            "",
        ])
    mean_cosine = float(np.mean([row["cosine_similarity_full_vocabulary"] for row in stability_rows]))
    mean_jaccard = float(np.mean([row["top15_keyword_jaccard"] for row in stability_rows]))
    lines.extend([
        "## Stability",
        "",
        f"Two additional seeds were matched to the base topics with Hungarian assignment. "
        f"Mean full-vocabulary cosine similarity was {mean_cosine:.4f}; mean top-15 Jaccard was {mean_jaccard:.4f}.",
        "",
        "## Attention definition",
        "",
        "Overall attention is the mean full topic probability across effective articles. Daily attention is "
        "the mean full topic probability across articles published on each New York local date. Each article "
        "has equal weight regardless of length. Dates with no effective articles remain missing. The plotted "
        "7-day series is the unweighted rolling mean of available calendar-day daily shares and is for display only.",
    ])
    (output_dir / "lda_analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_outputs(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    dictionary: Dictionary,
    corpus: list[list[tuple[int, int]]],
    matrix: np.ndarray,
    model: LdaModel,
    final_path: Path,
    output_dir: Path,
) -> None:
    k = model.num_topics
    probability_sums = matrix.sum(axis=1)
    max_sum_error = float(np.max(np.abs(probability_sums - 1.0)))
    unique_ids = len({row["article_id"] for row in rows})
    article_csv = pd.read_csv(output_dir / "article_topic_probabilities.csv", encoding="utf-8-sig")
    daily = pd.read_csv(output_dir / "daily_topic_attention.csv", encoding="utf-8-sig")
    attention_columns = [f"topic_{topic_id}_attention" for topic_id in range(k)]
    daily_valid = daily.loc[daily["article_count"].notna() & (daily["article_count"] > 0)]
    daily_errors = np.abs(daily_valid[attention_columns].sum(axis=1) - 1.0)
    reloaded_model = LdaModel.load(str(final_path))
    reloaded_dictionary = Dictionary.load(str(Path(config["model_dir"]) / "guardian_lda_dictionary.dict"))
    reloaded_phraser = Phraser.load(str(Path(config["model_dir"]) / "bigram_phraser.phraser"))
    inference = reloaded_model.get_document_topics(corpus[0], minimum_probability=0.0)
    checks = {
        "validation_status": "pass",
        "effective_articles": len(rows),
        "article_probability_rows": len(article_csv),
        "unique_article_ids": unique_ids,
        "article_ids_aligned": article_csv["article_id"].tolist() == [row["article_id"] for row in rows],
        "topic_probability_column_count": len([col for col in article_csv if col.startswith("topic_") and col.endswith("_probability")]),
        "max_article_probability_sum_error": max_sum_error,
        "max_daily_attention_sum_error": float(daily_errors.max()) if len(daily_errors) else None,
        "empty_bow_count_in_training_corpus": sum(not bow for bow in corpus),
        "saved_model_reloaded": reloaded_model.num_topics == k,
        "saved_dictionary_reloaded": len(reloaded_dictionary) == len(dictionary),
        "saved_phrase_model_reloaded": reloaded_phraser is not None,
        "reloaded_inference_probability_sum": float(sum(prob for _, prob in inference)),
    }
    if not (
        len(article_csv) == len(rows)
        and unique_ids == len(rows)
        and checks["article_ids_aligned"]
        and checks["topic_probability_column_count"] == k
        and max_sum_error < 1e-5
        and (checks["max_daily_attention_sum_error"] is None or checks["max_daily_attention_sum_error"] < 1e-5)
        and checks["empty_bow_count_in_training_corpus"] == 0
        and checks["saved_model_reloaded"]
        and checks["saved_dictionary_reloaded"]
        and abs(checks["reloaded_inference_probability_sum"] - 1.0) < 1e-5
    ):
        checks["validation_status"] = "fail"
        write_json(output_dir / "validation_report.json", checks)
        raise RuntimeError("Final output validation failed; see validation_report.json")
    write_json(output_dir / "validation_report.json", checks)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="lda_config.yaml")
    parser.add_argument("--stage", choices=["selection", "candidates", "final", "all"], default="all")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(Path(args.config))
    output_dir, model_dir, figure_dir = ensure_dirs(config)
    selection = audit_and_select(config, output_dir)
    print(
        f"Input={len(selection.input_rows)} relevant={len(selection.relevant)} "
        f"uncertain={len(selection.uncertain)} irrelevant={len(selection.irrelevant)}",
        flush=True,
    )
    if args.stage == "selection":
        return 0
    if args.stage in {"candidates", "all"}:
        rows, tokens, dictionary, corpus = preprocess_and_build(
            config, selection, output_dir, model_dir
        )
        train_candidates(config, rows, tokens, dictionary, corpus, output_dir, model_dir)
        if args.stage == "candidates":
            return 0
    else:
        rows, tokens, dictionary, corpus = load_training_artifacts(
            config, selection, model_dir, output_dir
        )
    generate_final_outputs(
        config, selection, rows, tokens, dictionary, corpus, output_dir, model_dir, figure_dir
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
