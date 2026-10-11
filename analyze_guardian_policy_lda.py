#!/usr/bin/env python
"""Policy-focused LDA analysis for the Guardian US midterm corpus.

The script keeps the original collected data unchanged. It separates substantive
policy coverage from campaign-process coverage, then fits the main LDA only to
articles classified as policy substance. Mixed articles are retained separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import permutations
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import spacy
import yaml
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import CoherenceModel, LdaModel
from gensim.models.phrases import Phrases, Phraser
from PIL import Image, ImageDraw, ImageFont

import analyze_guardian_lda as base


CSV_ENCODING = "utf-8-sig"


def rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


# These broad domains are used only to decide whether an article contains
# substantive policy content. They do not determine the LDA topic labels.
POLICY_DOMAINS: dict[str, re.Pattern[str]] = {
    "economy_labor_housing": rx(
        r"\b(econom(?:y|ic|ics)|inflation|affordab(?:ility|le)|cost of living|price(?:s)?|"
        r"tariff(?:s)?|tax(?:es|ation)?|trade|job(?:s)?|employment|unemployment|wage(?:s)?|"
        r"labor|labour|union(?:s)?|housing|rent(?:s)?|mortgage(?:s)?|budget|debt|deficit|"
        r"spending|bank(?:ing)?|interest rate(?:s)?|financial|crypto(?:currency)?|farm(?:er|ing|s)?)\b"
    ),
    "energy_climate_environment": rx(
        r"\b(energy|oil|gasoline|gas price(?:s)?|diesel|fuel(?:s)?|pipeline(?:s)?|refiner(?:y|ies)|"
        r"electricity|power grid|renewable(?:s)?|solar|wind power|fossil fuel(?:s)?|emission(?:s)?|"
        r"climate|environment(?:al)?|pollution|wildfire(?:s)?|drought|water policy|conservation|"
        r"national monument(?:s)?|public land(?:s)?|tribal land(?:s)?)\b"
    ),
    "technology_ai": rx(
        r"(?<![A-Za-z])AI(?![A-Za-z])|\b(artificial intelligence|generative ai|algorithm(?:s|ic)?|"
        r"data ?cent(?:er|re)s?|technology|tech industr(?:y|ies)|semiconductor(?:s)?|chipmaker(?:s)?|"
        r"social media|online platform(?:s)?|digital privacy|cybersecurity|automation|robot(?:s|ics)?)\b"
    ),
    "health_reproductive": rx(
        r"\b(health ?care|public health|medicare|medicaid|insurance|hospital(?:s)?|vaccine(?:s)?|"
        r"measles|covid|disease|cancer|drug price(?:s)?|prescription(?:s)?|opioid(?:s)?|addiction|"
        r"abortion|reproductive|maternal|pregnan(?:cy|t)|contraception|planned parenthood)\b"
    ),
    "immigration_border": rx(
        r"\b(immigra(?:tion|nt|nts)|migrant(?:s)?|border|deport(?:ation|ations|ed|ing)?|asylum|"
        r"refugee(?:s)?|citizenship|visa(?:s)?|ICE agent(?:s)?|immigration enforcement|detention center(?:s)?)\b"
    ),
    "foreign_policy_security": rx(
        r"\b(foreign policy|national security|military|defen[cs]e|war|airstrike(?:s)?|missile(?:s)?|"
        r"sanction(?:s|ed|ing)?|diploma(?:cy|tic)|ceasefire|nato|iran(?:ian)?|israel(?:i)?|gaza|"
        r"palestin(?:e|ian|ians)|ukrain(?:e|ian)|russia(?:n)?|china|taiwan|venezuela|cuba|"
        r"saudi arabia|strait of hormuz|nuclear weapon(?:s)?|arms|foreign aid)\b"
    ),
    "education_research": rx(
        r"\b(education|school(?:s)?|teacher(?:s)?|student(?:s)?|university|universities|college(?:s)?|"
        r"tuition|student debt|research funding|science funding|grant(?:s)?|academic freedom)\b"
    ),
    "civil_social_rights": rx(
        r"\b(civil rights|human rights|gun(?:s)?|firearm(?:s)?|police|policing|crime|criminal justice|"
        r"prison(?:s)?|death penalty|lgbtq|transgender|same-sex|racial|racism|discrimination|"
        r"religious freedom|free speech|homelessness|poverty|food assistance|snap benefits)\b"
    ),
    "infrastructure_consumer": rx(
        r"\b(infrastructure|transportation|transit|road(?:s)?|bridge(?:s)?|rail|aviation|consumer protection|"
        r"antitrust|regulation|regulatory|right to repair|disaster insurance|emergency relief|federal funding)\b"
    ),
}

PROCESS_RE = rx(
    r"\b(midterm(?:s)?|election(?:s)?|campaign(?:s|ing)?|candidate(?:s|cy)?|primary|primaries|"
    r"poll(?:s|ing)?|ballot(?:s)?|voter(?:s)?|vote(?:s|d|ing)?|race|races|running for|run for|"
    r"nominee|nomination|endorse(?:ment|ments|d)?|swing state(?:s)?|battleground|seat(?:s)?|"
    r"reelection|re-election|turnout|fundrais(?:er|ers|ing)|"
    r"support(?:s|ed|ing)? (?:democrats?|republicans?)|back(?:s|ed|ing)? (?:democrats?|republicans?))\b"
)
POLICY_ACTION_RE = rx(
    r"\b(policy|policies|law|laws|bill|bills|ban|bans|banned|fund|funding|cut|cuts|subsidy|subsidies|"
    r"regulat(?:e|es|ed|ion|ions|ory)|reform|program|benefit|benefits|rights|cost|costs|price|prices|"
    r"spending|budget|tax|tariff|sanction|aid|coverage|enforcement|administration proposal|executive order)\b"
)
PURE_PROCESS_TITLE_RE = rx(
    r"\b(wins?|loses?|defeats?|primary result(?:s)?|poll shows?|polling|drops out|withdraws candidacy|"
    r"launches? (?:a )?(?:senate|house|congressional|governor) bid|running for (?:senate|congress|house)|"
    r"campaign trail|reelection bid|re-election bid|race to replace|scramble to replace)\b"
)
STATIC_TITLE_RE = rx(r"^corrections and clarifications$")
ROUNDUP_TITLE_RE = rx(
    r"\b(first thing|morning briefing|monday briefing|wednesday briefing|friday briefing|"
    r"morning mail|afternoon update|war briefing|the week in|today in focus)\b"
)
PROFILE_TITLE_RE = rx(r"\b(obituary|in his own words|in her own words|diaries|conspiracy theories abound)\b")
STRONG_POLICY_ACTION_RE = rx(
    r"\b(policy|law|bill|ban|funding|grant|cut|subsidy|regulation|reform|benefit|rights|"
    r"cost|price|spending|budget|tax|tariff|sanction|aid|coverage|enforcement|deportation|"
    r"health ?care|medicare|medicaid|abortion|immigration|emission|pollution|antitrust)\b"
)


@dataclass
class PolicySelection:
    input_rows: list[dict[str, Any]]
    included: list[dict[str, Any]]
    policy_substance: list[dict[str, Any]]
    mixed: list[dict[str, Any]]
    campaign_process: list[dict[str, Any]]
    uncertain: list[dict[str, Any]]
    excluded: list[dict[str, Any]]
    report: dict[str, Any]


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Policy configuration must be a YAML mapping")
    return config


def ensure_dirs(config: dict[str, Any]) -> tuple[Path, Path, Path]:
    paths = tuple(Path(config[key]) for key in ("output_dir", "model_dir", "figure_dir"))
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
    return paths


def domain_hits(text: str) -> list[str]:
    return [name for name, pattern in POLICY_DOMAINS.items() if pattern.search(text)]


def domain_occurrences(text: str, domains: Sequence[str]) -> int:
    return sum(len(POLICY_DOMAINS[name].findall(text)) for name in domains)


def original_screen(
    row: dict[str, Any], base_config: dict[str, Any], start: date, end: date
) -> tuple[str, str]:
    try:
        base.parse_timestamp(row["published_at_utc"])
        local = base.parse_timestamp(row["published_at_new_york"]).date()
    except (ValueError, TypeError) as exc:
        return "irrelevant", f"invalid_timestamp: {exc}"
    if local < start or local > end:
        return "irrelevant", "outside_new_york_date_range"
    if row.get("body_missing") or not str(row.get("body_text", "")).strip():
        return "irrelevant", "body_missing"
    if row.get("duplicate_body"):
        return "irrelevant", "duplicate_body"
    decision, reason, _ = base.deterministic_screen(row)
    override = base_config.get("selection", {}).get("manual_overrides", {}).get(row["article_id"])
    if override:
        decision = override["decision"]
        reason = f"base manual override: {override.get('reason', '')}"
    return decision, reason


def policy_scope(row: dict[str, Any]) -> tuple[str, str, list[str]]:
    title = base.clean_for_screening(row.get("title", ""))
    summary = base.clean_for_screening(row.get("summary", ""))
    body = base.clean_for_screening(row.get("body_text", ""))
    headline = f"{title} {summary}".strip()
    lead = body[:3000]
    front = f"{headline} {lead}".strip()
    title_domains = domain_hits(title)
    summary_domains = domain_hits(summary)
    central_domains = sorted(set(title_domains) | set(summary_domains))
    lead_domains = domain_hits(front)
    process_headline = bool(PROCESS_RE.search(headline))
    process_front = bool(PROCESS_RE.search(front))
    policy_action = bool(POLICY_ACTION_RE.search(front))
    summary_occurrences = domain_occurrences(summary, summary_domains)

    if STATIC_TITLE_RE.search(title):
        return "excluded", "template/corrections page, not a substantive policy article", central_domains
    if ROUNDUP_TITLE_RE.search(title):
        return "uncertain", "multi-story briefing is not a coherent single-policy document", central_domains
    if PROFILE_TITLE_RE.search(title) and not STRONG_POLICY_ACTION_RE.search(headline):
        return "uncertain", "profile/obituary framing does not center a substantive policy action", central_domains
    if row.get("section") in {"Film", "Sport", "Football", "Culture", "Books", "Television & radio", "Life and style"}:
        if not STRONG_POLICY_ACTION_RE.search(headline):
            return "uncertain", "special-interest coverage lacks a central, explicit policy action", central_domains
    if title_domains:
        if process_headline or PURE_PROCESS_TITLE_RE.search(title):
            return (
                "mixed",
                "title contains substantive policy content and explicit campaign-process content",
                central_domains,
            )
        return "policy_substance", "title centers one or more substantive policy domains", central_domains
    if summary_domains and policy_action and summary_occurrences >= 2 and not PURE_PROCESS_TITLE_RE.search(title):
        if process_headline or process_front:
            return "mixed", "summary contains repeated policy/action evidence plus campaign-process context", central_domains
        return "policy_substance", "summary contains repeated policy-domain and policy-action evidence", central_domains
    if process_headline or PURE_PROCESS_TITLE_RE.search(title):
        return "campaign_process", "article centers election mechanics or candidate competition without central policy content", central_domains
    if central_domains or lead_domains:
        return "uncertain", "policy language exists but is not central enough in the title/summary for the policy model", central_domains
    return "uncertain", "no sufficiently central substantive policy domain detected", central_domains


def scope_export(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "article_id": row["article_id"],
        "title": row["title"],
        "published_at_new_york": row.get("published_at_new_york", ""),
        "section": row.get("section", ""),
        "legacy_relevance_flag": row.get("relevance_flag", ""),
        "original_screening_decision": row.get("original_screening_decision", ""),
        "original_screening_reason": row.get("original_screening_reason", ""),
        "scope_flag": row.get("scope_flag", ""),
        "scope_reason": row.get("scope_reason", ""),
        "policy_domain_signals": "; ".join(row.get("policy_domain_signals", [])),
        "included_in_policy_model": row.get("scope_flag") == "policy_substance",
        "manual_override_applied": row.get("policy_manual_override_applied", False),
        "url": row.get("url", ""),
        "body_excerpt": base.clean_for_screening(row.get("body_text", ""))[:500],
    }


def select_policy_sample(config: dict[str, Any], output_dir: Path) -> PolicySelection:
    rows = base.load_jsonl(Path(config["input_jsonl"]))
    base_config = base.load_config(Path(config["base_config"]))
    start = date.fromisoformat(config["date_range"]["start_new_york"])
    end = date.fromisoformat(config["date_range"]["end_new_york"])
    overrides = config["selection"].get("manual_overrides", {}) or {}
    buckets: dict[str, list[dict[str, Any]]] = {
        "policy_substance": [], "mixed": [], "campaign_process": [], "uncertain": [], "excluded": []
    }
    exported: list[dict[str, Any]] = []

    for source in rows:
        row = dict(source)
        original_decision, original_reason = original_screen(row, base_config, start, end)
        row["original_screening_decision"] = original_decision
        row["original_screening_reason"] = original_reason
        if original_decision == "irrelevant":
            flag, reason, domains = "excluded", f"base relevance screen: {original_reason}", []
        else:
            flag, reason, domains = policy_scope(row)
        if flag in {"policy_substance", "mixed"} and row.get("section") != "US news":
            headline = f"{row.get('title', '')} {row.get('summary', '')}"
            if not (
                base.US_CORE_RE.search(headline)
                or base.US_POLITICAL_RE.search(headline)
                or base.US_ABBREV_RE.search(headline)
            ):
                flag = "uncertain"
                reason = "non-US-news article lacks explicit US political context in title/summary"
        override = overrides.get(row["article_id"])
        if override:
            proposed = override.get("scope_flag", "")
            if proposed not in buckets:
                raise ValueError(f"Invalid policy scope override for {row['article_id']}: {proposed}")
            flag = proposed
            reason = f"manual override: {override.get('reason', '').strip()}"
        row["scope_flag"] = flag
        row["scope_reason"] = reason
        row["policy_domain_signals"] = domains
        row["policy_manual_override_applied"] = bool(override)
        buckets[flag].append(row)
        exported.append(scope_export(row))

    included = list(buckets["policy_substance"])
    included.sort(key=lambda row: (row["published_at_new_york"], row["article_id"]))
    daily = Counter(row["published_at_new_york"][:10] for row in included)
    report = {
        "input_file": config["input_jsonl"],
        "input_format_used": "JSONL only; CSV was not merged",
        "total_input_articles": len(rows),
        "requested_new_york_date_range": {"start": str(start), "end": str(end)},
        "original_screening_counts": dict(Counter(row["original_screening_decision"] for row in buckets["policy_substance"] + buckets["mixed"] + buckets["campaign_process"] + buckets["uncertain"] + buckets["excluded"])),
        "policy_scope_counts": {key: len(value) for key, value in buckets.items()},
        "included_before_token_filters": len(included),
        "included_scope_flags": ["policy_substance"],
        "selection_principle": "policy centrality is assessed from title, summary and lead using broad domain evidence; domains only screen scope and do not seed or label LDA topics",
        "policy_domain_signal_counts_in_included": dict(Counter(domain for row in included for domain in row["policy_domain_signals"])),
        "included_section_counts": dict(Counter(row.get("section", "") for row in included).most_common()),
        "included_daily_counts_new_york": dict(sorted(daily.items())),
        "manual_override_count": len(overrides),
    }
    prior = output_dir / "policy_sample_selection_report.json"
    if prior.exists():
        try:
            old = json.loads(prior.read_text(encoding="utf-8"))
            if "preprocessing" in old:
                report["preprocessing"] = old["preprocessing"]
        except (OSError, json.JSONDecodeError):
            pass
    base.write_csv(output_dir / "article_scope_labels.csv", exported)
    base.write_csv(output_dir / "policy_uncertain_articles.csv", [scope_export(row) for row in buckets["uncertain"]])
    base.write_csv(
        output_dir / "policy_excluded_articles.csv",
        [scope_export(row) for row in buckets["mixed"] + buckets["campaign_process"] + buckets["uncertain"] + buckets["excluded"]],
    )
    base.write_json(output_dir / "policy_sample_selection_report.json", report)
    return PolicySelection(
        rows, included, buckets["policy_substance"], buckets["mixed"], buckets["campaign_process"],
        buckets["uncertain"], buckets["excluded"], report,
    )


def preprocess(
    config: dict[str, Any], selection: PolicySelection, output_dir: Path, model_dir: Path
) -> tuple[list[dict[str, Any]], list[list[str]], Dictionary, list[list[tuple[int, int]]]]:
    cfg = config["preprocessing"]
    try:
        nlp = spacy.load(cfg["spacy_model"], disable=["parser", "ner", "textcat"])
    except OSError as exc:
        raise RuntimeError(f"spaCy model {cfg['spacy_model']!r} is unavailable") from exc
    custom_stop = {word.casefold() for word in cfg.get("custom_stopwords", [])}
    keep_short = {word.casefold() for word in cfg.get("keep_short_tokens", [])}
    min_len = int(cfg["min_token_length"])
    cleaned = [base.remove_templates(row["body_text"]) for row in selection.included]
    base_tokens: list[list[str]] = []
    for doc in nlp.pipe(cleaned, batch_size=int(cfg["spacy_batch_size"]), n_process=1):
        tokens: list[str] = []
        for token in doc:
            lemma = token.lemma_.casefold().strip()
            if (
                token.is_alpha
                and not token.is_stop
                and (len(lemma) >= min_len or lemma in keep_short)
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
    phraser.save(str(model_dir / "policy_bigram_phraser.phraser"))
    phrased = [list(phraser[tokens]) for tokens in base_tokens]
    min_tokens = int(cfg["min_document_tokens"])

    token_rows: list[dict[str, Any]] = []
    eligible_rows: list[dict[str, Any]] = []
    eligible_tokens: list[list[str]] = []
    for row, cleaned_body, tokens in zip(selection.included, cleaned, phrased):
        short = len(tokens) < min_tokens
        token_rows.append({
            "article_id": row["article_id"], "title": row["title"],
            "published_at_new_york": row["published_at_new_york"], "section": row["section"],
            "scope_flag": row["scope_flag"], "policy_domain_signals": row["policy_domain_signals"],
            "url": row["url"], "body_text": row["body_text"], "cleaned_body_text": cleaned_body,
            "original_character_count": len(row["body_text"]),
            "original_word_count": len(re.findall(r"\b\w+\b", row["body_text"])),
            "lda_token_count": len(tokens), "below_min_token_threshold": short, "tokens": tokens,
        })
        if not short:
            eligible_rows.append(row)
            eligible_tokens.append(tokens)

    dictionary = Dictionary(eligible_tokens)
    vocab_before = len(dictionary)
    dictionary.filter_extremes(
        no_below=int(cfg["no_below"]), no_above=float(cfg["no_above"]), keep_n=None
    )
    dictionary.compactify()
    if len(dictionary) < 100:
        raise RuntimeError(f"Policy vocabulary too small after filtering: {len(dictionary)}")
    bows = [dictionary.doc2bow(tokens) for tokens in eligible_tokens]
    final_rows: list[dict[str, Any]] = []
    final_tokens: list[list[str]] = []
    final_bows: list[list[tuple[int, int]]] = []
    empty_ids: set[str] = set()
    for row, tokens, bow in zip(eligible_rows, eligible_tokens, bows):
        if bow:
            final_rows.append(row); final_tokens.append(tokens); final_bows.append(bow)
        else:
            empty_ids.add(row["article_id"])
    for item in token_rows:
        item["empty_bow_after_dictionary_filter"] = item["article_id"] in empty_ids
        item["used_for_policy_lda"] = not item["below_min_token_threshold"] and not item["empty_bow_after_dictionary_filter"]

    base.write_jsonl(output_dir / "policy_tokenized_documents.jsonl", token_rows)
    dictionary.save(str(model_dir / "policy_lda_dictionary.dict"))
    MmCorpus.serialize(str(model_dir / "policy_lda_corpus.mm"), final_bows)
    base.write_json(model_dir / "policy_training_document_index.json", [
        {"corpus_index": i, "article_id": row["article_id"], "title": row["title"],
         "published_at_new_york": row["published_at_new_york"]}
        for i, row in enumerate(final_rows)
    ])
    counts = [item["lda_token_count"] for item in token_rows]
    report = dict(selection.report)
    report["preprocessing"] = {
        "spacy_model": cfg["spacy_model"],
        "document_unit": "one complete article body; title stored separately and not appended",
        "keep_short_tokens": sorted(keep_short),
        "policy_process_stopwords": [word for word in cfg["custom_stopwords"] if word in {
            "election", "elections", "midterm", "midterms", "campaign", "campaigns", "candidate",
            "candidates", "voter", "voters", "vote", "votes", "voting", "primary", "primaries",
            "race", "races", "run", "running", "win", "wins", "won", "poll", "polls", "ballot",
            "ballots", "seat", "seats"
        }],
        "stopword_change_log": cfg.get("stopword_change_log", {}),
        "included_before_token_filters": len(selection.included),
        "below_minimum_token_count": sum(item["below_min_token_threshold"] for item in token_rows),
        "minimum_token_count": min_tokens,
        "empty_bow_count": len(empty_ids),
        "final_policy_lda_documents": len(final_rows),
        "dictionary_size_before_filter": vocab_before,
        "dictionary_size_after_filter": len(dictionary),
        "no_below": int(cfg["no_below"]), "no_above": float(cfg["no_above"]),
        "token_count_summary": {
            "min": min(counts, default=0), "median": float(np.median(counts)) if counts else 0,
            "mean": float(np.mean(counts)) if counts else 0, "max": max(counts, default=0),
        },
        "ai_token_document_count": sum("ai" in item["tokens"] for item in token_rows),
        "artificial_intelligence_phrase_document_count": sum("artificial_intelligence" in item["tokens"] for item in token_rows),
    }
    base.write_json(output_dir / "policy_sample_selection_report.json", report)
    return final_rows, final_tokens, dictionary, final_bows


def probability_matrix(model: LdaModel, corpus: Sequence[list[tuple[int, int]]]) -> np.ndarray:
    return np.asarray([
        [prob for _, prob in model.get_document_topics(bow, minimum_probability=0.0)]
        for bow in corpus
    ], dtype=float)


def representative_rows(
    model: LdaModel, matrix: np.ndarray, rows: list[dict[str, Any]], per_topic: int
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for topic_id in range(model.num_topics):
        for rank, index in enumerate(np.argsort(matrix[:, topic_id])[::-1][:per_topic], 1):
            row = rows[int(index)]
            result.append({
                "num_topics": model.num_topics, "topic_id": topic_id, "rank": rank,
                "topic_probability": float(matrix[int(index), topic_id]),
                "article_id": row["article_id"], "title": row["title"],
                "published_at_new_york": row["published_at_new_york"], "section": row["section"],
                "scope_flag": row["scope_flag"], "url": row["url"],
                "body_excerpt": base.clean_for_screening(row["body_text"])[:350],
            })
    return result


def train_candidates(
    config: dict[str, Any], rows: list[dict[str, Any]], tokens: list[list[str]],
    dictionary: Dictionary, corpus: list[list[tuple[int, int]]], output_dir: Path, model_dir: Path,
) -> None:
    cfg = config["lda"]
    comparison: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    reps: list[dict[str, Any]] = []
    topn = int(cfg["topn_keywords"])
    notes = cfg.get("candidate_selection_notes", {}) or {}
    for k in map(int, cfg["candidate_topics"]):
        print(f"Training policy candidate K={k}...", flush=True)
        model = LdaModel(
            corpus=corpus, id2word=dictionary, num_topics=k, random_state=int(cfg["random_state"]),
            passes=int(cfg["passes"]), iterations=int(cfg["iterations"]), alpha=cfg["alpha"],
            eta=cfg["eta"], minimum_probability=0.0, eval_every=None,
        )
        model.save(str(model_dir / f"policy_candidate_lda_k{k}.model"))
        coherence = CoherenceModel(model=model, texts=tokens, dictionary=dictionary, coherence="c_v", processes=1)
        overall = float(coherence.get_coherence())
        per_topic = [float(value) for value in coherence.get_coherence_per_topic()]
        matrix = probability_matrix(model, corpus)
        unique_ratio, mean_jaccard, max_jaccard = base.repetition_metrics(model, topn)
        comparison.append({
            "num_topics": k, "overall_c_v_coherence": overall,
            "mean_topic_c_v_coherence": float(np.mean(per_topic)), "min_topic_c_v_coherence": min(per_topic),
            "max_topic_c_v_coherence": max(per_topic), "top15_unique_term_ratio": unique_ratio,
            "mean_pairwise_top15_jaccard": mean_jaccard, "max_pairwise_top15_jaccard": max_jaccard,
            "mean_dominant_topic_probability": float(matrix.max(axis=1).mean()),
            "training_documents": len(corpus), "dictionary_terms": len(dictionary),
            "random_state": int(cfg["random_state"]), "passes": int(cfg["passes"]),
            "iterations": int(cfg["iterations"]), "recommended": k == int(cfg["selected_k"]),
            "interpretive_selection_note": notes.get(k, notes.get(str(k), "")),
        })
        shares = matrix.mean(axis=0)
        for topic_id in range(k):
            terms = model.show_topic(topic_id, topn=topn)
            diagnostics.append({
                "num_topics": k, "topic_id": topic_id, "c_v_coherence": per_topic[topic_id],
                "average_probability_share": float(shares[topic_id]),
                "top_keywords": "; ".join(term for term, _ in terms),
                "weighted_keywords": "; ".join(f"{term}:{weight:.6f}" for term, weight in terms),
            })
        reps.extend(representative_rows(model, matrix, rows, int(cfg["candidate_representative_articles"])))
        print(f"Policy K={k} c_v={overall:.4f}", flush=True)
    base.write_csv(output_dir / "policy_model_comparison.csv", comparison)
    base.write_csv(output_dir / "policy_candidate_topic_diagnostics.csv", diagnostics)
    base.write_csv(output_dir / "policy_candidate_representative_articles.csv", reps)


def load_artifacts(
    config: dict[str, Any], selection: PolicySelection, output_dir: Path, model_dir: Path
) -> tuple[list[dict[str, Any]], list[list[str]], Dictionary, list[list[tuple[int, int]]]]:
    dictionary = Dictionary.load(str(model_dir / "policy_lda_dictionary.dict"))
    corpus = list(MmCorpus(str(model_dir / "policy_lda_corpus.mm")))
    index = json.loads((model_dir / "policy_training_document_index.json").read_text(encoding="utf-8"))
    by_id = {row["article_id"]: row for row in selection.included}
    rows = [by_id[item["article_id"]] for item in index]
    token_rows = base.load_jsonl(output_dir / "policy_tokenized_documents.jsonl", validate_input_schema=False)
    token_map = {item["article_id"]: item["tokens"] for item in token_rows if item["used_for_policy_lda"]}
    tokens = [token_map[row["article_id"]] for row in rows]
    if not (len(rows) == len(tokens) == len(corpus)):
        raise RuntimeError("Saved policy artifacts are not aligned")
    # Restore preprocessing information if the selection stage refreshed its report.
    report = dict(selection.report)
    old = json.loads((output_dir / "policy_sample_selection_report.json").read_text(encoding="utf-8"))
    if "preprocessing" in old:
        report["preprocessing"] = old["preprocessing"]
    base.write_json(output_dir / "policy_sample_selection_report.json", report)
    return rows, tokens, dictionary, corpus


def labels_for(config: dict[str, Any], k: int) -> dict[int, dict[str, str]]:
    raw = config.get("topic_labels", {}) or {}
    result: dict[int, dict[str, str]] = {}
    for topic_id in range(k):
        item = raw.get(topic_id, raw.get(str(topic_id), {})) or {}
        result[topic_id] = {
            "english_name": item.get("english_name", f"Policy Topic {topic_id}"),
            "chinese_name": item.get("chinese_name", f"政策主题 {topic_id}"),
            "definition": item.get("definition", "Pending human confirmation."),
            "category": item.get("category", "policy_or_social_issue"),
            "naming_basis": item.get("naming_basis", "Pending review of keywords and representative articles."),
            "naming_status": item.get("naming_status", "pending_human_confirmation"),
        }
    return result


def topic_matching(base_model: LdaModel, other: LdaModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left, right = base_model.get_topics(), other.get_topics()
    # Avoid numpy.linalg and BLAS-backed matrix multiplication here. On one
    # Windows Conda build those native calls raise 0xc06d007f, while elementary
    # NumPy reductions produce the identical cosine-similarity calculation.
    left = left / np.maximum(np.sqrt(np.sum(left * left, axis=1, keepdims=True)), 1e-12)
    right = right / np.maximum(np.sqrt(np.sum(right * right, axis=1, keepdims=True)), 1e-12)
    similarities = np.sum(left[:, None, :] * right[None, :, :], axis=2)
    # The configured policy models are small, so exact permutation search gives
    # the maximum-weight one-to-one assignment without a native optimizer.
    k = similarities.shape[0]
    if k > 9:
        raise ValueError("Exact stability matching supports K <= 9; reduce selected_k or add a safe optimizer.")
    best = max(permutations(range(k)), key=lambda order: sum(float(similarities[i, order[i]]) for i in range(k)))
    row_ind = np.arange(k, dtype=int)
    col_ind = np.asarray(best, dtype=int)
    return row_ind, col_ind, similarities


def run_stability(
    config: dict[str, Any], base_model: LdaModel, dictionary: Dictionary,
    corpus: list[list[tuple[int, int]]], output_dir: Path, model_dir: Path,
) -> list[dict[str, Any]]:
    cfg = config["lda"]
    rows: list[dict[str, Any]] = []
    base_sets = base.topic_term_sets(base_model, int(cfg["topn_keywords"]))
    for seed in map(int, cfg["stability_seeds"]):
        stability_path = model_dir / f"policy_stability_k{base_model.num_topics}_seed{seed}.model"
        if stability_path.exists():
            print(f"Loading policy stability seed={seed}...", flush=True)
            model = LdaModel.load(str(stability_path))
        else:
            print(f"Training policy stability seed={seed}...", flush=True)
            model = LdaModel(
                corpus=corpus, id2word=dictionary, num_topics=base_model.num_topics, random_state=seed,
                passes=int(cfg["passes"]), iterations=int(cfg["iterations"]), alpha=cfg["alpha"],
                eta=cfg["eta"], minimum_probability=0.0, eval_every=None,
            )
            model.save(str(stability_path))
        row_ind, col_ind, similarities = topic_matching(base_model, model)
        other_sets = base.topic_term_sets(model, int(cfg["topn_keywords"]))
        for left, right in zip(row_ind, col_ind):
            union = base_sets[int(left)] | other_sets[int(right)]
            rows.append({
                "base_random_state": int(cfg["random_state"]), "comparison_random_state": seed,
                "base_topic_id": int(left), "matched_topic_id": int(right),
                "cosine_similarity_full_vocabulary": float(similarities[left, right]),
                "top15_keyword_jaccard": len(base_sets[int(left)] & other_sets[int(right)]) / max(1, len(union)),
                "matching_method": "Exact permutation assignment maximizing cosine similarity of full topic-word distributions",
            })
    base.write_csv(output_dir / "policy_topic_stability.csv", rows)
    return rows


def create_figures(
    config: dict[str, Any], labels: dict[int, dict[str, str]], overall: np.ndarray,
    daily: pd.DataFrame, smoothed: pd.DataFrame, output_dir: Path, figure_dir: Path,
) -> None:
    """Draw the three required PNGs without a BLAS-backed plotting backend.

    This project environment has a Windows DLL failure in Matplotlib's affine
    renderer. Pillow keeps figure production inside the same Conda environment
    while avoiding that native code path.
    """
    figure_dir.mkdir(parents=True, exist_ok=True)
    try:
        regular = ImageFont.truetype("arial.ttf", 19)
        small = ImageFont.truetype("arial.ttf", 15)
        title_font = ImageFont.truetype("arialbd.ttf", 28)
    except OSError:
        regular = ImageFont.load_default()
        small = ImageFont.load_default()
        title_font = ImageFont.load_default()

    def canvas(width: int, height: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
        image = Image.new("RGB", (width, height), "white")
        return image, ImageDraw.Draw(image)

    def axes(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
        x0, y0, x1, y1 = box
        draw.line((x0, y1, x1, y1), fill="#333333", width=2)
        draw.line((x0, y0, x0, y1), fill="#333333", width=2)

    comparison = pd.read_csv(output_dir / "policy_model_comparison.csv", encoding=CSV_ENCODING)
    image, draw = canvas(1296, 810)
    box = (145, 105, 1235, 660); axes(draw, box)
    draw.text((145, 35), "Policy-focused candidate LDA coherence", font=title_font, fill="#111111")
    ks = comparison["num_topics"].astype(int).tolist()
    vals = comparison["overall_c_v_coherence"].astype(float).tolist()
    ymin, ymax = min(vals) - .01, max(vals) + .01
    points = []
    for i, (k_value, value) in enumerate(zip(ks, vals)):
        x = box[0] + i * (box[2] - box[0]) / max(1, len(ks) - 1)
        y = box[3] - (value - ymin) / (ymax - ymin) * (box[3] - box[1])
        points.append((x, y))
        draw.text((x - 8, box[3] + 14), str(k_value), font=regular, fill="#222222")
    for j in range(5):
        value = ymin + j * (ymax - ymin) / 4
        y = box[3] - j * (box[3] - box[1]) / 4
        draw.line((box[0], y, box[2], y), fill="#DDDDDD", width=1)
        draw.text((55, y - 9), f"{value:.3f}", font=small, fill="#555555")
    draw.line(points, fill="#24527A", width=5)
    for x, y in points:
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill="#24527A")
    draw.text((545, 708), "Number of policy topics (K)", font=regular, fill="#222222")
    draw.text((75, 35), "c_v", font=regular, fill="#222222")
    draw.text((25, 770), "Guardian policy-focused news sample; not a measure of US voter preferences.", font=small, fill="#555555")
    image.save(figure_dir / "policy_coherence_by_k.png")

    order = np.argsort(overall)
    image, draw = canvas(1530, 880)
    draw.text((45, 30), "Substantive policy attention in the Guardian sample", font=title_font, fill="#111111")
    left, right, top, bottom = 670, 1470, 115, 720
    max_value = max(float(overall.max()) * 1.12, .05)
    row_h = (bottom - top) / len(order)
    for rank, topic_id in enumerate(order):
        y0 = top + rank * row_h + 9; y1 = top + (rank + 1) * row_h - 9
        width = float(overall[topic_id]) / max_value * (right - left)
        label = f"T{topic_id}: {labels[int(topic_id)]['english_name']}"
        draw.text((35, y0 + 9), label, font=regular, fill="#222222")
        draw.rectangle((left, y0, left + width, y1), fill="#3C78A8")
        draw.text((left + width + 10, y0 + 9), f"{float(overall[topic_id]):.1%}", font=regular, fill="#222222")
    draw.line((left, bottom + 3, right, bottom + 3), fill="#333333", width=2)
    draw.text((870, 755), "Mean article topic probability", font=regular, fill="#222222")
    draw.text((25, 835), "Conditional on policy-bearing articles; not a measure of US voter preferences.", font=small, fill="#555555")
    image.save(figure_dir / "policy_overall_attention.png")

    k = len(labels)
    image, draw = canvas(1980, 390 + 245 * k)
    draw.text((55, 25), "Daily substantive policy attention in the Guardian sample", font=title_font, fill="#111111")
    dates = list(smoothed.index)
    plot_left, plot_right = 150, 1910
    for topic_id in range(k):
        panel_top = 105 + topic_id * 245; panel_bottom = panel_top + 175
        draw.text((55, panel_top - 27), f"T{topic_id}: {labels[topic_id]['english_name']}", font=regular, fill="#222222")
        axes(draw, (plot_left, panel_top, plot_right, panel_bottom))
        col = f"topic_{topic_id}_attention"
        combined = [float(v) for v in daily[col].dropna().tolist()] + [float(v) for v in smoothed[col].dropna().tolist()]
        ymax = max(max(combined, default=1.0) * 1.08, .05)
        for series, color, width in ((daily[col], "#AFC8DD", 2), (smoothed[col], "#24527A", 4)):
            segments: list[list[tuple[float, float]]] = [[]]
            for idx, value in enumerate(series.tolist()):
                if pd.isna(value):
                    if segments[-1]: segments.append([])
                    continue
                x = plot_left + idx * (plot_right - plot_left) / max(1, len(dates) - 1)
                y = panel_bottom - float(value) / ymax * (panel_bottom - panel_top)
                segments[-1].append((x, y))
            for segment in segments:
                if len(segment) >= 2: draw.line(segment, fill=color, width=width)
        draw.text((75, panel_top - 5), f"{ymax:.0%}", font=small, fill="#666666")
    for idx in (0, len(dates) // 2, len(dates) - 1):
        x = plot_left + idx * (plot_right - plot_left) / max(1, len(dates) - 1)
        draw.text((x - 45, 105 + k * 245), str(dates[idx]), font=small, fill="#333333")
    note_y = 310 + 245 * k
    draw.text((55, note_y), "Light line: daily | Dark line: 7-day mean | New York local date", font=regular, fill="#333333")
    draw.text((55, note_y + 45), "Conditional on policy-bearing articles; 7-day mean is for display only.", font=small, fill="#555555")
    image.save(figure_dir / "policy_daily_attention_timeseries.png")


def final_outputs(
    config: dict[str, Any], selection: PolicySelection, rows: list[dict[str, Any]], tokens: list[list[str]],
    dictionary: Dictionary, corpus: list[list[tuple[int, int]]], output_dir: Path, model_dir: Path, figure_dir: Path,
) -> None:
    cfg = config["lda"]
    k = int(cfg["selected_k"])
    comparison_path = output_dir / "policy_model_comparison.csv"
    comparison = pd.read_csv(comparison_path, encoding=CSV_ENCODING)
    comparison["recommended"] = comparison["num_topics"].astype(int).eq(k)
    notes = cfg.get("candidate_selection_notes", {}) or {}
    comparison["interpretive_selection_note"] = comparison["num_topics"].map(
        lambda value: notes.get(int(value), notes.get(str(int(value)), ""))
    )
    comparison.to_csv(comparison_path, index=False, encoding=CSV_ENCODING)
    candidate = model_dir / f"policy_candidate_lda_k{k}.model"
    if not candidate.exists():
        raise FileNotFoundError(f"Selected policy candidate not found: {candidate}")
    model = LdaModel.load(str(candidate))
    final_path = model_dir / f"guardian_policy_lda_final_k{k}.model"
    model.save(str(final_path))
    labels = labels_for(config, k)
    matrix = probability_matrix(model, corpus)
    topn = int(cfg["topn_keywords"])

    keyword_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    for topic_id in range(k):
        label_rows.append({"topic_id": topic_id, **labels[topic_id]})
        for rank, (term, weight) in enumerate(model.show_topic(topic_id, topn=topn), 1):
            keyword_rows.append({
                "topic_id": topic_id, "english_name": labels[topic_id]["english_name"],
                "chinese_name": labels[topic_id]["chinese_name"], "rank": rank,
                "keyword": term, "weight": float(weight),
            })
    base.write_csv(output_dir / "policy_topic_keywords.csv", keyword_rows)
    base.write_csv(output_dir / "policy_topic_labels.csv", label_rows)

    reps = representative_rows(model, matrix, rows, int(cfg["representative_articles"]))
    for item in reps:
        item.update({"english_name": labels[int(item["topic_id"])]["english_name"],
                     "chinese_name": labels[int(item["topic_id"])]["chinese_name"]})
    base.write_csv(output_dir / "policy_topic_representative_articles.csv", reps)

    probability_rows: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        probabilities = matrix[i]; dominant = int(np.argmax(probabilities))
        item: dict[str, Any] = {
            "article_id": row["article_id"], "title": row["title"],
            "published_at_new_york": row["published_at_new_york"], "section": row["section"],
            "scope_flag": row["scope_flag"],
        }
        item.update({f"topic_{topic_id}_probability": float(probabilities[topic_id]) for topic_id in range(k)})
        item["dominant_topic"] = dominant; item["dominant_topic_probability"] = float(probabilities[dominant])
        probability_rows.append(item)
    base.write_csv(output_dir / "policy_article_topic_probabilities.csv", probability_rows)

    overall = matrix.mean(axis=0)
    base.write_csv(output_dir / "policy_overall_topic_attention.csv", [{
        "topic_id": topic_id, "english_name": labels[topic_id]["english_name"],
        "chinese_name": labels[topic_id]["chinese_name"], "overall_attention_share": float(overall[topic_id]),
        "effective_policy_article_count": len(rows),
    } for topic_id in range(k)])

    frame = pd.DataFrame({
        "new_york_date": [row["published_at_new_york"][:10] for row in rows],
        **{f"topic_{topic_id}_attention": matrix[:, topic_id] for topic_id in range(k)},
    })
    grouped = frame.groupby("new_york_date", sort=True)
    daily = grouped.mean(numeric_only=True); daily.insert(0, "article_count", grouped.size())
    start = date.fromisoformat(config["date_range"]["start_new_york"])
    end = date.fromisoformat(config["date_range"]["end_new_york"])
    dates = [(start + timedelta(days=i)).isoformat() for i in range((end - start).days + 1)]
    daily = daily.reindex(dates); daily.index.name = "new_york_date"; daily["article_count"] = daily["article_count"].astype("Int64")
    daily.reset_index().to_csv(output_dir / "daily_policy_topic_attention.csv", index=False, encoding=CSV_ENCODING)
    attention_cols = [f"topic_{topic_id}_attention" for topic_id in range(k)]
    smoothed = daily[attention_cols].rolling(7, min_periods=1).mean(); smoothed.insert(0, "article_count", daily["article_count"])
    smoothed.index.name = "new_york_date"
    smoothed.reset_index().to_csv(output_dir / "daily_policy_topic_attention_7d.csv", index=False, encoding=CSV_ENCODING)

    stability = run_stability(config, model, dictionary, corpus, output_dir, model_dir)
    create_figures(config, labels, overall, daily, smoothed, output_dir, figure_dir)
    build_report(config, selection, rows, dictionary, model, overall, labels, stability, output_dir)
    validate(config, rows, dictionary, corpus, matrix, model, final_path, output_dir)


def build_report(
    config: dict[str, Any], selection: PolicySelection, rows: list[dict[str, Any]], dictionary: Dictionary,
    model: LdaModel, overall: np.ndarray, labels: dict[int, dict[str, str]], stability: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    comparison = pd.read_csv(output_dir / "policy_model_comparison.csv", encoding=CSV_ENCODING)
    selected = comparison.loc[comparison["num_topics"] == model.num_topics].iloc[0]
    lines = [
        "# Guardian substantive-policy LDA analysis", "",
        "This policy-focused model is conditional on articles classified as policy-substance. "
        "Mixed policy/campaign and campaign-process-only articles are reserved outside the main model. "
        "The results do not measure US voter preferences.", "",
        "## Sample", "",
        f"- Input articles: {len(selection.input_rows):,}",
        f"- Policy-substance articles before token filters: {len(selection.policy_substance):,}",
        f"- Mixed policy/campaign articles reserved for sensitivity analysis: {len(selection.mixed):,}",
        f"- Campaign-process articles excluded: {len(selection.campaign_process):,}",
        f"- Uncertain articles reserved for review: {len(selection.uncertain):,}",
        f"- Final policy LDA documents: {len(rows):,}", f"- Dictionary terms: {len(dictionary):,}", "",
        "## Model choice", "",
        f"Selected K={model.num_topics}; c_v coherence={selected['overall_c_v_coherence']:.4f}. "
        "Selection considered coherence, keyword overlap, representative articles and stability.", "",
        "## Policy topics", "",
    ]
    for topic_id in np.argsort(overall)[::-1]:
        label = labels[int(topic_id)]
        terms = ", ".join(term for term, _ in model.show_topic(int(topic_id), topn=10))
        lines += [
            f"### Topic {topic_id}: {label['english_name']} / {label['chinese_name']}", "",
            f"- Conditional attention share: {overall[int(topic_id)]:.4f}",
            f"- Definition: {label['definition']}", f"- Naming basis: {label['naming_basis']}",
            f"- Top terms: {terms}", "",
        ]
    mean_cosine = float(np.mean([row["cosine_similarity_full_vocabulary"] for row in stability]))
    mean_jaccard = float(np.mean([row["top15_keyword_jaccard"] for row in stability]))
    lines += [
        "## Stability", "",
        f"Matched-seed mean full-vocabulary cosine similarity={mean_cosine:.4f}; mean top-15 Jaccard={mean_jaccard:.4f}.", "",
        "## Interpretation", "",
        "Attention shares are conditional on policy-bearing articles. Each article contributes equally regardless of length. "
        "The full three-month sample makes this a retrospective descriptive analysis, not a real-time forecasting model.",
    ]
    (output_dir / "policy_lda_analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate(
    config: dict[str, Any], rows: list[dict[str, Any]], dictionary: Dictionary,
    corpus: list[list[tuple[int, int]]], matrix: np.ndarray, model: LdaModel,
    final_path: Path, output_dir: Path,
) -> None:
    k = model.num_topics
    article_csv = pd.read_csv(output_dir / "policy_article_topic_probabilities.csv", encoding=CSV_ENCODING)
    daily = pd.read_csv(output_dir / "daily_policy_topic_attention.csv", encoding=CSV_ENCODING)
    attention_cols = [f"topic_{i}_attention" for i in range(k)]
    valid_days = daily.loc[daily["article_count"].notna() & (daily["article_count"] > 0)]
    reloaded = LdaModel.load(str(final_path))
    reloaded_dict = Dictionary.load(str(Path(config["model_dir"]) / "policy_lda_dictionary.dict"))
    reloaded_phraser = Phraser.load(str(Path(config["model_dir"]) / "policy_bigram_phraser.phraser"))
    inference = reloaded.get_document_topics(corpus[0], minimum_probability=0.0)
    checks = {
        "validation_status": "pass", "effective_policy_articles": len(rows),
        "article_probability_rows": len(article_csv), "unique_article_ids": len(set(article_csv["article_id"])),
        "article_ids_aligned": article_csv["article_id"].tolist() == [row["article_id"] for row in rows],
        "topic_probability_column_count": len([c for c in article_csv if c.startswith("topic_") and c.endswith("_probability")]),
        "max_article_probability_sum_error": float(np.max(np.abs(matrix.sum(axis=1) - 1))),
        "max_daily_attention_sum_error": float(np.max(np.abs(valid_days[attention_cols].sum(axis=1) - 1))) if len(valid_days) else None,
        "empty_bow_count": sum(not bow for bow in corpus), "saved_model_reloaded": reloaded.num_topics == k,
        "saved_dictionary_reloaded": len(reloaded_dict) == len(dictionary),
        "saved_phrase_model_reloaded": reloaded_phraser is not None,
        "reloaded_inference_probability_sum": float(sum(prob for _, prob in inference)),
    }
    ok = (
        len(article_csv) == len(rows) == checks["unique_article_ids"] and checks["article_ids_aligned"]
        and checks["topic_probability_column_count"] == k
        and checks["max_article_probability_sum_error"] < 1e-5
        and (checks["max_daily_attention_sum_error"] is None or checks["max_daily_attention_sum_error"] < 1e-5)
        and checks["empty_bow_count"] == 0 and checks["saved_model_reloaded"]
        and checks["saved_dictionary_reloaded"] and abs(checks["reloaded_inference_probability_sum"] - 1) < 1e-5
    )
    if not ok:
        checks["validation_status"] = "fail"
    base.write_json(output_dir / "policy_validation_report.json", checks)
    if not ok:
        raise RuntimeError("Policy output validation failed")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", default="policy_lda_config.yaml")
    result.add_argument("--stage", choices=["selection", "candidates", "final", "all"], default="all")
    return result


def main() -> int:
    args = parser().parse_args()
    config = load_config(Path(args.config))
    output_dir, model_dir, figure_dir = ensure_dirs(config)
    selection = select_policy_sample(config, output_dir)
    print(
        f"Policy scope: substance={len(selection.policy_substance)} mixed={len(selection.mixed)} "
        f"campaign_process={len(selection.campaign_process)} uncertain={len(selection.uncertain)} "
        f"excluded={len(selection.excluded)}",
        flush=True,
    )
    if args.stage == "selection":
        return 0
    if args.stage in {"candidates", "all"}:
        rows, tokens, dictionary, corpus = preprocess(config, selection, output_dir, model_dir)
        print(f"Policy LDA documents={len(rows)} dictionary_terms={len(dictionary)}", flush=True)
        train_candidates(config, rows, tokens, dictionary, corpus, output_dir, model_dir)
        if args.stage == "candidates":
            return 0
    else:
        rows, tokens, dictionary, corpus = load_artifacts(config, selection, output_dir, model_dir)
    final_outputs(config, selection, rows, tokens, dictionary, corpus, output_dir, model_dir, figure_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
