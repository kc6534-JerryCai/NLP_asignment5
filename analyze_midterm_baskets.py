#!/usr/bin/env python
"""Build and test exploratory mid-term equity baskets.

The full-sample baskets intentionally reuse the stock-level association results.
Their p-values therefore describe the constructed basket in the same sample and
are not independent confirmation.  A chronological early-selection/late-
validation exercise is also produced without using validation-period returns in
component selection.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw
from scipy.stats import t as student_t

from analyze_equity_sentiment import (
    MAIN_TYPES,
    bh_adjust,
    block_bootstrap_slopes,
    hac_ols,
    load_font,
)


ROOT = Path(__file__).resolve().parent
CSV_ENCODING = "utf-8-sig"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding=CSV_ENCODING)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def bool_series(values: pd.Series) -> pd.Series:
    return values.fillna(False).map(lambda value: str(value).strip().lower() == "true")


def signal_key(topic_id: int, index_type: str) -> str:
    return f"T{int(topic_id)}_{index_type}"


def load_inputs(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    project = config["project"]
    returns = pd.read_csv(ROOT / project["returns"], encoding="utf-8-sig", parse_dates=["date"])
    returns = returns.loc[returns["ticker"].ne("SPY")].copy()
    returns["ticker"] = returns["ticker"].astype(str)
    signals = pd.read_csv(ROOT / project["signals"], encoding="utf-8-sig")
    signals = signals.loc[signals["index_type"].isin(MAIN_TYPES)].copy()
    signals["topic_id"] = pd.to_numeric(signals["topic_id"], errors="raise").astype(int)
    signals["signal_value"] = pd.to_numeric(signals["signal_value"], errors="coerce")
    signals["target_trading_date"] = signals["target_trading_date"].astype(str)
    fdr = pd.read_csv(ROOT / project["fdr_significant"], encoding="utf-8-sig")
    strict = pd.read_csv(ROOT / project["bootstrap_supported"], encoding="utf-8-sig")
    for frame in (fdr, strict):
        frame["topic_id"] = pd.to_numeric(frame["topic_id"], errors="raise").astype(int)
        frame["beta"] = pd.to_numeric(frame["beta"], errors="coerce")
        frame["q_value"] = pd.to_numeric(frame["q_value"], errors="coerce")
    return returns, signals, fdr, strict


def make_component_rows(
    basket_id: str,
    source: pd.DataFrame,
    selection_rule: str,
    scope: str,
    direction_mode: str,
    topic_id: int | None,
    index_type: str | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if scope == "union":
        for ticker, group in source.groupby("ticker", sort=True):
            associations = sorted({signal_key(int(r.topic_id), str(r.index_type)) for r in group.itertuples()})
            rows.append({
                "basket_id": basket_id,
                "ticker": ticker,
                "selection_rule": selection_rule,
                "scope": scope,
                "direction_mode": direction_mode,
                "topic_id": np.nan,
                "topic_name": "",
                "index_type": "",
                "selection_beta": np.nan,
                "selection_sign": np.nan,
                "source_min_q": float(group["q_value"].min()),
                "source_associations": "|".join(associations),
            })
    else:
        for row in source.sort_values(["q_value", "ticker"]).drop_duplicates("ticker").itertuples():
            rows.append({
                "basket_id": basket_id,
                "ticker": row.ticker,
                "selection_rule": selection_rule,
                "scope": scope,
                "direction_mode": direction_mode,
                "topic_id": int(topic_id),
                "topic_name": str(row.topic_name),
                "index_type": str(index_type),
                "selection_beta": float(row.beta),
                "selection_sign": float(np.sign(row.beta)),
                "source_min_q": float(row.q_value),
                "source_associations": signal_key(int(topic_id), str(index_type)),
            })
    return pd.DataFrame(rows)


def build_full_sample_specs(
    fdr: pd.DataFrame,
    strict: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    variants = set(config["baskets"]["variants"])
    minimum = int(config["baskets"]["minimum_components"])
    top_n = int(config["baskets"]["top_n"])
    specs: list[dict[str, Any]] = []
    component_frames: list[pd.DataFrame] = []

    for variant, source, rule in (
        ("fdr_union_long", fdr, "full_sample_global_fdr_q_lt_0.05"),
        ("bootstrap_union_long", strict, "full_sample_fdr_and_block_bootstrap"),
    ):
        if variant not in variants:
            continue
        basket_id = variant.upper()
        comp = make_component_rows(basket_id, source, rule, "union", "long_only_equal_weight", None, None)
        if len(comp) >= minimum:
            specs.append({"basket_id": basket_id, "scope": "union", "direction_mode": "long_only_equal_weight",
                          "topic_id": None, "index_type": None, "components": comp})
            component_frames.append(comp)

    group_keys = sorted({(int(r.topic_id), str(r.index_type)) for r in fdr.itertuples()})
    targeted_variants = [
        ("bootstrap_signal_long", strict, "full_sample_fdr_and_block_bootstrap", "long_only_equal_weight", None),
        ("bootstrap_top20_long", strict, "full_sample_fdr_and_block_bootstrap_top20_q", "long_only_equal_weight", top_n),
        ("fdr_signal_long", fdr, "full_sample_global_fdr_q_lt_0.05", "long_only_equal_weight", None),
        ("bootstrap_signal_directional", strict, "full_sample_fdr_and_block_bootstrap", "direction_aligned_equal_weight", None),
        ("bootstrap_top20_directional", strict, "full_sample_fdr_and_block_bootstrap_top20_q", "direction_aligned_equal_weight", top_n),
    ]
    for variant, source, rule, mode, limit in targeted_variants:
        if variant not in variants:
            continue
        for topic_id, index_type in group_keys:
            group = source.loc[source["topic_id"].eq(topic_id) & source["index_type"].eq(index_type)].copy()
            group = group.sort_values(["q_value", "ticker"])
            if limit is not None:
                group = group.head(limit)
            if len(group) < minimum:
                continue
            basket_id = f"{variant.upper()}__{signal_key(topic_id, index_type)}"
            comp = make_component_rows(basket_id, group, rule, "signal_specific", mode, topic_id, index_type)
            specs.append({"basket_id": basket_id, "scope": "signal_specific", "direction_mode": mode,
                          "topic_id": topic_id, "index_type": index_type, "components": comp})
            component_frames.append(comp)
    # Different labels can produce an identical component set (for example,
    # "all strict" and "strict top 20" when only seven stocks qualify). Keep
    # the first, stricter specification so the same hypothesis is not counted
    # repeatedly in the basket-level FDR family.
    unique_specs: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for spec in specs:
        comp = spec["components"].drop_duplicates("ticker").sort_values("ticker")
        signed = tuple(zip(comp["ticker"].astype(str), comp["selection_sign"].fillna(0).astype(float))) if spec["direction_mode"] == "direction_aligned_equal_weight" else tuple(comp["ticker"].astype(str))
        signature = (spec["scope"], spec["direction_mode"], spec["topic_id"], spec["index_type"], signed)
        if signature in seen:
            continue
        seen.add(signature)
        unique_specs.append(spec)
    components = pd.concat([spec["components"] for spec in unique_specs], ignore_index=True) if unique_specs else pd.DataFrame()
    return unique_specs, components


def basket_series(
    return_wide: pd.DataFrame,
    components: pd.DataFrame,
    direction_mode: str,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    tickers = [ticker for ticker in components["ticker"].astype(str).unique() if ticker in return_wide.columns]
    panel = return_wide.reindex(columns=tickers)
    valid_count = panel.notna().sum(axis=1)
    if direction_mode == "long_only_equal_weight":
        values = panel.mean(axis=1, skipna=True)
        positive_count = valid_count.copy()
        negative_count = pd.Series(0, index=panel.index, dtype=int)
    elif direction_mode == "direction_aligned_equal_weight":
        signs = components.drop_duplicates("ticker").set_index("ticker")["selection_sign"].reindex(tickers)
        signed = panel.mul(signs, axis=1)
        values = signed.mean(axis=1, skipna=True)
        positive_count = panel.reindex(columns=list(signs.index[signs.gt(0)])).notna().sum(axis=1)
        negative_count = panel.reindex(columns=list(signs.index[signs.lt(0)])).notna().sum(axis=1)
    else:
        raise ValueError(f"Unknown direction mode: {direction_mode}")
    values = values.where(valid_count.gt(0))
    return values, valid_count, positive_count, negative_count


def construct_daily_baskets(
    specs: list[dict[str, Any]],
    returns: pd.DataFrame,
) -> pd.DataFrame:
    market_dates = sorted(returns["date"].dropna().unique())
    wide = returns.pivot(index="date", columns="ticker", values="market_adjusted_open_close_return").reindex(market_dates)
    rows: list[pd.DataFrame] = []
    for spec in specs:
        values, valid, positive, negative = basket_series(wide, spec["components"], spec["direction_mode"])
        frame = pd.DataFrame({
            "date": wide.index,
            "basket_id": spec["basket_id"],
            "scope": spec["scope"],
            "direction_mode": spec["direction_mode"],
            "topic_id": spec["topic_id"],
            "index_type": spec["index_type"],
            "component_count": int(spec["components"]["ticker"].nunique()),
            "valid_component_count": valid.to_numpy(),
            "valid_positive_component_count": positive.to_numpy(),
            "valid_negative_component_count": negative.to_numpy(),
            "basket_market_adjusted_open_close_return": values.to_numpy(),
        })
        frame["cumulative_market_adjusted_return"] = (1 + frame["basket_market_adjusted_open_close_return"].fillna(0)).cumprod() - 1
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def fit_basket_test(
    y: pd.Series,
    signal_group: pd.DataFrame,
    minimum: int,
    config: dict[str, Any],
    seed_offset: int,
) -> dict[str, Any]:
    dates = [str(value) for value in y.index]
    x_series = signal_group.set_index("target_trading_date")["signal_value"].reindex(dates)
    x = x_series.to_numpy(float)
    y_values = y.to_numpy(float)
    fit = hac_ols(y_values, x, int(config["regression"]["hac_maxlags"]))
    valid_mask = np.isfinite(x) & np.isfinite(y_values)
    n = int(valid_mask.sum())
    base: dict[str, Any] = {
        "status": "valid" if fit["status"] == "valid" and n >= minimum else "insufficient",
        "N": n,
        "start_date": dates[int(np.flatnonzero(valid_mask)[0])] if n else "",
        "end_date": dates[int(np.flatnonzero(valid_mask)[-1])] if n else "",
        "minimum_required": minimum,
        "hac_maxlags": int(config["regression"]["hac_maxlags"]),
        "bootstrap_block_length": int(config["regression"]["block_bootstrap_length"]),
        "bootstrap_repetitions": int(config["regression"]["block_bootstrap_repetitions"]),
        "insufficient_reason": "" if fit["status"] == "valid" and n >= minimum else (fit.get("reason") or f"{n} valid pairs < {minimum}"),
    }
    if base["status"] != "valid":
        return base
    beta = float(fit["beta"][1])
    se = float(fit["se"][1])
    df = int(fit["df"])
    critical = float(student_t.ppf(0.975, df=max(df, 1)))
    signal_sd = float(np.nanstd(x[valid_mask], ddof=1))
    low, high, reps = block_bootstrap_slopes(
        x,
        y_values[:, None],
        minimum,
        int(config["regression"]["block_bootstrap_length"]),
        int(config["regression"]["block_bootstrap_repetitions"]),
        int(config["regression"]["random_seed"]) + seed_offset,
    )
    bootstrap_excludes = bool(np.isfinite(low[0]) and np.isfinite(high[0]) and (low[0] > 0 or high[0] < 0))
    base.update({
        "alpha": float(fit["beta"][0]),
        "beta": beta,
        "hac_standard_error": se,
        "hac_t_value": float(fit["t"][1]),
        "raw_p_value": float(fit["p"][1]),
        "beta_ci_low": beta - critical * se,
        "beta_ci_high": beta + critical * se,
        "signal_sample_sd": signal_sd,
        "standardized_effect_bps": beta * signal_sd * 10000,
        "standardized_ci_low_bps": (beta - critical * se) * signal_sd * 10000,
        "standardized_ci_high_bps": (beta + critical * se) * signal_sd * 10000,
        "r_squared": float(fit["r2"]),
        "bootstrap_repetitions_valid": int(reps[0]),
        "bootstrap_beta_ci_low": float(low[0]),
        "bootstrap_beta_ci_high": float(high[0]),
        "bootstrap_ci_excludes_zero": bootstrap_excludes,
        "bootstrap_direction_agrees": bool(bootstrap_excludes and np.sign(beta) == np.sign((low[0] + high[0]) / 2)),
    })
    return base


def run_full_sample_tests(
    specs: list[dict[str, Any]],
    baskets: pd.DataFrame,
    signals: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    signal_groups = {(int(topic_id), str(index_type)): group.copy()
                     for (topic_id, index_type), group in signals.groupby(["topic_id", "index_type"], sort=True)}
    for basket_number, spec in enumerate(specs):
        daily = baskets.loc[baskets["basket_id"].eq(spec["basket_id"])].copy()
        daily["date_key"] = pd.to_datetime(daily["date"]).dt.strftime("%Y-%m-%d")
        y = daily.set_index("date_key")["basket_market_adjusted_open_close_return"]
        test_keys = list(signal_groups) if spec["scope"] == "union" else [(int(spec["topic_id"]), str(spec["index_type"]))]
        for local_number, key in enumerate(test_keys):
            topic_id, index_type = key
            signal_group = signal_groups[key]
            fit = fit_basket_test(y, signal_group, int(config["regression"]["minimum_observations"]), config,
                                  basket_number * 100 + local_number)
            rows.append({
                "family": "post_selection_midterm_baskets",
                "inference_scope": "in_sample_post_selection_exploratory",
                "basket_id": spec["basket_id"],
                "scope": spec["scope"],
                "direction_mode": spec["direction_mode"],
                "selection_rule": str(spec["components"]["selection_rule"].iloc[0]),
                "component_count": int(spec["components"]["ticker"].nunique()),
                "topic_id": topic_id,
                "topic_name": str(signal_group["topic_name"].iloc[0]),
                "index_type": index_type,
                **fit,
            })
    result = pd.DataFrame(rows)
    result["q_value"] = np.nan
    valid = result["status"].eq("valid") & result["raw_p_value"].notna()
    result.loc[valid, "q_value"] = bh_adjust(result.loc[valid, "raw_p_value"].astype(float))
    result["fdr_significant"] = valid & result["q_value"].lt(float(config["regression"]["fdr_alpha"]))
    result["fdr_and_bootstrap_supported"] = result["fdr_significant"] & result["bootstrap_ci_excludes_zero"].fillna(False)
    return result.sort_values(["q_value", "basket_id"], na_position="last").reset_index(drop=True)


def early_selection(
    returns: pd.DataFrame,
    signals: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str]:
    market_dates = sorted(pd.to_datetime(returns["date"].dropna().unique()))
    split_position = max(1, min(len(market_dates) - 1, int(np.ceil(len(market_dates) * float(config["holdout"]["training_fraction"])))))
    training_dates = market_dates[:split_position]
    validation_dates = market_dates[split_position:]
    training_end = pd.Timestamp(training_dates[-1]).strftime("%Y-%m-%d")
    validation_start = pd.Timestamp(validation_dates[0]).strftime("%Y-%m-%d")
    tickers = sorted(returns["ticker"].unique())
    wide = returns.pivot(index="date", columns="ticker", values="market_adjusted_open_close_return").reindex(market_dates).reindex(columns=tickers)
    date_keys = [pd.Timestamp(value).strftime("%Y-%m-%d") for value in market_dates]
    training_mask = np.array([pd.Timestamp(value) <= pd.Timestamp(training_dates[-1]) for value in market_dates])
    minimum_selection = int(config["holdout"]["minimum_selection_observations"])
    minimum_components = int(config["baskets"]["minimum_components"])
    top_n = int(config["holdout"]["top_n"])
    component_frames: list[pd.DataFrame] = []
    specs: list[dict[str, Any]] = []

    for (topic_id, index_type), group in signals.groupby(["topic_id", "index_type"], sort=True):
        x = group.set_index("target_trading_date")["signal_value"].reindex(date_keys).to_numpy(float)
        rows = []
        for ticker in tickers:
            y = wide[ticker].to_numpy(float)
            train_x = np.where(training_mask, x, np.nan)
            train_y = np.where(training_mask, y, np.nan)
            fit = hac_ols(train_y, train_x, int(config["regression"]["hac_maxlags"]))
            if fit["status"] == "valid" and int(fit["n"]) >= minimum_selection:
                rows.append({"ticker": ticker, "N_selection": int(fit["n"]), "selection_beta": float(fit["beta"][1]),
                             "selection_p": float(fit["p"][1]), "selection_t": float(fit["t"][1])})
        if not rows:
            continue
        candidates = pd.DataFrame(rows).sort_values(["selection_p", "ticker"])
        candidates["selection_q_within_signal"] = bh_adjust(candidates["selection_p"])
        choices = [
            ("EARLY_FDR", candidates.loc[candidates["selection_q_within_signal"].lt(float(config["holdout"]["within_signal_selection_fdr"]))],
             "early_period_within_signal_fdr_q_lt_0.05"),
            ("EARLY_TOP20", candidates.head(top_n), "early_period_top20_by_hac_p"),
        ]
        for label, selected, rule in choices:
            if len(selected) < minimum_components:
                continue
            for mode in ("long_only_equal_weight", "direction_aligned_equal_weight"):
                basket_id = f"{label}_{'LONG' if mode.startswith('long') else 'DIRECTIONAL'}__{signal_key(int(topic_id), str(index_type))}"
                comp = selected.copy()
                comp["basket_id"] = basket_id
                comp["selection_rule"] = rule
                comp["scope"] = "signal_specific"
                comp["direction_mode"] = mode
                comp["topic_id"] = int(topic_id)
                comp["topic_name"] = str(group["topic_name"].iloc[0])
                comp["index_type"] = str(index_type)
                comp["selection_sign"] = np.sign(comp["selection_beta"])
                comp["source_min_q"] = np.nan
                comp["source_associations"] = signal_key(int(topic_id), str(index_type))
                comp["selection_period_end"] = training_end
                comp["validation_period_start"] = validation_start
                keep = ["basket_id", "ticker", "selection_rule", "scope", "direction_mode", "topic_id", "topic_name",
                        "index_type", "selection_beta", "selection_sign", "selection_p", "selection_q_within_signal",
                        "selection_t", "N_selection", "source_min_q", "source_associations", "selection_period_end",
                        "validation_period_start"]
                comp = comp[keep]
                component_frames.append(comp)
                specs.append({"basket_id": basket_id, "scope": "signal_specific", "direction_mode": mode,
                              "topic_id": int(topic_id), "index_type": str(index_type), "components": comp})
    components = pd.concat(component_frames, ignore_index=True) if component_frames else pd.DataFrame()
    all_daily = construct_daily_baskets(specs, returns) if specs else pd.DataFrame()
    validation_daily = all_daily.loc[pd.to_datetime(all_daily["date"]).ge(pd.Timestamp(validation_start))].copy() if len(all_daily) else all_daily
    rows = []
    signal_groups = {(int(t), str(i)): g.copy() for (t, i), g in signals.groupby(["topic_id", "index_type"], sort=True)}
    for number, spec in enumerate(specs):
        daily = validation_daily.loc[validation_daily["basket_id"].eq(spec["basket_id"])].copy()
        daily["date_key"] = pd.to_datetime(daily["date"]).dt.strftime("%Y-%m-%d")
        y = daily.set_index("date_key")["basket_market_adjusted_open_close_return"]
        key = (int(spec["topic_id"]), str(spec["index_type"]))
        fit = fit_basket_test(y, signal_groups[key], int(config["holdout"]["minimum_validation_observations"]), config, 5000 + number)
        rows.append({
            "family": "chronological_early_selection_late_validation",
            "inference_scope": "late_period_holdout_exploratory_short_sample",
            "basket_id": spec["basket_id"],
            "direction_mode": spec["direction_mode"],
            "selection_rule": str(spec["components"]["selection_rule"].iloc[0]),
            "component_count": int(spec["components"]["ticker"].nunique()),
            "selection_end_date": training_end,
            "validation_start_date": validation_start,
            "topic_id": key[0],
            "topic_name": str(signal_groups[key]["topic_name"].iloc[0]),
            "index_type": key[1],
            **fit,
        })
    tests = pd.DataFrame(rows)
    if len(tests):
        tests["q_value"] = np.nan
        valid = tests["status"].eq("valid") & tests["raw_p_value"].notna()
        tests.loc[valid, "q_value"] = bh_adjust(tests.loc[valid, "raw_p_value"].astype(float))
        tests["fdr_significant"] = valid & tests["q_value"].lt(float(config["regression"]["fdr_alpha"]))
        tests["fdr_and_bootstrap_supported"] = tests["fdr_significant"] & tests["bootstrap_ci_excludes_zero"].fillna(False)
        tests = tests.sort_values(["q_value", "basket_id"], na_position="last").reset_index(drop=True)
    return components, validation_daily, tests, training_end, validation_start


def draw_results(results: pd.DataFrame, path: Path, title: str, subtitle: str) -> None:
    valid = results.loc[results["status"].eq("valid")].sort_values("q_value").head(25).copy()
    width, height = 1500, max(520, 130 + 38 * max(1, len(valid)))
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(28, True)
    text_font = load_font(16)
    small_font = load_font(14)
    draw.text((35, 24), title, font=title_font, fill="#111827")
    draw.text((35, 66), subtitle, font=small_font, fill="#4b5563")
    if valid.empty:
        draw.text((35, 130), "No valid basket tests.", font=text_font, fill="#991b1b")
    else:
        scale = max(1.0, float(valid["standardized_effect_bps"].abs().max()))
        zero_x = 940
        draw.line((zero_x, 105, zero_x, height - 35), fill="#6b7280", width=2)
        for i, row in enumerate(valid.itertuples()):
            y = 122 + i * 38
            label = f"{row.basket_id} vs T{int(row.topic_id)} {row.index_type}"
            effect = float(row.standardized_effect_bps)
            bar = effect / scale * 420
            color = "#2563eb" if effect >= 0 else "#dc2626"
            draw.text((35, y - 9), label[:88], font=small_font, fill="#111827")
            draw.rectangle((min(zero_x, zero_x + bar), y - 10, max(zero_x, zero_x + bar), y + 10), fill=color)
            marker = "FDR+boot" if bool(row.fdr_and_bootstrap_supported) else ("FDR" if bool(row.fdr_significant) else "n.s.")
            draw.text((1390, y - 9), f"{effect:+.1f} bps  {marker}", font=small_font, fill="#111827", anchor="ra")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def validate(
    returns: pd.DataFrame,
    components: pd.DataFrame,
    daily: pd.DataFrame,
    tests: pd.DataFrame,
    holdout_components: pd.DataFrame,
    holdout_daily: pd.DataFrame,
    holdout_tests: pd.DataFrame,
) -> dict[str, Any]:
    return_tickers = set(returns["ticker"].astype(str))
    checks = {
        "full_component_keys_unique": not components.duplicated(["basket_id", "ticker"]).any(),
        "full_components_have_returns": set(components["ticker"].astype(str)).issubset(return_tickers),
        "full_daily_keys_unique": not daily.duplicated(["basket_id", "date"]).any(),
        "full_test_keys_unique": not tests.duplicated(["basket_id", "topic_id", "index_type"]).any(),
        "full_q_values_in_range": bool(tests["q_value"].dropna().between(0, 1).all()),
        "full_valid_effects_finite": bool(np.isfinite(tests.loc[tests["status"].eq("valid"), "standardized_effect_bps"]).all()),
        "no_missing_returns_filled_with_zero": bool(daily.loc[daily["valid_component_count"].eq(0), "basket_market_adjusted_open_close_return"].isna().all()),
        "holdout_component_keys_unique": True if holdout_components.empty else not holdout_components.duplicated(["basket_id", "ticker"]).any(),
        "holdout_daily_keys_unique": True if holdout_daily.empty else not holdout_daily.duplicated(["basket_id", "date"]).any(),
        "holdout_test_keys_unique": True if holdout_tests.empty else not holdout_tests.duplicated(["basket_id", "topic_id", "index_type"]).any(),
        "holdout_selection_precedes_validation": True if holdout_tests.empty else bool((pd.to_datetime(holdout_tests["selection_end_date"]) < pd.to_datetime(holdout_tests["validation_start_date"])).all()),
    }
    return {"created_at_utc": utc_now(), "validation_status": "pass" if all(checks.values()) else "fail", "checks": checks}


def create_report(
    output_dir: Path,
    full_tests: pd.DataFrame,
    components: pd.DataFrame,
    holdout_tests: pd.DataFrame,
    training_end: str,
    validation_start: str,
    validation: dict[str, Any],
) -> None:
    full_valid = full_tests.loc[full_tests["status"].eq("valid")]
    full_sig = full_valid.loc[full_valid["fdr_significant"]]
    full_robust = full_valid.loc[full_valid["fdr_and_bootstrap_supported"]]
    hold_valid = holdout_tests.loc[holdout_tests["status"].eq("valid")] if len(holdout_tests) else holdout_tests
    hold_sig = hold_valid.loc[hold_valid["fdr_significant"]] if len(hold_valid) else hold_valid
    lines = [
        "# Mid-term equity basket 检验",
        "",
        f"生成时间：{utc_now()}",
        "",
        "## 设计",
        "",
        "- 收益保持原项目口径：复权开盘至收盘股票收益减去同区间 SPY 收益。",
        "- 全样本篮子优先使用股票层面全局 BH-FDR q<0.05 的股票；严格版本进一步要求 5 日移动区块 bootstrap 区间排除 0。",
        "- long 版本等权持有成分股；directional 版本按原股票回归 beta 的符号调整方向后等权。",
        "- 每个 signal-specific 篮子只检验用于选择它的同一主题和指标。union 篮子作为广义 mid-term basket 检验全部实际存在的 18 个指标。",
        "- 全样本篮子使用了同一时期的显著性结果选股，显著性属于事后构造，不能解释为独立确认。",
        "",
        "## 全样本探索结果",
        "",
        f"- 构建 {components['basket_id'].nunique()} 个篮子，共 {len(full_valid)} 个有效篮子检验。",
        f"- 在本轮篮子检验家族内统一 BH-FDR 后，q<0.05 为 {len(full_sig)} 个；其中同时通过篮子层面区块 bootstrap 为 {len(full_robust)} 个。",
    ]
    if len(full_valid):
        lines += ["", "| Basket | Topic/index | Components | N | Effect bps/1SD | raw p | q | bootstrap |",
                  "|---|---|---:|---:|---:|---:|---:|---|"]
        for row in full_valid.sort_values("q_value").head(25).itertuples():
            lines.append(f"| {row.basket_id} | T{int(row.topic_id)} {row.index_type} | {int(row.component_count)} | {int(row.N)} | {float(row.standardized_effect_bps):+.2f} | {float(row.raw_p_value):.4g} | {float(row.q_value):.4g} | {'yes' if bool(row.bootstrap_ci_excludes_zero) else 'no'} |")
    lines += [
        "",
        "## 时间顺序留出检查",
        "",
        f"- 成分选择期截至 {training_end}；验证期从 {validation_start} 开始。验证期收益没有参与成分选择。",
        "- early-FDR 使用选择期内按单一信号跨股票校正后的 q<0.05；early-top20 只使用选择期 HAC p 值最小的 20 只股票。",
        f"- 留出期有效检验 {len(hold_valid)} 个，留出检验家族 BH-FDR q<0.05 为 {len(hold_sig)} 个。由于验证期很短，最低仅要求 8 个有效信号日，全部结果仍属探索性。",
    ]
    if len(hold_valid):
        lines += ["", "| Basket | Topic/index | Components | N | Effect bps/1SD | raw p | q | bootstrap |",
                  "|---|---|---:|---:|---:|---:|---:|---|"]
        for row in hold_valid.sort_values("q_value").head(25).itertuples():
            lines.append(f"| {row.basket_id} | T{int(row.topic_id)} {row.index_type} | {int(row.component_count)} | {int(row.N)} | {float(row.standardized_effect_bps):+.2f} | {float(row.raw_p_value):.4g} | {float(row.q_value):.4g} | {'yes' if bool(row.bootstrap_ci_excludes_zero) else 'no'} |")
    lines += [
        "",
        "## 解释限制",
        "",
        "1. 全样本显著股票篮子具有明确的数据挖掘和 winner's curse；其很强的 p 值不能当作新的独立证据。",
        "2. 留出期只有约四成交易日，稀疏新闻指标的有效 N 更少；无显著结果不等于不存在关系。",
        "3. directional 篮子包含按历史符号做多或做空的统计因子，不是考虑手续费、借券、滑点和容量后的可交易策略。",
        "4. Guardian 文本呈现的情绪和党派态度不代表选民偏好；相关关系也不构成因果关系。",
        "",
        f"验证状态：`{validation['validation_status']}`。",
    ]
    (output_dir / "midterm_basket_analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="midterm_basket_config.yaml")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    output_dir = ROOT / config["project"]["output_dir"]
    figure_dir = ROOT / config["project"]["figure_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    returns, signals, fdr, strict = load_inputs(config)
    specs, components = build_full_sample_specs(fdr, strict, config)
    daily = construct_daily_baskets(specs, returns)
    full_tests = run_full_sample_tests(specs, daily, signals, config)
    hold_components, hold_daily, hold_tests, training_end, validation_start = early_selection(returns, signals, config)

    write_csv(output_dir / "midterm_basket_components.csv", components)
    write_csv(output_dir / "midterm_basket_daily_returns.csv", daily)
    write_csv(output_dir / "midterm_basket_tests.csv", full_tests)
    write_csv(output_dir / "holdout_basket_components.csv", hold_components)
    write_csv(output_dir / "holdout_basket_daily_returns.csv", hold_daily)
    write_csv(output_dir / "holdout_basket_tests.csv", hold_tests)

    validation = validate(returns, components, daily, full_tests, hold_components, hold_daily, hold_tests)
    write_json(output_dir / "midterm_basket_validation.json", validation)
    full_valid = full_tests.loc[full_tests["status"].eq("valid")]
    hold_valid = hold_tests.loc[hold_tests["status"].eq("valid")] if len(hold_tests) else hold_tests
    summary = {
        "created_at_utc": utc_now(),
        "full_sample_baskets": int(components["basket_id"].nunique()),
        "full_sample_valid_tests": int(len(full_valid)),
        "full_sample_fdr_significant": int(full_valid["fdr_significant"].sum()),
        "full_sample_fdr_and_bootstrap_supported": int(full_valid["fdr_and_bootstrap_supported"].sum()),
        "holdout_selection_end": training_end,
        "holdout_validation_start": validation_start,
        "holdout_baskets": int(hold_components["basket_id"].nunique()) if len(hold_components) else 0,
        "holdout_valid_tests": int(len(hold_valid)),
        "holdout_fdr_significant": int(hold_valid["fdr_significant"].sum()) if len(hold_valid) else 0,
        "holdout_fdr_and_bootstrap_supported": int(hold_valid["fdr_and_bootstrap_supported"].sum()) if len(hold_valid) else 0,
        "validation_status": validation["validation_status"],
        "warning": "Full-sample baskets are post-selection and do not provide independent confirmatory inference.",
    }
    write_json(output_dir / "midterm_basket_summary.json", summary)
    create_report(output_dir, full_tests, components, hold_tests, training_end, validation_start, validation)
    draw_results(full_tests, figure_dir / "midterm_basket_full_sample_results.png",
                 "Mid-term baskets: post-selection full-sample tests",
                 "Standardized market-adjusted intraday effect; FDR is within the basket-test family. Full-sample selection is exploratory.")
    draw_results(hold_tests, figure_dir / "midterm_basket_holdout_results.png",
                 "Mid-term baskets: early-selection / late-validation tests",
                 f"Selection through {training_end}; validation from {validation_start}. Short validation samples remain exploratory.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if validation["validation_status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
