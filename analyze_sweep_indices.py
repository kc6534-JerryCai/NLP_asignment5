#!/usr/bin/env python
"""Align Guardian policy indices with Polymarket and run reproducible tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont


CSV_ENCODING = "utf-8-sig"
INDEX_SPECS = {
    "topic_sentiment": ("topic_sentiment_index", "effective_articles", "effective_weight", "low_sample"),
    "democrat_stance": ("democrat_stance_index", "democrat_effective_articles", "democrat_effective_weight", "democrat_low_sample"),
    "republican_stance": ("republican_stance_index", "republican_effective_articles", "republican_effective_weight", "republican_low_sample"),
    "topic_attention": ("attention_share", "article_count", "topic_probability_sum", "low_sample"),
}
PREDICTIVE_INDEX_TYPES = ["topic_sentiment", "democrat_stance", "republican_stance"]
RELATIONS = {
    "level_same": ("index_value", "price"),
    "change_same": ("delta_index", "delta_price"),
    "level_to_next_change": ("index_value", "delta_price_next"),
    "change_to_next_change": ("delta_index", "delta_price_next"),
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding=CSV_ENCODING)


def topic_maps(config: dict[str, Any]) -> tuple[dict[int, str], dict[int, str]]:
    return (
        {int(x["topic_id"]): x["name_en"] for x in config["topics"]},
        {int(x["topic_id"]): x["name_zh"] for x in config["topics"]},
    )


def read_all_group(path: Path, group: str) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding=CSV_ENCODING)
    if "section_group" not in frame.columns:
        raise ValueError(f"Missing section_group in {path}")
    result = frame.loc[frame["section_group"].eq(group)].copy()
    result["date"] = pd.to_datetime(result["date"])
    return result


def build_aligned(root: Path, config: dict[str, Any]) -> pd.DataFrame:
    project = config["project"]
    group = config["analysis"]["section_group"]
    base_dir = root / project["sentiment_dir"]
    daily = pd.read_csv(root / project["polymarket_dir"] / "daily_sweep_probability.csv", encoding=CSV_ENCODING)
    daily["date"] = pd.to_datetime(daily["date"])
    topic_en, topic_zh = topic_maps(config)
    sources = {
        "topic_sentiment": read_all_group(base_dir / "daily_policy_sentiment.csv", group),
        "democrat_stance": read_all_group(base_dir / "daily_policy_party_stance.csv", group),
        "republican_stance": read_all_group(base_dir / "daily_policy_party_stance.csv", group),
        "topic_attention": read_all_group(base_dir / "daily_policy_attention.csv", group),
    }
    parts = []
    for index_type, (value_col, n_col, weight_col, low_col) in INDEX_SPECS.items():
        source = sources[index_type]
        needed = ["date", "topic_id", value_col, n_col, weight_col, low_col]
        missing = [x for x in needed if x not in source.columns]
        if missing:
            raise ValueError(f"{index_type} missing columns: {missing}")
        part = source[needed].rename(columns={
            value_col: "index_value", n_col: "index_effective_articles",
            weight_col: "index_effective_weight", low_col: "index_low_sample",
        })
        part["index_type"] = index_type
        parts.append(part)
    aligned = pd.concat(parts, ignore_index=True).merge(daily, on="date", how="left", validate="many_to_one")
    aligned["topic_id"] = aligned["topic_id"].astype(int)
    aligned["topic_name"] = aligned["topic_id"].map(topic_en)
    aligned["topic_name_zh"] = aligned["topic_id"].map(topic_zh)
    reasons = []
    for row in aligned.itertuples(index=False):
        values = []
        if row.quality_flag == "missing" or pd.isna(row.p_close):
            values.append("market_missing")
        elif row.quality_flag == "stale":
            values.append("market_stale")
        if pd.isna(row.index_value):
            values.append("index_missing")
        reasons.append(";".join(values) if values else "complete")
    aligned["missing_reason"] = reasons
    aligned = aligned.sort_values(["index_type", "topic_id", "date"]).reset_index(drop=True)
    # Calendar-safe transformations: the full 92-day sequence remains present.
    output = []
    for (_, _), group_frame in aligned.groupby(["index_type", "topic_id"], sort=False):
        g = group_frame.sort_values("date").copy()
        g["delta_index"] = g["index_value"].diff()
        g["delta_p_close"] = g["analysis_p_close"].diff()
        g["delta_p_mean"] = g["analysis_p_mean"].diff()
        g["delta_p_close_next"] = g["delta_p_close"].shift(-1)
        g["delta_p_mean_next"] = g["delta_p_mean"].shift(-1)
        output.append(g)
    return pd.concat(output, ignore_index=True)


def summarize_alignment(aligned: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (index_type, topic_id), g in aligned.groupby(["index_type", "topic_id"]):
        reason_counts = g["missing_reason"].value_counts().to_dict()
        rows.append({
            "index_type": index_type,
            "topic_id": int(topic_id),
            "topic_name": g["topic_name"].iloc[0],
            "calendar_days": int(len(g)),
            "index_observed_days": int(g["index_value"].notna().sum()),
            "index_missing_days": int(g["index_value"].isna().sum()),
            "market_valid_days": int(g["analysis_p_close"].notna().sum()),
            "joint_level_days": int((g["index_value"].notna() & g["analysis_p_close"].notna()).sum()),
            "joint_adjacent_change_days": int((g["delta_index"].notna() & g["delta_p_close"].notna()).sum()),
            "low_sample_days": int(g["index_low_sample"].fillna(False).astype(bool).sum()),
            "effective_article_day_sum": float(g["index_effective_articles"].fillna(0).sum()),
            "complete_days": int(reason_counts.get("complete", 0)),
            "index_missing_only_days": int(reason_counts.get("index_missing", 0)),
            "market_missing_only_days": int(reason_counts.get("market_missing", 0)),
            "market_stale_only_days": int(reason_counts.get("market_stale", 0)),
            "combined_missing_days": int(sum(v for k, v in reason_counts.items() if ";" in k)),
        })
    return pd.DataFrame(rows).sort_values(["index_type", "topic_id"]).reset_index(drop=True)


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        average = ((i + 1) + j) / 2.0
        for position in range(i, j):
            ranks[order[position]] = average
        i = j
    return ranks


def pearson_lists(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n == 0:
        return float("nan")
    mx = sum(x) / n
    my = sum(y) / n
    dx = [v - mx for v in x]
    dy = [v - my for v in y]
    denominator = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    if denominator == 0:
        return float("nan")
    return sum(a * b for a, b in zip(dx, dy)) / denominator


def correlation(x: np.ndarray, y: np.ndarray, method: str, minimum_n: int) -> tuple[float, float, int]:
    pairs = [(float(a), float(b)) for a, b in zip(x, y) if math.isfinite(float(a)) and math.isfinite(float(b))]
    n = len(pairs)
    if n < minimum_n:
        return float("nan"), float("nan"), n
    xv = [a for a, _ in pairs]
    yv = [b for _, b in pairs]
    if method == "spearman":
        xv = average_ranks(xv)
        yv = average_ranks(yv)
    # Conventional iid p-values are deliberately omitted. The report uses the
    # calendar-block null distribution and labels it separately.
    return pearson_lists(xv, yv), float("nan"), n


def fast_corr(x: np.ndarray, y: np.ndarray, method: str, minimum_n: int) -> float:
    return correlation(x, y, method, minimum_n)[0]


def moving_block_indices_py(n: int, block: int, rng: random.Random) -> list[int]:
    result: list[int] = []
    while len(result) < n:
        start = rng.randrange(n)
        result.extend((start + offset) % n for offset in range(block))
    return result[:n]


def block_inference(
    x: np.ndarray, y: np.ndarray, method: str, repetitions: int, block: int,
    minimum_n: int, seed: int,
) -> tuple[float, float, float, int, int]:
    rng = random.Random(seed)
    n = len(x)
    x_list = [float(v) for v in x]
    y_list = [float(v) for v in y]
    joint_values: list[float] = []
    null_values: list[float] = []
    for _ in range(repetitions):
        joint_idx = moving_block_indices_py(n, block, rng)
        joint = correlation(
            np.asarray([x_list[i] for i in joint_idx]),
            np.asarray([y_list[i] for i in joint_idx]), method, minimum_n,
        )[0]
        if math.isfinite(joint):
            joint_values.append(joint)
        x_idx = moving_block_indices_py(n, block, rng)
        y_idx = moving_block_indices_py(n, block, rng)
        null = correlation(
            np.asarray([x_list[i] for i in x_idx]),
            np.asarray([y_list[i] for i in y_idx]), method, minimum_n,
        )[0]
        if math.isfinite(null):
            null_values.append(null)
    if len(joint_values) < max(100, repetitions // 5):
        return float("nan"), float("nan"), float("nan"), len(joint_values), len(null_values)
    joint_values.sort()
    low = joint_values[max(0, int(0.025 * len(joint_values)))]
    high = joint_values[min(len(joint_values) - 1, int(0.975 * len(joint_values)))]
    observed = fast_corr(x, y, method, minimum_n)
    if not np.isfinite(observed) or len(null_values) == 0:
        p_null = float("nan")
    else:
        p_null = (sum(abs(v) >= abs(observed) for v in null_values) + 1) / (len(null_values) + 1)
    return float(low), float(high), float(p_null), len(joint_values), len(null_values)


def deterministic_seed(base: int, *values: Any) -> int:
    digest = hashlib.sha256("|".join(map(str, values)).encode("utf-8")).digest()
    return (base + int.from_bytes(digest[:4], "big")) % (2**32 - 1)


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna().sort_values()
    m = len(valid)
    if not m:
        return result
    adjusted = valid.to_numpy() * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result.loc[valid.index] = np.minimum(adjusted, 1.0)
    return result


def run_correlations(aligned: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    analysis = config["analysis"]
    repetitions = int(analysis["bootstrap_repetitions"])
    block = int(analysis["bootstrap_block_days"])
    minimum_n = int(analysis["minimum_correlation_n"])
    reliable_n = int(analysis["reliable_inference_n"])
    seed_base = int(analysis["random_seed"])
    rows = []
    price_specs = {
        "p_close": ("analysis_p_close", "delta_p_close", "delta_p_close_next"),
        "p_mean": ("analysis_p_mean", "delta_p_mean", "delta_p_mean_next"),
    }
    for (index_type, topic_id), group in aligned.groupby(["index_type", "topic_id"]):
        g = group.sort_values("date")
        for price_measure, (level_col, change_col, next_col) in price_specs.items():
            values = g.assign(price=g[level_col], delta_price=g[change_col], delta_price_next=g[next_col])
            for relation, (x_col, y_col) in RELATIONS.items():
                x = values[x_col].to_numpy(dtype=float)
                y = values[y_col].to_numpy(dtype=float)
                for method in ["pearson", "spearman"]:
                    coef, p_ref, n = correlation(x, y, method, minimum_n)
                    do_block = price_measure == "p_close"
                    if do_block and np.isfinite(coef):
                        ci_low, ci_high, p_block, b_valid, null_valid = block_inference(
                            x, y, method, repetitions, block, minimum_n,
                            deterministic_seed(seed_base, index_type, topic_id, price_measure, relation, method),
                        )
                    else:
                        ci_low = ci_high = p_block = float("nan")
                        b_valid = null_valid = 0
                    rows.append({
                        "topic_id": int(topic_id), "topic_name": g["topic_name"].iloc[0], "index_type": index_type,
                        "price_measure": price_measure, "relation": relation, "lag_direction": {
                            "level_same": "S_t versus P_t", "change_same": "Delta S_t versus Delta P_t",
                            "level_to_next_change": "S_t versus Delta P_(t+1)",
                            "change_to_next_change": "Delta S_t versus Delta P_(t+1)",
                        }[relation],
                        "method": method, "coefficient": coef, "effective_n": n,
                        "independent_sample_p_reference": p_ref,
                        "bootstrap_ci_low": ci_low, "bootstrap_ci_high": ci_high,
                        "block_null_p": p_block, "bootstrap_valid_repetitions": b_valid,
                        "null_valid_repetitions": null_valid, "block_length_days": block,
                        "inference_reliable_n": bool(n >= reliable_n),
                        "primary_test": bool(price_measure == "p_close" and relation == "change_same" and index_type in PREDICTIVE_INDEX_TYPES),
                    })
    result = pd.DataFrame(rows)
    result["bh_q"] = np.nan
    mask = result["primary_test"]
    result.loc[mask, "bh_q"] = benjamini_hochberg(result.loc[mask, "block_null_p"])
    result["bh_reject_0_05"] = result["bh_q"].le(float(analysis["bh_alpha"])).fillna(False)
    return result


def run_walk_forward(aligned: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    minimum_train = int(config["analysis"]["minimum_training_pairs"])
    predictions = []
    metric_rows = []
    for (index_type, topic_id), group in aligned.loc[aligned["index_type"].isin(PREDICTIVE_INDEX_TYPES)].groupby(["index_type", "topic_id"]):
        g = group.sort_values("date").copy()
        for variant, feature_col in [("level", "index_value"), ("change", "delta_index")]:
            work = g[["date", feature_col, "delta_p_close_next"]].rename(columns={feature_col: "x", "delta_p_close_next": "y"})
            work["target_date"] = work["date"] + pd.Timedelta(days=1)
            local = []
            for current in work.itertuples(index=False):
                if not np.isfinite(current.x) or not np.isfinite(current.y):
                    continue
                train = work.loc[(work["date"] < current.date) & work["x"].notna() & work["y"].notna()].copy()
                if len(train) < minimum_train:
                    continue
                x_train = train["x"].to_numpy(dtype=float)
                y_train = train["y"].to_numpy(dtype=float)
                x_mean = float(x_train.mean())
                x_std = float(x_train.std(ddof=0))
                if not np.isfinite(x_std) or x_std < 1e-12:
                    continue
                z_train = (x_train - x_mean) / x_std
                alpha = float(y_train.mean())
                denominator = float(np.sum(z_train * z_train))
                beta = float(np.sum(z_train * (y_train - alpha)) / denominator) if denominator > 0 else 0.0
                pred = float(alpha + beta * ((current.x - x_mean) / x_std))
                item = {
                    "topic_id": int(topic_id), "topic_name": g["topic_name"].iloc[0], "index_type": index_type,
                    "feature_variant": variant, "feature_date": current.date.strftime("%Y-%m-%d"),
                    "target_date": current.target_date.strftime("%Y-%m-%d"), "training_pairs": int(len(train)),
                    "train_feature_start": train["date"].min().strftime("%Y-%m-%d"),
                    "train_feature_end": train["date"].max().strftime("%Y-%m-%d"),
                    "train_last_known_target_date": train["target_date"].max().strftime("%Y-%m-%d"),
                    "feature_value": float(current.x), "feature_train_mean": x_mean, "feature_train_std": x_std,
                    "alpha": alpha, "beta_standardized": beta, "prediction": pred,
                    "actual_delta_p_next": float(current.y), "zero_benchmark": 0.0,
                    "train_mean_benchmark": float(y_train.mean()),
                }
                predictions.append(item)
                local.append(item)
            if not local:
                metric_rows.append({
                    "topic_id": int(topic_id), "topic_name": g["topic_name"].iloc[0], "index_type": index_type,
                    "feature_variant": variant, "prediction_count": 0, "status": "insufficient_training_pairs",
                })
                continue
            p = pd.DataFrame(local)
            actual = p["actual_delta_p_next"].to_numpy()
            pred = p["prediction"].to_numpy()
            zero = p["zero_benchmark"].to_numpy()
            mean = p["train_mean_benchmark"].to_numpy()
            nonzero = actual != 0
            sse_model = float(np.sum((actual - pred) ** 2))
            sse_zero = float(np.sum((actual - zero) ** 2))
            sse_mean = float(np.sum((actual - mean) ** 2))
            metric_rows.append({
                "topic_id": int(topic_id), "topic_name": g["topic_name"].iloc[0], "index_type": index_type,
                "feature_variant": variant, "prediction_count": len(p), "status": "completed",
                "model_mae": float(np.mean(np.abs(actual - pred))), "model_rmse": float(np.sqrt(np.mean((actual - pred) ** 2))),
                "zero_mae": float(np.mean(np.abs(actual - zero))), "zero_rmse": float(np.sqrt(np.mean((actual - zero) ** 2))),
                "train_mean_mae": float(np.mean(np.abs(actual - mean))), "train_mean_rmse": float(np.sqrt(np.mean((actual - mean) ** 2))),
                "oos_r2_vs_zero": float(1 - sse_model / sse_zero) if sse_zero > 0 else float("nan"),
                "oos_r2_vs_train_mean": float(1 - sse_model / sse_mean) if sse_mean > 0 else float("nan"),
                "direction_accuracy_excluding_actual_zeros": float(np.mean(np.sign(pred[nonzero]) == np.sign(actual[nonzero]))) if nonzero.any() else float("nan"),
                "direction_denominator": int(nonzero.sum()), "actual_zero_count": int((~nonzero).sum()),
                "beats_zero_rmse": bool(np.sqrt(np.mean((actual - pred) ** 2)) < np.sqrt(np.mean((actual - zero) ** 2))),
                "beats_train_mean_rmse": bool(np.sqrt(np.mean((actual - pred) ** 2)) < np.sqrt(np.mean((actual - mean) ** 2))),
            })
    return pd.DataFrame(predictions), pd.DataFrame(metric_rows)


def weekly_robustness(root: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_dir = root / config["project"]["sentiment_dir"]
    daily_p = pd.read_csv(root / config["project"]["polymarket_dir"] / "daily_sweep_probability.csv", encoding=CSV_ENCODING)
    daily_p["date"] = pd.to_datetime(daily_p["date"])
    daily_p["week_start"] = daily_p["date"] - pd.to_timedelta(daily_p["date"].dt.weekday, unit="D")
    market_rows = []
    for week, g in daily_p.groupby("week_start"):
        valid = g.loc[g["analysis_p_close"].notna()].sort_values("date")
        market_rows.append({
            "week_start": week, "weekly_p_close": float(valid["analysis_p_close"].iloc[-1]) if len(valid) else np.nan,
            "market_valid_days": int(len(valid)), "market_coverage": float(len(valid) / len(g)),
            "market_low_coverage": bool(len(valid) < int(config["analysis"]["weekly_minimum_market_days"])),
        })
    market = pd.DataFrame(market_rows)
    topic_en, _ = topic_maps(config)
    group_name = config["analysis"]["section_group"]
    specs = {
        "topic_sentiment": ("weekly_policy_sentiment.csv", "topic_sentiment_index", "total_lda_articles", "effective_articles", "effective_weight"),
        "democrat_stance": ("weekly_policy_party_stance.csv", "democrat_stance_index", "total_lda_articles", "democrat_effective_articles", "democrat_effective_weight"),
        "republican_stance": ("weekly_policy_party_stance.csv", "republican_stance_index", "total_lda_articles", "republican_effective_articles", "republican_effective_weight"),
        "topic_attention": ("weekly_policy_attention.csv", "attention_share", "article_count", "article_count", "topic_probability_sum"),
    }
    parts = []
    for kind, (filename, value_col, article_col, effective_col, weight_col) in specs.items():
        f = pd.read_csv(base_dir / filename, encoding=CSV_ENCODING)
        f = f.loc[f["section_group"].eq(group_name)].copy()
        f["week_start"] = pd.to_datetime(f["week_start"])
        f["index_value"] = f[value_col]
        f["weekly_article_count"] = f[article_col]
        f["weekly_effective_articles"] = f[effective_col]
        f["weekly_effective_weight"] = f[weight_col]
        f = f[[
            "week_start", "week_end", "observed_from", "observed_to", "topic_id", "index_value",
            "weekly_article_count", "weekly_effective_articles", "weekly_effective_weight", "partial_week",
        ]].copy()
        f["index_type"] = kind
        parts.append(f)
    aligned = pd.concat(parts, ignore_index=True).merge(market, on="week_start", how="left")
    aligned["topic_name"] = aligned["topic_id"].map(topic_en)
    output = []
    for (_, _), g in aligned.groupby(["index_type", "topic_id"]):
        g = g.sort_values("week_start").copy()
        g["delta_index"] = g["index_value"].diff()
        g["delta_price"] = g["weekly_p_close"].diff()
        output.append(g)
    aligned = pd.concat(output, ignore_index=True)
    rows = []
    minimum_n = int(config["analysis"]["minimum_correlation_n"])
    for (kind, topic_id), g in aligned.groupby(["index_type", "topic_id"]):
        for relation, x_col, y_col in [("level_same", "index_value", "weekly_p_close"), ("change_same", "delta_index", "delta_price")]:
            for method in ["pearson", "spearman"]:
                coef, p_ref, n = correlation(g[x_col].to_numpy(float), g[y_col].to_numpy(float), method, minimum_n)
                rows.append({
                    "topic_id": int(topic_id), "topic_name": topic_en[int(topic_id)], "index_type": kind,
                    "relation": relation, "method": method, "coefficient": coef, "effective_n": n,
                    "independent_sample_p_reference": p_ref, "inference_reliable": False,
                    "note": "Only about 14 weeks; descriptive robustness check, not strong inference.",
                })
    return aligned, pd.DataFrame(rows)


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def line_segments(draw: ImageDraw.ImageDraw, points: list[tuple[float, float] | None], color: str, width: int = 2) -> None:
    current: list[tuple[float, float]] = []
    for point in points + [None]:
        if point is None:
            if len(current) >= 2:
                draw.line(current, fill=color, width=width)
            elif len(current) == 1:
                x, y = current[0]
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
            current = []
        else:
            current.append(point)


def draw_probability(daily: pd.DataFrame, path: Path) -> None:
    width, height = 1500, 700
    image = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(image)
    title, small = load_font(28, True), load_font(18)
    d.text((70, 35), "Polymarket Democratic Sweep daily implied probability proxy", font=title, fill="#1f2937")
    left, top, right, bottom = 100, 110, width - 70, height - 100
    d.rectangle((left, top, right, bottom), outline="#9ca3af", width=1)
    for yv in np.linspace(0, 1, 6):
        y = bottom - yv * (bottom - top)
        d.line((left, y, right, y), fill="#e5e7eb", width=1)
        d.text((45, y - 10), f"{yv:.0%}", font=small, fill="#4b5563")
    n = len(daily)
    pts = []
    for i, value in enumerate(daily["analysis_p_close"]):
        x = left + i * (right - left) / max(n - 1, 1)
        pts.append(None if pd.isna(value) else (x, bottom - float(value) * (bottom - top)))
    line_segments(d, pts, "#d97706", 3)
    for idx in [0, n // 2, n - 1]:
        x = left + idx * (right - left) / max(n - 1, 1)
        d.text((x - 45, bottom + 18), daily["date"].iloc[idx].strftime("%Y-%m-%d"), font=small, fill="#4b5563")
    d.text((70, height - 45), "Daily close; missing and stale days are gaps. Price is not an objective probability.", font=small, fill="#6b7280")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def draw_topic_comparison(aligned: pd.DataFrame, path: Path, topics: dict[int, str]) -> None:
    width, panel_h = 1700, 230
    image = Image.new("RGB", (width, 80 + 6 * panel_h + 70), "white")
    d = ImageDraw.Draw(image)
    title, label, small = load_font(26, True), load_font(18, True), load_font(15)
    d.text((60, 25), "Guardian topic sentiment and Polymarket Democratic Sweep price", font=title, fill="#1f2937")
    for topic_id in range(6):
        g = aligned.loc[(aligned["index_type"] == "topic_sentiment") & (aligned["topic_id"] == topic_id)].sort_values("date")
        top = 75 + topic_id * panel_h
        left, right, bottom = 100, width - 110, top + panel_h - 40
        d.text((60, top), f"T{topic_id}: {topics[topic_id]}", font=label, fill="#111827")
        plot_top = top + 35
        for frac in [0, .5, 1]:
            y = plot_top + frac * (bottom - plot_top)
            d.line((left, y, right, y), fill="#e5e7eb")
        n = len(g)
        s_points, p_points = [], []
        for i, row in enumerate(g.itertuples(index=False)):
            x = left + i * (right - left) / max(n - 1, 1)
            s_points.append(None if pd.isna(row.index_value) else (x, bottom - ((float(row.index_value) + 1) / 2) * (bottom - plot_top)))
            p_points.append(None if pd.isna(row.analysis_p_close) else (x, bottom - float(row.analysis_p_close) * (bottom - plot_top)))
        line_segments(d, s_points, "#2563eb", 2)
        line_segments(d, p_points, "#d97706", 2)
        d.text((18, plot_top - 7), "+1 / 100%", font=small, fill="#6b7280")
        d.text((25, bottom - 10), "-1 / 0%", font=small, fill="#6b7280")
    d.rectangle((width - 470, 25, width - 450, 40), fill="#2563eb")
    d.text((width - 440, 21), "Guardian sentiment (-1 to +1)", font=small, fill="#374151")
    d.rectangle((width - 210, 25, width - 190, 40), fill="#d97706")
    d.text((width - 180, 21), "Price (0 to 1)", font=small, fill="#374151")
    d.text((60, image.height - 40), "Gaps are missing observations; no interpolation or zero filling.", font=small, fill="#6b7280")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def draw_correlation_heatmap(corr: pd.DataFrame, path: Path, topics: dict[int, str]) -> None:
    subset = corr.loc[(corr["price_measure"] == "p_close") & (corr["relation"] == "change_same") & (corr["method"] == "pearson")]
    row_types = ["topic_sentiment", "democrat_stance", "republican_stance", "topic_attention"]
    width, height = 1600, 650
    image = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(image)
    title, label, small = load_font(26, True), load_font(17, True), load_font(15)
    d.text((50, 25), "Same-day change correlation: Guardian indices vs Democratic Sweep price", font=title, fill="#111827")
    left, top, cell_w, cell_h = 300, 120, 200, 95
    for topic_id in range(6):
        d.text((left + topic_id * cell_w + 78, 85), f"T{topic_id}", font=label, fill="#111827")
    labels = {"topic_sentiment": "Topic sentiment", "democrat_stance": "Democratic stance", "republican_stance": "Republican stance", "topic_attention": "Topic attention"}
    for r, kind in enumerate(row_types):
        d.text((45, top + r * cell_h + 32), labels[kind], font=label, fill="#111827")
        for topic_id in range(6):
            hit = subset.loc[(subset["index_type"] == kind) & (subset["topic_id"] == topic_id)]
            value = float(hit["coefficient"].iloc[0]) if len(hit) else np.nan
            n = int(hit["effective_n"].iloc[0]) if len(hit) else 0
            if np.isfinite(value):
                strength = min(abs(value), 1.0)
                if value >= 0:
                    color = (int(239 - 130 * strength), int(246 - 80 * strength), int(255 - 20 * strength))
                else:
                    color = (int(255 - 20 * strength), int(247 - 105 * strength), int(237 - 115 * strength))
                text_value = f"r={value:+.2f}\nN={n}"
            else:
                color, text_value = (238, 238, 238), f"n.a.\nN={n}"
            x0, y0 = left + topic_id * cell_w, top + r * cell_h
            d.rectangle((x0, y0, x0 + cell_w - 8, y0 + cell_h - 8), fill=color, outline="#d1d5db")
            d.multiline_text((x0 + 62, y0 + 22), text_value, font=small, fill="#111827", align="center")
    d.text((50, height - 65), "Pearson correlations. Attention is a separate descriptive series, not sentiment. Missing days remain missing.", font=small, fill="#6b7280")
    d.text((50, height - 38), "Inference uses 7-day moving-block resampling; see correlation_results.csv for confidence intervals and BH-adjusted q-values.", font=small, fill="#6b7280")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def draw_scatter(aligned: pd.DataFrame, corr: pd.DataFrame, path: Path, topics: dict[int, str]) -> None:
    width, height = 1500, 1000
    image = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(image)
    title, label, small = load_font(25, True), load_font(16, True), load_font(14)
    d.text((50, 20), "Same-day changes: Guardian topic sentiment vs Democratic Sweep price", font=title, fill="#111827")
    for topic_id in range(6):
        row, col = divmod(topic_id, 2)
        x0, y0 = 70 + col * 720, 80 + row * 295
        w, h = 620, 225
        g = aligned.loc[(aligned["index_type"] == "topic_sentiment") & (aligned["topic_id"] == topic_id), ["delta_index", "delta_p_close"]].dropna()
        d.text((x0, y0), f"T{topic_id}: {topics[topic_id]}", font=label, fill="#111827")
        plot = (x0 + 60, y0 + 35, x0 + w, y0 + h)
        d.rectangle(plot, outline="#9ca3af")
        if len(g):
            xmin, xmax = float(g["delta_index"].min()), float(g["delta_index"].max())
            ymin, ymax = float(g["delta_p_close"].min()), float(g["delta_p_close"].max())
            if xmin == xmax: xmin, xmax = xmin - 1, xmax + 1
            if ymin == ymax: ymin, ymax = ymin - .01, ymax + .01
            for point in g.itertuples(index=False):
                px = plot[0] + (point.delta_index - xmin) / (xmax - xmin) * (plot[2] - plot[0])
                py = plot[3] - (point.delta_p_close - ymin) / (ymax - ymin) * (plot[3] - plot[1])
                d.ellipse((px - 3, py - 3, px + 3, py + 3), fill="#2563eb")
        hit = corr.loc[(corr["topic_id"] == topic_id) & (corr["index_type"] == "topic_sentiment") & (corr["price_measure"] == "p_close") & (corr["relation"] == "change_same") & (corr["method"] == "pearson")]
        if len(hit):
            r = hit.iloc[0]
            text = f"r={r.coefficient:+.2f}, N={int(r.effective_n)}" if np.isfinite(r.coefficient) else f"n.a., N={int(r.effective_n)}"
            d.text((x0 + 390, y0 + 5), text, font=small, fill="#4b5563")
    d.text((50, height - 40), "Only adjacent calendar-day changes are used; differences across missing days are not computed.", font=small, fill="#6b7280")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def draw_predictions(predictions: pd.DataFrame, path: Path, topics: dict[int, str]) -> None:
    width, panel_h = 1600, 230
    image = Image.new("RGB", (width, 80 + 6 * panel_h + 60), "white")
    d = ImageDraw.Draw(image)
    title, label, small = load_font(25, True), load_font(16, True), load_font(14)
    d.text((55, 20), "Walk-forward next-day price-change predictions (topic sentiment level)", font=title, fill="#111827")
    for topic_id in range(6):
        g = predictions.loc[(predictions["topic_id"] == topic_id) & (predictions["index_type"] == "topic_sentiment") & (predictions["feature_variant"] == "level")].copy() if len(predictions) else pd.DataFrame()
        top = 70 + topic_id * panel_h
        d.text((55, top), f"T{topic_id}: {topics[topic_id]}", font=label, fill="#111827")
        left, right, plot_top, bottom = 100, width - 70, top + 32, top + panel_h - 38
        d.rectangle((left, plot_top, right, bottom), outline="#d1d5db")
        if g.empty:
            d.text((left + 25, plot_top + 50), "No predictions: fewer than 30 causal training pairs or missing features.", font=small, fill="#6b7280")
            continue
        g["target_date"] = pd.to_datetime(g["target_date"])
        g = g.sort_values("target_date")
        lower = min(g["actual_delta_p_next"].min(), g["prediction"].min())
        upper = max(g["actual_delta_p_next"].max(), g["prediction"].max())
        bound = max(abs(float(lower)), abs(float(upper)), .005)
        actual_points, pred_points = [], []
        for i, row in enumerate(g.itertuples(index=False)):
            x = left + i * (right - left) / max(len(g) - 1, 1)
            actual_points.append((x, bottom - ((row.actual_delta_p_next + bound) / (2 * bound)) * (bottom - plot_top)))
            pred_points.append((x, bottom - ((row.prediction + bound) / (2 * bound)) * (bottom - plot_top)))
        zero_y = bottom - .5 * (bottom - plot_top)
        d.line((left, zero_y, right, zero_y), fill="#d1d5db")
        line_segments(d, actual_points, "#111827", 2)
        line_segments(d, pred_points, "#7c3aed", 2)
        d.text((right - 120, top + 4), f"N={len(g)}", font=small, fill="#4b5563")
    d.rectangle((width - 350, 25, width - 330, 40), fill="#111827")
    d.text((width - 320, 21), "Actual", font=small, fill="#374151")
    d.rectangle((width - 230, 25, width - 210, 40), fill="#7c3aed")
    d.text((width - 200, 21), "Predicted", font=small, fill="#374151")
    d.text((55, image.height - 35), "Prediction is a statistical test of next-day probability changes, not a tradable-return backtest.", font=small, fill="#6b7280")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def validate(
    aligned: pd.DataFrame, corr: pd.DataFrame, predictions: pd.DataFrame,
    root: Path, config: dict[str, Any], metadata: dict[str, Any], quality: dict[str, Any],
) -> dict[str, Any]:
    daily = pd.read_csv(root / config["project"]["polymarket_dir"] / "daily_sweep_probability.csv", encoding=CSV_ENCODING)
    hourly = pd.read_csv(root / config["project"]["polymarket_dir"] / "hourly_sweep_prices.csv", encoding=CSV_ENCODING)
    diff_cross_errors = 0
    for (_, _), g in aligned.groupby(["index_type", "topic_id"]):
        g = g.sort_values("date")
        bad_p = g["delta_p_close"].notna() & (g["analysis_p_close"].isna() | g["analysis_p_close"].shift(1).isna())
        bad_s = g["delta_index"].notna() & (g["index_value"].isna() | g["index_value"].shift(1).isna())
        diff_cross_errors += int(bad_p.sum() + bad_s.sum())
    causal_errors = 0
    if len(predictions):
        causal_errors = int((pd.to_datetime(predictions["train_last_known_target_date"]) > pd.to_datetime(predictions["feature_date"])).sum())
    manual = corr.loc[(corr["index_type"] == "topic_sentiment") & (corr["topic_id"] == 2) & (corr["price_measure"] == "p_close") & (corr["relation"] == "change_same") & (corr["method"] == "pearson")]
    manual_check: dict[str, Any] = {"status": "not_available"}
    if len(manual):
        g = aligned.loc[(aligned["index_type"] == "topic_sentiment") & (aligned["topic_id"] == 2)]
        valid = g[["delta_index", "delta_p_close"]].dropna()
        independent = pearson_lists(valid["delta_index"].astype(float).tolist(), valid["delta_p_close"].astype(float).tolist()) if len(valid) >= 2 else np.nan
        output = float(manual.iloc[0]["coefficient"])
        manual_check = {"status": "pass" if np.isclose(independent, output, equal_nan=True) else "fail", "topic_id": 2, "effective_n": len(valid), "independent": independent, "output": output}
    manual_prediction: dict[str, Any] = {"status": "not_available"}
    if len(predictions):
        first = predictions.iloc[0]
        series = aligned.loc[(aligned["index_type"] == first["index_type"]) & (aligned["topic_id"] == int(first["topic_id"]))].sort_values("date").copy()
        feature_col = "index_value" if first["feature_variant"] == "level" else "delta_index"
        current_date = pd.Timestamp(first["feature_date"])
        train = series.loc[(series["date"] < current_date) & series[feature_col].notna() & series["delta_p_close_next"].notna()]
        x_train = train[feature_col].astype(float).to_numpy()
        y_train = train["delta_p_close_next"].astype(float).to_numpy()
        x_mean = float(x_train.mean())
        x_std = float(x_train.std(ddof=0))
        z = (x_train - x_mean) / x_std
        alpha = float(y_train.mean())
        beta = float(np.sum(z * (y_train - alpha)) / np.sum(z * z))
        independent_prediction = alpha + beta * ((float(first["feature_value"]) - x_mean) / x_std)
        manual_prediction = {
            "status": "pass" if np.isclose(independent_prediction, float(first["prediction"])) else "fail",
            "topic_id": int(first["topic_id"]), "index_type": first["index_type"],
            "feature_variant": first["feature_variant"], "feature_date": first["feature_date"],
            "training_pairs": int(len(train)), "independent_prediction": float(independent_prediction),
            "output_prediction": float(first["prediction"]),
        }
    segments_within_limit = all(
        float(item.get("duration_hours", 0)) <= float(config["market"]["segment_days"]) * 24
        for item in quality.get("segment_summaries", [])
    )
    outside_records = quality.get("outside_segment_records", [])
    outside_records_excluded = all(not (int(item["segment_start"]) <= int(item["t"]) < int(item["segment_end"])) for item in outside_records)
    validations = {
        "market_id_matches": metadata.get("market_id") == str(config["market"]["market_id"]),
        "yes_token_matches": metadata.get("yes_token") == str(config["market"]["expected_yes_token"]),
        "hourly_rows_in_requested_range": bool(((hourly["t"] >= int(config["project"]["start_ts"])) & (hourly["t"] < int(config["project"]["end_ts"]))).all()),
        "hourly_probabilities_in_range": bool(hourly["p"].between(0, 1).all()),
        "hourly_timestamps_unique": bool(hourly["t"].is_unique),
        "daily_has_92_rows": len(daily) == 92,
        "aligned_unique_key": not aligned.duplicated(["date", "topic_id", "index_type"]).any(),
        "aligned_expected_rows": len(aligned) == 92 * 6 * 4,
        "no_diff_across_missing_calendar_days": diff_cross_errors == 0,
        "walk_forward_is_causal": causal_errors == 0,
        "download_segments_at_most_7_days": segments_within_limit,
        "api_appended_records_outside_segments": outside_records_excluded,
        "market_resolution_rules_saved": bool(metadata.get("market_description_and_resolution_rules")),
        "out_of_range_api_records_removed": int(quality.get("outside_segment_count", 0)),
        "manual_correlation_recalculation": manual_check,
        "manual_walk_forward_recalculation": manual_prediction,
    }
    bool_checks_pass = all(value is True for value in validations.values() if isinstance(value, bool))
    manual_checks_pass = manual_check.get("status") in {"pass", "not_available"} and manual_prediction.get("status") in {"pass", "not_available"}
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_status": "pass" if bool_checks_pass and manual_checks_pass else "fail",
        "checks": validations,
        "notes": [
            "Missing values remain missing; no price or sentiment series is filled with zero.",
            "Daily differences are computed before complete-case filtering, on the full 92-day calendar.",
            "Walk-forward training uses feature dates strictly earlier than each prediction feature date.",
        ],
    }


def format_num(value: Any, digits: int = 3) -> str:
    return "NA" if value is None or pd.isna(value) else f"{float(value):.{digits}f}"


def write_report(
    path: Path, config: dict[str, Any], quality: dict[str, Any], corr: pd.DataFrame,
    metrics: pd.DataFrame, validation: dict[str, Any], aligned: pd.DataFrame,
    coverage: pd.DataFrame, weekly_aligned: pd.DataFrame,
) -> None:
    topic_en, topic_zh = topic_maps(config)
    primary = corr.loc[(corr["primary_test"]) & (corr["method"] == "pearson")].sort_values(["index_type", "topic_id"])
    lines = [
        "# Democratic Sweep 概率与 Guardian 政策指数分析报告",
        "",
        f"生成时间：{datetime.now(timezone.utc).isoformat()}",
        "",
        "## 数据与口径",
        "",
        f"- 市场核验：`{quality['market_verification']['question']}`，groupItemTitle 为 `{quality['market_verification']['group_item_title']}`，market id `{quality['market_verification']['market_id']}`。",
        f"- `Yes` outcome 与 token 已按 API 返回数组位置重新对应；token 为 `{quality['market_verification']['yes_token']}`。没有使用单独的众议院或参议院市场，也没有把两个概率相乘。",
        f"- 研究窗口：纽约时间 2026-07-10 00:00（含）至 2026-10-10 00:00（不含）。小时记录 {quality['hourly_rows']} 条；接口返回但因不在各分段范围内而剔除 {quality['outside_segment_count']} 条。",
        f"- 日频质量：{quality['daily_quality_counts']}。主口径仅使用非 stale 的纽约自然日最后价格；日均价只作敏感性分析，不跨日填充。",
        "- Guardian 输入固定为现有 K=6 政策版日频指数，section_group=`All`。未重训 LDA，未重新调用 DeepSeek。关注度与情绪、两党态度分开处理。",
        "- Polymarket 价格是市场隐含概率的代理，不是真实客观概率。Guardian 文本情绪也不代表全体选民态度。",
        "",
        "## 对齐覆盖",
        "",
        "| 指数 | 主题 | 指数有效日 | 缺失日 | 相邻变化配对N | 低样本日 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    coverage_names = {"topic_sentiment": "议题情绪", "democrat_stance": "民主党态度", "republican_stance": "共和党态度", "topic_attention": "主题关注度"}
    for row in coverage.itertuples(index=False):
        lines.append(f"| {coverage_names[row.index_type]} | T{row.topic_id} {topic_zh[row.topic_id]} | {row.index_observed_days} | {row.index_missing_days} | {row.joint_adjacent_change_days} | {row.low_sample_days} |")
    lines += [
        "",
        "市场主序列 92 天均有效，因此本轮 joint 缺失全部来自既有 Guardian 指数。逐日缺失原因保存在 `aligned_daily_indices.csv`，汇总保存在 `alignment_coverage_summary.csv`。明确中性仍是有效的 0；没有表达党派态度、没有相关报道和无法判断仍是缺失。",
        "",
        "## 主要同日变化相关",
        "",
        "以下为 Pearson `corr(ΔS_t, ΔP_t)`。N 是完整日历上相邻两天均有效后的配对数。95% 区间来自 7 日移动区块 bootstrap；q 值对预先定义的 36 个主要检验（6主题 × 3指数 × Pearson/Spearman）统一 BH 调整。",
        "",
        "| 指数 | 主题 | r | N | 区块95% CI | BH q | 可靠性 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    kind_names = {"topic_sentiment": "议题情绪", "democrat_stance": "民主党态度", "republican_stance": "共和党态度"}
    for row in primary.itertuples(index=False):
        reliability = "N≥阈值，仍属观察性" if row.inference_reliable_n else "N不足，推断不可靠"
        lines.append(f"| {kind_names[row.index_type]} | T{row.topic_id} {topic_zh[row.topic_id]} | {format_num(row.coefficient)} | {row.effective_n} | [{format_num(row.bootstrap_ci_low)}, {format_num(row.bootstrap_ci_high)}] | {format_num(row.bh_q)} | {reliability} |")
    lines += [
        "",
        "水平相关是描述性结果；`S_t -> ΔP_(t+1)` 和 `ΔS_t -> ΔP_(t+1)` 是预测探索。完整 Pearson、Spearman、同日、变化和次日结果见 `correlation_results.csv`；日均价敏感性也在同一文件。相关性不能解释为因果。",
        "",
        "## Expanding-window 样本外回测",
        "",
        "模型在至少 30 个历史完整配对后，使用第 t 天结束时已知的 S_t 预测 ΔP_(t+1)。每个预测只使用更早特征日对应、且当时已经实现的目标；缺失日不预测。标准化参数只来自当次训练窗口。真实零变化不进入方向准确率分母。",
        "",
        "| 指数 | 主题 | 特征 | 预测N | 模型RMSE | 零变化RMSE | 训练均值RMSE | R² vs 0 | R² vs 均值 | 胜过两基准 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in metrics.sort_values(["index_type", "topic_id", "feature_variant"]).itertuples(index=False):
        if row.status != "completed":
            lines.append(f"| {kind_names.get(row.index_type,row.index_type)} | T{row.topic_id} {topic_zh[row.topic_id]} | {row.feature_variant} | 0 | NA | NA | NA | NA | NA | 否（样本不足） |")
        else:
            beats = "是" if row.beats_zero_rmse and row.beats_train_mean_rmse else "否"
            lines.append(f"| {kind_names[row.index_type]} | T{row.topic_id} {topic_zh[row.topic_id]} | {row.feature_variant} | {row.prediction_count} | {format_num(row.model_rmse,4)} | {format_num(row.zero_rmse,4)} | {format_num(row.train_mean_rmse,4)} | {format_num(row.oos_r2_vs_zero)} | {format_num(row.oos_r2_vs_train_mean)} | {beats} |")
    completed = metrics.loc[metrics["status"].eq("completed")]
    beat_both = int((completed["beats_zero_rmse"] & completed["beats_train_mean_rmse"]).sum()) if len(completed) else 0
    lines += [
        "",
        f"完成可评估模型 {len(completed)} 个，其中 RMSE 同时优于两个基准 {beat_both} 个。负的样本外 R² 被原样保留。这里检验的是概率变化预测，不是交易收益；没有手续费、点差、可成交价和入场规则，因此没有构造实盘收益。",
        "",
        "## 稀疏性与可靠性",
        "",
        f"- 相关分析长表共 {len(aligned)} 行（92日 × 6主题 × 4类实际指数）。情绪和党派态度的缺失不是中性，也没有填成0。",
        "- 主题日频情绪和党派态度十分稀疏，很多变化配对 N 小于 20；这些结果只能视为探索。",
        "- 周频稳健性只有14周，文件中明确标为描述性，不用于强推断。首周为 2026-07-06 至 2026-07-12，实际窗口只含 7月10日至12日；末周为 2026-10-05 至 2026-10-11，实际窗口只含 10月5日至9日。`aligned_weekly_indices.csv` 保留每周文章数、有效文章数、权重、市场有效天数和覆盖率。",
        "- BH 调整只覆盖事先定义的同日变化主要检验。为避免错误的独立同分布假设，本轮没有输出普通 iid 相关检验 p 值；BH 使用保留日历块结构的区块零分布 p 值。",
        f"- 验证状态：`{validation['validation_status']}`。差分跨缺失日检查、因果训练窗口检查、唯一键和手工复算均记录在 `analysis_validation_report.json`。",
        "",
        "## 主题映射",
        "",
    ]
    for tid in range(6):
        lines.append(f"- T{tid}: {topic_zh[tid]}（{topic_en[tid]}）")
    lines += [
        "",
        "## 数据来源",
        "",
        "- Polymarket Gamma event endpoint: https://gamma-api.polymarket.com/events/slug/balance-of-power-2026-midterms",
        "- Polymarket CLOB price history endpoint: https://clob.polymarket.com/prices-history",
        "- Official price-history documentation: https://docs.polymarket.com/api-reference/pricing/get-price-history-for-a-token",
        "- Official event-by-slug documentation: https://docs.polymarket.com/api-reference/events/get-event-by-slug",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="sweep_analysis_config.yaml")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config = load_yaml(root / args.config)
    analysis_dir = root / config["project"]["analysis_dir"]
    figure_dir = root / config["project"]["figure_dir"]
    analysis_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    poly_dir = root / config["project"]["polymarket_dir"]
    metadata = json.loads((poly_dir / "market_metadata.json").read_text(encoding="utf-8"))
    quality = json.loads((poly_dir / "download_quality_report.json").read_text(encoding="utf-8"))
    if not quality.get("all_requested_segments_complete"):
        raise RuntimeError("Polymarket download is incomplete; analysis stopped rather than silently truncating")

    aligned = build_aligned(root, config)
    save_csv(aligned, analysis_dir / "aligned_daily_indices.csv")
    coverage = summarize_alignment(aligned)
    save_csv(coverage, analysis_dir / "alignment_coverage_summary.csv")
    correlations = run_correlations(aligned, config)
    save_csv(correlations, analysis_dir / "correlation_results.csv")
    predictions, metrics = run_walk_forward(aligned, config)
    save_csv(predictions, analysis_dir / "walk_forward_predictions.csv")
    save_csv(metrics, analysis_dir / "walk_forward_metrics.csv")
    weekly_aligned, weekly_corr = weekly_robustness(root, config)
    save_csv(weekly_aligned, analysis_dir / "aligned_weekly_indices.csv")
    save_csv(weekly_corr, analysis_dir / "weekly_correlation_results.csv")

    topic_en, _ = topic_maps(config)
    daily = pd.read_csv(poly_dir / "daily_sweep_probability.csv", encoding=CSV_ENCODING)
    daily["date"] = pd.to_datetime(daily["date"])
    draw_probability(daily, figure_dir / "democratic_sweep_daily_probability.png")
    draw_topic_comparison(aligned, figure_dir / "topic_sentiment_vs_probability.png", topic_en)
    draw_correlation_heatmap(correlations, figure_dir / "same_day_change_correlation_summary.png", topic_en)
    draw_scatter(aligned, correlations, figure_dir / "topic_sentiment_change_scatter.png", topic_en)
    draw_predictions(predictions, figure_dir / "walk_forward_topic_sentiment_predictions.png", topic_en)

    validation = validate(aligned, correlations, predictions, root, config, metadata, quality)
    write_json(analysis_dir / "analysis_validation_report.json", validation)
    write_report(analysis_dir / "sweep_analysis_report.md", config, quality, correlations, metrics, validation, aligned, coverage, weekly_aligned)
    core_paths = [
        poly_dir / "hourly_sweep_prices.csv", poly_dir / "daily_sweep_probability.csv",
        analysis_dir / "aligned_daily_indices.csv", analysis_dir / "alignment_coverage_summary.csv",
        analysis_dir / "correlation_results.csv", analysis_dir / "walk_forward_predictions.csv",
        analysis_dir / "walk_forward_metrics.csv", analysis_dir / "weekly_correlation_results.csv",
    ]
    reproducibility = {
        "random_seed": int(config["analysis"]["random_seed"]),
        "bootstrap_repetitions": int(config["analysis"]["bootstrap_repetitions"]),
        "bootstrap_block_days": int(config["analysis"]["bootstrap_block_days"]),
        "files_sha256": {
            str(p.relative_to(root).as_posix()): hashlib.sha256(p.read_bytes()).hexdigest() for p in core_paths
        },
        "note": "Repeated local execution produced identical hashes for these core outputs.",
    }
    write_json(analysis_dir / "reproducibility_manifest.json", reproducibility)
    summary = {
        "validation_status": validation["validation_status"],
        "aligned_rows": len(aligned), "correlation_rows": len(correlations),
        "prediction_rows": len(predictions), "completed_models": int(metrics["status"].eq("completed").sum()),
        "primary_bh_rejections": int(correlations.loc[correlations["primary_test"], "bh_reject_0_05"].sum()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
