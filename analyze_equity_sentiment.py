#!/usr/bin/env python
"""Relate frozen Guardian policy sentiment indices to broad US equity returns.

Main design:
* Article-level LDA-probability weighted signals, recomputed for a strictly
  pre-open news window.
* Adjusted open-to-close stock return minus the identical SPY interval.
* Fixed HAC(5) inference, calendar-preserving moving-block bootstrap, and one
  global BH-FDR family across every valid stock x sentiment-index test.
* Expanding-window predictions with no random split or zero-filled signals.

The script does not retrain LDA, call DeepSeek, or alter source news data.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import chi2, t as student_t


ROOT = Path(__file__).resolve().parent
CSV_ENCODING = "utf-8-sig"
MAIN_TYPES = ("topic_sentiment", "democrat_stance", "republican_stance")
TYPE_LABELS = {
    "topic_sentiment": "Topic sentiment",
    "democrat_stance": "Democratic stance",
    "republican_stance": "Republican stance",
    "attention": "Topic attention",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame, compressed: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compressed:
        frame.to_csv(path, index=False, encoding="utf-8", compression="gzip")
    else:
        frame.to_csv(path, index=False, encoding=CSV_ENCODING)


def to_bool(series: pd.Series) -> pd.Series:
    return series.map(lambda x: str(x).strip().lower() == "true" if pd.notna(x) else False)


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    for name in (["arialbd.ttf", "segoeuib.ttf"] if bold else ["arial.ttf", "segoeui.ttf"]):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def load_topics(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    topics = sorted(payload["topics"], key=lambda x: int(x["topic_id"]))
    if [int(t["topic_id"]) for t in topics] != list(range(6)):
        raise ValueError("Expected frozen K=6 topic definitions with topic IDs 0..5")
    return topics


def prepare_returns(config: dict[str, Any], out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[pd.Timestamp]]:
    prices = pd.read_csv(out_dir / "raw_daily_prices.csv", encoding="utf-8-sig")
    universe = pd.read_csv(out_dir / "frozen_stock_universe.csv", encoding="utf-8-sig")
    benchmark = str(config["prices"]["benchmark"])
    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits", "Capital Gains"]
    for col in numeric_cols:
        prices[col] = pd.to_numeric(prices[col], errors="coerce")
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices = prices.dropna(subset=["date", "ticker"]).sort_values(["ticker", "date"])
    input_duplicate_keys = int(prices.duplicated(["ticker", "date"]).sum())
    prices = prices.drop_duplicates(["ticker", "date"], keep="last")
    valid_factor = prices["Close"].gt(0) & prices["Adj Close"].notna()
    prices["adjustment_factor"] = np.where(valid_factor, prices["Adj Close"] / prices["Close"], np.nan)
    prices["adjusted_open"] = prices["Open"] * prices["adjustment_factor"]
    prices["adjusted_close"] = prices["Adj Close"]
    prices["open_close_return"] = np.where(prices["adjusted_open"].gt(0), prices["adjusted_close"] / prices["adjusted_open"] - 1, np.nan)

    spy = prices.loc[prices["ticker"].eq(benchmark) & prices["open_close_return"].notna()].copy()
    if spy.empty:
        raise ValueError("SPY benchmark is missing or has no valid adjusted open-to-close returns")
    full_market_dates = sorted(spy["date"].unique())
    close_wide = prices.pivot(index="date", columns="ticker", values="adjusted_close").reindex(full_market_dates)
    previous_close = close_wide.shift(1)
    ctc = close_wide / previous_close - 1
    # shift(1) on the complete SPY calendar prevents returns from jumping across a missing stock day.
    ctc_long = ctc.rename_axis(index="date", columns="ticker").reset_index().melt(
        id_vars="date", var_name="ticker", value_name="close_to_close_return")
    prices = prices.merge(ctc_long, on=["date", "ticker"], how="left", validate="one_to_one")
    spy_oc = prices.loc[prices["ticker"].eq(benchmark)].set_index("date")["open_close_return"]
    spy_cc = prices.loc[prices["ticker"].eq(benchmark)].set_index("date")["close_to_close_return"]
    prices["spy_open_close_return"] = prices["date"].map(spy_oc)
    prices["spy_close_to_close_return"] = prices["date"].map(spy_cc)
    prices["market_adjusted_open_close_return"] = prices["open_close_return"] - prices["spy_open_close_return"]
    prices["market_adjusted_close_to_close_return"] = prices["close_to_close_return"] - prices["spy_close_to_close_return"]
    prices["extreme_open_close_return"] = prices["open_close_return"].abs().gt(float(config["prices"]["outlier_abs_return_threshold"]))

    start = pd.Timestamp(config["project"]["start_date"])
    end = pd.Timestamp(config["project"]["end_date_inclusive"])
    study_dates = [pd.Timestamp(x) for x in full_market_dates if start <= pd.Timestamp(x) <= end]
    study = prices.loc[prices["date"].isin(study_dates)].copy()
    write_csv(out_dir / "adjusted_equity_returns.csv", study)

    requested = universe["ticker"].astype(str).tolist()
    rows = []
    minimum = int(config["prices"]["minimum_stock_trading_days"])
    for ticker in requested:
        sub = study.loc[study["ticker"].eq(ticker)]
        n = int(sub["market_adjusted_open_close_return"].notna().sum())
        if sub.empty:
            status, reason = "not_downloaded", "no yfinance rows"
        elif n < minimum:
            status, reason = "insufficient", f"{n} valid open-to-close market-adjusted returns < {minimum}"
        else:
            status, reason = "testable", ""
        rows.append({"ticker": ticker, "status": status, "valid_trading_days": n,
                     "first_valid_date": sub.loc[sub["market_adjusted_open_close_return"].notna(), "date"].min(),
                     "last_valid_date": sub.loc[sub["market_adjusted_open_close_return"].notna(), "date"].max(),
                     "extreme_return_days": int(sub["extreme_open_close_return"].sum()), "exclusion_reason": reason})
    testability = pd.DataFrame(rows)
    write_csv(out_dir / "stock_testability.csv", testability)
    testable = testability.loc[testability["status"].eq("testable"), "ticker"].tolist()
    write_json(out_dir / "return_construction_report.json", {
        "created_at_utc": utc_now(), "input_rows": len(prices), "input_duplicate_keys_removed": input_duplicate_keys,
        "study_trading_days": len(study_dates), "first_trading_day": min(study_dates).date(), "last_trading_day": max(study_dates).date(),
        "frozen_stocks": len(requested), "testable_stocks": len(testable), "benchmark": benchmark,
        "return_definition": "adjusted_close / (open * Adj Close / Close) - 1; stock return minus identical SPY interval",
        "close_to_close_missing_rule": "previous value is the immediately prior SPY trading-day row; a missing stock row makes the return missing",
    })
    return study, testability, testable, study_dates


def score_spec(index_type: str) -> tuple[str, str, str]:
    if index_type == "topic_sentiment":
        return "topic_sentiment_score", "topic_sentiment_label", "topic_sentiment_evidence_valid"
    if index_type == "democrat_stance":
        return "democrat_stance_score", "democrat_stance_label", "democrat_evidence_valid"
    if index_type == "republican_stance":
        return "republican_stance_score", "republican_stance_label", "republican_evidence_valid"
    raise KeyError(index_type)


def aggregate_signal_range(scored: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp,
                           target_date: pd.Timestamp, topics: list[dict[str, Any]], config: dict[str, Any],
                           window_type: str) -> list[dict[str, Any]]:
    window = scored.loc[(scored["published_ts"] >= start_ts) & (scored["published_ts"] < end_ts)]
    rows: list[dict[str, Any]] = []
    valid_status = window["processing_status"].isin(config["signals"]["success_statuses"])
    for topic in topics:
        topic_id = int(topic["topic_id"])
        subset = window.loc[window["topic_id"].eq(topic_id)].copy()
        status = valid_status.reindex(subset.index, fill_value=False)
        total_articles = len(subset)
        total_weight = float(subset["LDA_probability"].sum()) if total_articles else 0.0
        latest_any = subset["published_ts"].max() if total_articles else pd.NaT
        unique_news_dates = int(subset["published_ts"].dt.date.nunique()) if total_articles else 0
        for index_type in MAIN_TYPES:
            score_col, label_col, evidence_col = score_spec(index_type)
            eligible = status & subset["relevant"] & subset[score_col].notna() & subset[evidence_col]
            effective = subset.loc[eligible]
            weight = float(effective["LDA_probability"].sum())
            numerator = float((effective["LDA_probability"] * effective[score_col]).sum())
            value = numerator / weight if weight > 0 else np.nan
            counts = effective[label_col].value_counts().to_dict()
            rows.append({
                "target_trading_date": target_date.strftime("%Y-%m-%d"), "window_type": window_type,
                "window_start_new_york": start_ts.isoformat(), "window_end_new_york_exclusive": end_ts.isoformat(),
                "topic_id": topic_id, "topic_name": topic["english_name"], "index_type": index_type,
                "signal_value": value, "total_articles": total_articles, "total_topic_weight": total_weight,
                "effective_articles": len(effective), "effective_weight": weight,
                "article_coverage": len(effective) / total_articles if total_articles else np.nan,
                "weight_coverage": weight / total_weight if total_weight > 0 else np.nan,
                "news_calendar_days": unique_news_dates,
                "latest_news_timestamp_new_york": effective["published_ts"].max().isoformat() if len(effective) else "",
                "latest_any_article_timestamp_new_york": latest_any.isoformat() if pd.notna(latest_any) else "",
                "label_counts_json": json.dumps(counts, ensure_ascii=False, sort_keys=True),
                "low_sample": len(effective) < int(config["signals"]["low_sample_articles"]),
                "missing_reason": "" if pd.notna(value) else ("no articles in news window" if total_articles == 0 else "no eligible scored expression"),
            })
        attention = total_weight / total_articles if total_articles else np.nan
        rows.append({
            "target_trading_date": target_date.strftime("%Y-%m-%d"), "window_type": window_type,
            "window_start_new_york": start_ts.isoformat(), "window_end_new_york_exclusive": end_ts.isoformat(),
            "topic_id": topic_id, "topic_name": topic["english_name"], "index_type": "attention",
            "signal_value": attention, "total_articles": total_articles, "total_topic_weight": total_weight,
            "effective_articles": total_articles, "effective_weight": total_weight,
            "article_coverage": 1.0 if total_articles else np.nan, "weight_coverage": 1.0 if total_articles else np.nan,
            "news_calendar_days": unique_news_dates,
            "latest_news_timestamp_new_york": latest_any.isoformat() if pd.notna(latest_any) else "",
            "latest_any_article_timestamp_new_york": latest_any.isoformat() if pd.notna(latest_any) else "",
            "label_counts_json": "{}", "low_sample": total_articles < int(config["signals"]["low_sample_articles"]),
            "missing_reason": "" if pd.notna(attention) else "no articles in news window",
        })
    return rows


def build_news_signals(config: dict[str, Any], out_dir: Path, topics: list[dict[str, Any]],
                       study_dates: list[pd.Timestamp], all_price_dates: list[pd.Timestamp]) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = pd.read_csv(ROOT / config["project"]["article_scores"], encoding="utf-8-sig")
    scored["topic_id"] = pd.to_numeric(scored["topic_id"], errors="raise").astype(int)
    for col in ["LDA_probability", "topic_sentiment_score", "democrat_stance_score", "republican_stance_score"]:
        scored[col] = pd.to_numeric(scored[col], errors="coerce")
    scored["relevant"] = to_bool(scored["relevant"])
    for col in ["topic_sentiment_evidence_valid", "democrat_evidence_valid", "republican_evidence_valid"]:
        scored[col] = to_bool(scored[col])
    scored["published_ts"] = pd.to_datetime(scored["published_at_new_york"], utc=True, errors="raise").dt.tz_convert(config["project"]["timezone"])
    if scored.duplicated(["article_id", "topic_id"]).any():
        raise ValueError("article_id + topic_id is not unique in article sentiment input")
    full_dates = sorted(pd.Timestamp(x) for x in all_price_dates)
    previous = {full_dates[i]: full_dates[i - 1] for i in range(1, len(full_dates))}
    tz = config["project"]["timezone"]
    preopen_rows: list[dict[str, Any]] = []; same_day_rows: list[dict[str, Any]] = []
    for target in study_dates:
        if target not in previous:
            continue
        start = pd.Timestamp(datetime.combine(previous[target].date(), time.min), tz=tz)
        end = pd.Timestamp(datetime.combine(target.date(), time.min), tz=tz)
        preopen_rows.extend(aggregate_signal_range(scored, start, end, target, topics, config, "preopen"))
        next_midnight = end + pd.Timedelta(days=1)
        same_day_rows.extend(aggregate_signal_range(scored, end, next_midnight, target, topics, config, "same_calendar_day_descriptive"))
    preopen = pd.DataFrame(preopen_rows); same_day = pd.DataFrame(same_day_rows)
    write_csv(out_dir / "preopen_news_signals.csv", preopen)
    write_csv(out_dir / "same_day_news_signals.csv", same_day)
    return preopen, same_day


def invert_small(matrix: np.ndarray) -> np.ndarray:
    """Gauss-Jordan inverse for the 2x2--4x4 matrices used here.

    The project's Windows NumPy build can hang inside BLAS/LAPACK.  This tiny
    deterministic implementation avoids that native DLL path.
    """
    n = int(matrix.shape[0])
    augmented = [[float(matrix[i, j]) for j in range(n)] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(augmented[r][col]))
        if abs(augmented[pivot][col]) < 1e-14:
            raise ValueError("singular matrix")
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        divisor = augmented[col][col]
        augmented[col] = [v / divisor for v in augmented[col]]
        for row in range(n):
            if row == col: continue
            factor = augmented[row][col]
            augmented[row] = [augmented[row][j] - factor * augmented[col][j] for j in range(2 * n)]
    return np.array([row[n:] for row in augmented], dtype=float)


def matmul_small(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([[sum(float(a[i, k]) * float(b[k, j]) for k in range(a.shape[1]))
                      for j in range(b.shape[1])] for i in range(a.shape[0])], dtype=float)


def matvec_small(a: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.array([sum(float(a[i, k]) * float(v[k]) for k in range(a.shape[1])) for i in range(a.shape[0])], dtype=float)


def condition_number_1(matrix: np.ndarray, inverse: np.ndarray) -> float:
    norm = max(sum(abs(float(matrix[i, j])) for i in range(matrix.shape[0])) for j in range(matrix.shape[1]))
    inv_norm = max(sum(abs(float(inverse[i, j])) for i in range(inverse.shape[0])) for j in range(inverse.shape[1]))
    return norm * inv_norm


def hac_ols(y_full: np.ndarray, x_full: np.ndarray, maxlags: int) -> dict[str, Any]:
    if x_full.ndim == 1: x_full = x_full[:, None]
    valid = np.isfinite(y_full) & np.isfinite(x_full).all(axis=1)
    n = int(valid.sum()); p = x_full.shape[1] + 1
    if n <= p:
        return {"status": "insufficient", "n": n, "reason": "fewer observations than parameters"}
    Xfull = np.column_stack([np.ones(len(y_full)), x_full])
    X = Xfull[valid]; y = y_full[valid]
    if any(float(np.nanstd(X[:, j], ddof=1)) <= 1e-10 for j in range(1, X.shape[1])):
        return {"status": "insufficient", "n": n, "reason": "near-constant signal"}
    xtx = np.array([[float(np.sum(X[:, i] * X[:, j])) for j in range(p)] for i in range(p)], dtype=float)
    try:
        inv = invert_small(xtx)
    except ValueError:
        return {"status": "insufficient", "n": n, "reason": "singular design"}
    xty = np.array([float(np.sum(X[:, i] * y)) for i in range(p)], dtype=float)
    beta = matvec_small(inv, xty)
    residual = np.full(len(y_full), np.nan)
    fitted_valid = sum(X[:, j] * beta[j] for j in range(p))
    residual[valid] = y - fitted_valid
    S = np.zeros((p, p), dtype=float)
    for t_idx in np.flatnonzero(valid):
        v = Xfull[t_idx]
        S += residual[t_idx] ** 2 * np.outer(v, v)
    for lag in range(1, min(maxlags, len(y_full) - 1) + 1):
        weight = 1 - lag / (maxlags + 1)
        for t_idx in range(lag, len(y_full)):
            if not (valid[t_idx] and valid[t_idx - lag]):
                continue
            cross = residual[t_idx] * residual[t_idx - lag]
            S += weight * cross * (np.outer(Xfull[t_idx], Xfull[t_idx - lag]) + np.outer(Xfull[t_idx - lag], Xfull[t_idx]))
    covariance = (n / max(1, n - p)) * matmul_small(matmul_small(inv, S), inv)
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    t_values = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    p_values = 2 * student_t.sf(np.abs(t_values), df=max(1, n - p))
    fitted = fitted_valid
    sst = float(((y - y.mean()) ** 2).sum()); sse = float(((y - fitted) ** 2).sum())
    r2 = 1 - sse / sst if sst > 0 else np.nan
    return {"status": "valid", "n": n, "beta": beta, "se": se, "t": t_values, "p": p_values,
            "covariance": covariance, "r2": r2, "valid_mask": valid, "residual_full": residual,
            "condition_number": condition_number_1(xtx, inv), "df": n - p}


def block_bootstrap_slopes(x: np.ndarray, y_matrix: np.ndarray, minimum: int, block_length: int,
                           repetitions: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed); T, n_stocks = y_matrix.shape
    slopes = np.full((repetitions, n_stocks), np.nan, dtype=np.float32)
    blocks_needed = math.ceil(T / block_length)
    for rep in range(repetitions):
        starts = rng.integers(0, max(1, T - block_length + 1), size=blocks_needed)
        idx = np.concatenate([np.arange(s, min(s + block_length, T)) for s in starts])[:T]
        xb = x[idx]; yb = y_matrix[idx, :]
        mask = np.isfinite(xb[:, None]) & np.isfinite(yb)
        n = mask.sum(axis=0)
        xx = np.where(mask, xb[:, None], 0.0); yy = np.where(mask, yb, 0.0)
        sx = xx.sum(axis=0); sy = yy.sum(axis=0)
        sxx = (xx * xx).sum(axis=0); sxy = (xx * yy).sum(axis=0)
        denominator = n * sxx - sx * sx
        valid = (n >= minimum) & (np.abs(denominator) > 1e-14)
        slopes[rep, valid] = ((n * sxy - sx * sy)[valid] / denominator[valid]).astype(np.float32)
    with np.errstate(all="ignore"):
        low = np.nanpercentile(slopes, 2.5, axis=0); high = np.nanpercentile(slopes, 97.5, axis=0)
    valid_reps = np.isfinite(slopes).sum(axis=0)
    return low, high, valid_reps


def bh_adjust(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float); m = len(p)
    if not m: return np.array([], dtype=float)
    order = np.argsort(p); ranked = p[order]
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty(m); result[order] = np.minimum(adjusted, 1.0)
    return result


def apply_fdr(frame: pd.DataFrame, p_col: str = "raw_p_value") -> pd.DataFrame:
    frame = frame.copy()
    if p_col not in frame.columns: frame[p_col] = np.nan
    frame["q_value"] = np.nan
    valid = frame["status"].eq("valid") & frame[p_col].notna()
    frame.loc[valid, "q_value"] = bh_adjust(frame.loc[valid, p_col].astype(float))
    frame["result_category"] = "insufficient"
    frame.loc[valid, "result_category"] = "not_significant"
    frame.loc[valid & frame[p_col].lt(0.05), "result_category"] = "nominal_only"
    frame.loc[valid & frame["q_value"].ge(0.05) & frame["q_value"].lt(0.10), "result_category"] = "exploratory_q_0.05_to_0.10"
    frame.loc[valid & frame["q_value"].lt(0.05), "result_category"] = "fdr_significant_q_lt_0.05"
    return frame


def signals_to_wide(signals: pd.DataFrame, index_types: tuple[str, ...] | list[str]) -> pd.DataFrame:
    selected = signals.loc[signals["index_type"].isin(index_types)].copy()
    selected["signal_id"] = selected.apply(lambda r: f"T{int(r.topic_id)}_{r.index_type}", axis=1)
    return selected.pivot(index="target_trading_date", columns="signal_id", values="signal_value")


def run_univariate_family(returns: pd.DataFrame, tickers: list[str], signals: pd.DataFrame,
                          config: dict[str, Any], family_name: str, return_col: str,
                          bootstrap: bool) -> pd.DataFrame:
    dates = sorted(returns["date"].dt.strftime("%Y-%m-%d").unique())
    y_wide = returns.loc[returns["ticker"].isin(tickers)].pivot(index="date", columns="ticker", values=return_col)
    y_wide.index = y_wide.index.strftime("%Y-%m-%d"); y_wide = y_wide.reindex(dates).reindex(columns=tickers)
    y_matrix = y_wide.to_numpy(float)
    minimum = int(config["regression"]["minimum_observations"]); maxlags = int(config["regression"]["hac_maxlags"])
    topic_lookup = signals.drop_duplicates(["topic_id"])[["topic_id", "topic_name"]].set_index("topic_id")["topic_name"].to_dict()
    rows: list[dict[str, Any]] = []
    for signal_number, ((topic_id, index_type), group) in enumerate(signals.groupby(["topic_id", "index_type"], sort=True)):
        x_series = group.set_index("target_trading_date")["signal_value"].reindex(dates)
        x = x_series.to_numpy(float)
        if bootstrap and int(np.isfinite(x).sum()) >= minimum:
            boot_low, boot_high, boot_reps = block_bootstrap_slopes(
                x, y_matrix, minimum, int(config["regression"]["block_bootstrap_length"]),
                int(config["regression"]["block_bootstrap_repetitions"]), int(config["regression"]["random_seed"]) + signal_number)
        else:
            boot_low = np.full(len(tickers), np.nan); boot_high = np.full(len(tickers), np.nan); boot_reps = np.zeros(len(tickers), dtype=int)
        for j, ticker in enumerate(tickers):
            result = hac_ols(y_matrix[:, j], x, maxlags)
            if result["n"] < minimum:
                result = {"status": "insufficient", "n": result["n"],
                          "reason": f"{result['n']} valid pairs < fixed minimum {minimum}"}
            base = {"family": family_name, "ticker": ticker, "topic_id": int(topic_id), "topic_name": topic_lookup[int(topic_id)],
                    "index_type": index_type, "return_definition": return_col, "status": result["status"], "N": result["n"],
                    "start_date": x_series.index[np.flatnonzero(np.isfinite(x) & np.isfinite(y_matrix[:, j]))[0]] if np.any(np.isfinite(x) & np.isfinite(y_matrix[:, j])) else "",
                    "end_date": x_series.index[np.flatnonzero(np.isfinite(x) & np.isfinite(y_matrix[:, j]))[-1]] if np.any(np.isfinite(x) & np.isfinite(y_matrix[:, j])) else "",
                    "hac_maxlags": maxlags, "bootstrap_block_length": int(config["regression"]["block_bootstrap_length"]),
                    "bootstrap_repetitions_valid": int(boot_reps[j]), "bootstrap_beta_ci_low": boot_low[j], "bootstrap_beta_ci_high": boot_high[j],
                    "insufficient_reason": result.get("reason", "")}
            if result["status"] == "valid":
                beta = float(result["beta"][1]); se = float(result["se"][1]); df = int(result["df"])
                crit = float(student_t.ppf(0.975, df=max(1, df))); x_sd = float(np.nanstd(x[np.isfinite(x) & np.isfinite(y_matrix[:, j])], ddof=1))
                base.update({"alpha": float(result["beta"][0]), "beta": beta, "hac_standard_error": se,
                             "hac_t_value": float(result["t"][1]), "raw_p_value": float(result["p"][1]),
                             "beta_ci_low": beta - crit * se, "beta_ci_high": beta + crit * se,
                             "signal_sample_sd": x_sd, "standardized_effect_bps": beta * x_sd * 10000,
                             "standardized_ci_low_bps": (beta - crit * se) * x_sd * 10000,
                             "standardized_ci_high_bps": (beta + crit * se) * x_sd * 10000,
                             "r_squared": float(result["r2"])})
                boot_excludes = bool(np.isfinite(boot_low[j]) and np.isfinite(boot_high[j]) and (boot_low[j] > 0 or boot_high[j] < 0))
                base["bootstrap_ci_excludes_zero"] = boot_excludes
                base["bootstrap_direction_agrees"] = bool(boot_excludes and np.sign(beta) == np.sign((boot_low[j] + boot_high[j]) / 2))
            rows.append(base)
    return apply_fdr(pd.DataFrame(rows))


def build_alignment(returns: pd.DataFrame, tickers: list[str], preopen: pd.DataFrame, out_dir: Path) -> None:
    stock = returns.loc[returns["ticker"].isin(tickers), ["date", "ticker", "open_close_return", "spy_open_close_return",
        "market_adjusted_open_close_return", "close_to_close_return", "spy_close_to_close_return", "market_adjusted_close_to_close_return"]].copy()
    stock["target_trading_date"] = stock["date"].dt.strftime("%Y-%m-%d"); stock = stock.drop(columns="date")
    sig = preopen.loc[preopen["index_type"].isin(MAIN_TYPES)].copy()
    aligned = stock.merge(sig, on="target_trading_date", how="left", validate="many_to_many")
    aligned["target_market_open_new_york"] = aligned["target_trading_date"] + "T09:30:00-04:00"
    write_csv(out_dir / "stock_signal_alignment.csv.gz", aligned, compressed=True)


def walk_forward(returns: pd.DataFrame, tickers: list[str], signals: pd.DataFrame, config: dict[str, Any],
                 return_col: str, family: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(returns["date"].dt.strftime("%Y-%m-%d").unique())
    y_wide = returns.loc[returns["ticker"].isin(tickers)].pivot(index="date", columns="ticker", values=return_col)
    y_wide.index = y_wide.index.strftime("%Y-%m-%d"); y_wide = y_wide.reindex(dates).reindex(columns=tickers)
    y_matrix = y_wide.to_numpy(float); minimum = int(config["walk_forward"]["minimum_training_pairs"])
    tol = float(config["walk_forward"]["zero_direction_tolerance"])
    detail_rows: list[dict[str, Any]] = []; metric_rows: list[dict[str, Any]] = []
    for (topic_id, index_type), group in signals.groupby(["topic_id", "index_type"], sort=True):
        topic_name = str(group["topic_name"].iloc[0]); x = group.set_index("target_trading_date")["signal_value"].reindex(dates).to_numpy(float)
        for j, ticker in enumerate(tickers):
            local: list[dict[str, Any]] = []
            for pos in range(len(dates)):
                if not (np.isfinite(x[pos]) and np.isfinite(y_matrix[pos, j])): continue
                train_mask = np.isfinite(x[:pos]) & np.isfinite(y_matrix[:pos, j])
                n_train = int(train_mask.sum())
                if n_train < minimum: continue
                xtr = x[:pos][train_mask]; ytr = y_matrix[:pos, j][train_mask]
                xmean = float(xtr.mean()); xsd = float(xtr.std(ddof=1)); ymean = float(ytr.mean())
                if not np.isfinite(xsd) or xsd <= float(config["regression"]["near_constant_tolerance"]): continue
                z = (xtr - xmean) / xsd
                slope = float(np.sum(z * (ytr - ymean)) / np.sum(z * z)); prediction = ymean + slope * ((x[pos] - xmean) / xsd)
                row = {"family": family, "ticker": ticker, "topic_id": int(topic_id), "topic_name": topic_name,
                       "index_type": index_type, "target_date": dates[pos], "training_end_date": dates[pos - 1],
                       "training_pairs": n_train, "train_signal_mean": xmean, "train_signal_sd": xsd,
                       "signal_value": x[pos], "predicted_return": prediction, "actual_return": y_matrix[pos, j],
                       "zero_benchmark_prediction": 0.0, "historical_mean_prediction": ymean}
                local.append(row); detail_rows.append(row)
            if local:
                d = pd.DataFrame(local); actual = d["actual_return"].to_numpy(float); pred = d["predicted_return"].to_numpy(float)
                zero = d["zero_benchmark_prediction"].to_numpy(float); meanp = d["historical_mean_prediction"].to_numpy(float)
                rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
                mae = lambda a, b: float(np.mean(np.abs(a - b)))
                valid_direction = np.abs(actual) > tol
                metric_rows.append({"family": family, "ticker": ticker, "topic_id": int(topic_id), "topic_name": topic_name,
                    "index_type": index_type, "prediction_count": len(d), "mae_model": mae(actual, pred), "rmse_model": rmse(actual, pred),
                    "mae_zero": mae(actual, zero), "rmse_zero": rmse(actual, zero), "mae_historical_mean": mae(actual, meanp),
                    "rmse_historical_mean": rmse(actual, meanp),
                    "oos_r2_vs_zero": 1 - np.sum((actual - pred) ** 2) / np.sum((actual - zero) ** 2) if np.sum(actual ** 2) > 0 else np.nan,
                    "oos_r2_vs_historical_mean": 1 - np.sum((actual - pred) ** 2) / np.sum((actual - meanp) ** 2) if np.sum((actual - meanp) ** 2) > 0 else np.nan,
                    "direction_accuracy": float(np.mean(np.sign(pred[valid_direction]) == np.sign(actual[valid_direction]))) if valid_direction.any() else np.nan,
                    "zero_actuals_excluded_from_direction": int((~valid_direction).sum()),
                    "beats_zero_rmse": rmse(actual, pred) < rmse(actual, zero),
                    "beats_historical_mean_rmse": rmse(actual, pred) < rmse(actual, meanp),
                    "evidence_status": "evaluable" if len(d) >= int(config["walk_forward"]["minimum_predictions_for_evidence"]) else "fewer_than_15_predictions"})
            else:
                metric_rows.append({"family": family, "ticker": ticker, "topic_id": int(topic_id), "topic_name": topic_name,
                                    "index_type": index_type, "prediction_count": 0, "evidence_status": "no_predictions"})
    return pd.DataFrame(detail_rows), pd.DataFrame(metric_rows)


def prepare_baskets(returns: pd.DataFrame, mapping: pd.DataFrame, topics: list[dict[str, Any]], config: dict[str, Any],
                    out_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    benchmark = config["prices"]["benchmark"]
    stock_returns = returns.loc[~returns["ticker"].eq(benchmark)].copy()
    dates = sorted(returns.loc[returns["ticker"].eq(benchmark), "date"].unique())
    rows = []
    for topic in topics:
        topic_id = int(topic["topic_id"]); components = sorted(mapping.loc[mapping["topic_id"].eq(topic_id), "ticker"].unique())
        for dt in dates:
            sub = stock_returns.loc[stock_returns["date"].eq(dt) & stock_returns["ticker"].isin(components)]
            valid = sub.loc[sub["open_close_return"].notna()]
            market = sub["spy_open_close_return"].dropna()
            basket_return = float(valid["open_close_return"].mean()) if len(valid) else np.nan
            spy_return = float(market.iloc[0]) if len(market) else np.nan
            rows.append({"date": dt, "ticker": f"BASKET_T{topic_id}", "topic_id": topic_id, "topic_name": topic["english_name"],
                         "components": "|".join(components), "component_count": len(components), "valid_component_count": len(valid),
                         "open_close_return": basket_return, "spy_open_close_return": spy_return,
                         "market_adjusted_open_close_return": basket_return - spy_return if pd.notna(basket_return) and pd.notna(spy_return) else np.nan})
    baskets = pd.DataFrame(rows).sort_values(["topic_id", "date"])
    baskets["cumulative_market_adjusted_return"] = baskets.groupby("topic_id")["market_adjusted_open_close_return"].transform(
        lambda s: (1 + s).cumprod() - 1)
    write_csv(out_dir / "topic_basket_returns.csv", baskets)
    return baskets, sorted(baskets["ticker"].unique())


def multivariate_hac(y: np.ndarray, X: np.ndarray, maxlags: int) -> dict[str, Any]:
    return hac_ols(y, X, maxlags)


def run_joint_models(returns: pd.DataFrame, signals: pd.DataFrame, config: dict[str, Any], out_dir: Path) -> pd.DataFrame:
    dates = sorted(returns["date"].dt.strftime("%Y-%m-%d").unique())
    wide = signals_to_wide(signals, list(MAIN_TYPES)).reindex(dates)
    rows: list[dict[str, Any]] = []
    for spec in config["joint_models"].get("predefined", []):
        ticker = spec["ticker"]; topics = [int(x) for x in spec["topic_ids"]]; index_type = spec["index_type"]
        columns = [f"T{k}_{index_type}" for k in topics]
        y = returns.loc[returns["ticker"].eq(ticker)].set_index(returns.loc[returns["ticker"].eq(ticker), "date"].dt.strftime("%Y-%m-%d"))["market_adjusted_open_close_return"].reindex(dates).to_numpy(float)
        X = wide[columns].to_numpy(float)
        minimum = max(int(config["joint_models"]["minimum_observations"]), int(config["joint_models"]["observations_per_parameter"]) * (len(columns) + 1))
        result = multivariate_hac(y, X, int(config["regression"]["hac_maxlags"]))
        valid_n = int((np.isfinite(y) & np.isfinite(X).all(axis=1)).sum())
        corr = pd.DataFrame(X, columns=columns).corr().abs().where(~np.eye(len(columns), dtype=bool)).max().max() if len(columns) > 1 else np.nan
        base = {"model_name": spec["name"], "ticker": ticker, "index_type": index_type, "topic_ids": "|".join(map(str, topics)),
                "signal_columns": "|".join(columns), "N": valid_n, "minimum_required": minimum,
                "max_absolute_signal_correlation": corr, "status": "insufficient" if valid_n < minimum else result["status"],
                "condition_number": result.get("condition_number", np.nan)}
        if valid_n >= minimum and result["status"] == "valid":
            slopes = result["beta"][1:]; cov = result["covariance"][1:, 1:]
            try:
                inv_cov = invert_small(cov); wald = float(np.sum(slopes * matvec_small(inv_cov, slopes))); joint_p = float(chi2.sf(wald, df=len(slopes)))
            except ValueError:
                wald = joint_p = np.nan
            base.update({"alpha": float(result["beta"][0]), "coefficients_json": json.dumps(dict(zip(columns, map(float, slopes))), sort_keys=True),
                         "standard_errors_json": json.dumps(dict(zip(columns, map(float, result["se"][1:]))), sort_keys=True),
                         "raw_joint_p_value": joint_p, "wald_chi2": wald, "r_squared": result["r2"]})
        rows.append(base)
    frame = pd.DataFrame(rows)
    if "raw_joint_p_value" not in frame.columns: frame["raw_joint_p_value"] = np.nan
    frame["joint_q_value"] = np.nan
    valid = frame["status"].eq("valid") & frame["raw_joint_p_value"].notna()
    frame.loc[valid, "joint_q_value"] = bh_adjust(frame.loc[valid, "raw_joint_p_value"])
    write_csv(out_dir / "joint_model_results.csv", frame)
    return frame


def robustness_models(returns: pd.DataFrame, signals: pd.DataFrame, mapping: pd.DataFrame,
                      config: dict[str, Any], out_dir: Path) -> pd.DataFrame:
    dates = sorted(returns["date"].dt.strftime("%Y-%m-%d").unique())
    signal_wide = signals_to_wide(signals, list(MAIN_TYPES)).reindex(dates)
    etf_returns = returns.pivot(index="date", columns="ticker", values="open_close_return"); etf_returns.index = etf_returns.index.strftime("%Y-%m-%d")
    rows: list[dict[str, Any]] = []
    for _, item in mapping.drop_duplicates(["ticker", "topic_id"]).iterrows():
        ticker = str(item["ticker"]); topic_id = int(item["topic_id"]); sector_etf = str(item["sector_etf"])
        stock = returns.loc[returns["ticker"].eq(ticker)].copy(); stock.index = stock["date"].dt.strftime("%Y-%m-%d")
        raw_y = stock["open_close_return"].reindex(dates).to_numpy(float)
        ar_y = stock["market_adjusted_open_close_return"].reindex(dates).to_numpy(float)
        spy = stock["spy_open_close_return"].reindex(dates).to_numpy(float)
        sector = etf_returns[sector_etf].reindex(dates).to_numpy(float) if sector_etf in etf_returns.columns else np.full(len(dates), np.nan)
        for index_type in MAIN_TYPES:
            signal_id = f"T{topic_id}_{index_type}"; x = signal_wide[signal_id].to_numpy(float)
            market_model = hac_ols(raw_y, np.column_stack([x, spy]), int(config["regression"]["hac_maxlags"]))
            sector_model = hac_ols(raw_y, np.column_stack([x, spy, sector]), int(config["regression"]["hac_maxlags"]))
            univ = hac_ols(ar_y, x, int(config["regression"]["hac_maxlags"]))
            minimum = int(config["regression"]["minimum_observations"])
            for candidate in (market_model, sector_model, univ):
                if candidate["n"] < minimum:
                    candidate["status"] = "insufficient"
                    candidate["reason"] = f"{candidate['n']} valid pairs < fixed minimum {minimum}"
            valid = np.isfinite(ar_y) & np.isfinite(x)
            influence_removed = np.zeros(len(dates), dtype=bool)
            if univ["status"] == "valid":
                xv = x[valid]; yv = ar_y[valid]; alpha, beta = univ["beta"][:2]
                resid = yv - alpha - beta * xv; sxx = float(((xv - xv.mean()) ** 2).sum())
                leverage = 1 / len(xv) + ((xv - xv.mean()) ** 2 / sxx if sxx > 0 else 0)
                mse = float((resid ** 2).sum() / max(1, len(xv) - 2))
                cook = (resid ** 2 / max(mse * 2, 1e-20)) * leverage / np.maximum((1 - leverage) ** 2, 1e-20)
                remove_valid = (cook > 4 / len(xv)) | (np.abs(yv) > float(config["prices"]["outlier_abs_return_threshold"]))
                influence_removed[np.flatnonzero(valid)[remove_valid]] = True
            sensitivity_y = ar_y.copy(); sensitivity_y[influence_removed] = np.nan
            sensitivity = hac_ols(sensitivity_y, x, int(config["regression"]["hac_maxlags"]))
            if sensitivity["n"] < minimum:
                sensitivity["status"] = "insufficient"
                sensitivity["reason"] = f"{sensitivity['n']} valid pairs < fixed minimum {minimum}"
            def extract(result: dict[str, Any], signal_position: int = 1) -> tuple[Any, Any, Any, Any]:
                if result["status"] != "valid": return (np.nan, np.nan, np.nan, result["n"])
                return (float(result["beta"][signal_position]), float(result["se"][signal_position]), float(result["p"][signal_position]), result["n"])
            ub, use, up, un = extract(univ); mb, mse_, mp, mn = extract(market_model); sb, sse, sp, sn = extract(sector_model); ib, ise, ip, inn = extract(sensitivity)
            rows.append({"ticker": ticker, "topic_id": topic_id, "index_type": index_type, "sector_etf": sector_etf,
                         "main_ar_beta": ub, "main_ar_se": use, "main_ar_p": up, "main_ar_N": un,
                         "market_control_beta": mb, "market_control_se": mse_, "market_control_p": mp, "market_control_N": mn,
                         "sector_control_beta": sb, "sector_control_se": sse, "sector_control_p": sp, "sector_control_N": sn,
                         "influence_removed_count": int(influence_removed.sum()), "influence_sensitivity_beta": ib,
                         "influence_sensitivity_se": ise, "influence_sensitivity_p": ip, "influence_sensitivity_N": inn,
                         "direction_stable_market": bool(np.isfinite(ub) and np.isfinite(mb) and np.sign(ub) == np.sign(mb)),
                         "direction_stable_sector": bool(np.isfinite(ub) and np.isfinite(sb) and np.sign(ub) == np.sign(sb))})
    frame = pd.DataFrame(rows)
    write_csv(out_dir / "predetermined_robustness_results.csv", frame)
    return frame


def color_for_value(value: float, scale: float) -> tuple[int, int, int]:
    if not np.isfinite(value): return (235, 235, 235)
    z = max(-1.0, min(1.0, value / max(scale, 1e-12)))
    if z >= 0:
        return (255, int(245 - 120 * z), int(235 - 170 * z))
    return (int(235 - 170 * (-z)), int(245 - 110 * (-z)), 255)


def draw_heatmap(results: pd.DataFrame, path: Path) -> None:
    valid = results.loc[results["status"].eq("valid")].copy()
    if valid.empty: return
    valid["signal_id"] = valid.apply(lambda r: f"T{int(r.topic_id)}\n{TYPE_LABELS.get(r.index_type, r.index_type)}", axis=1)
    ranked = valid.groupby("ticker")["q_value"].min().sort_values().head(60).index.tolist()
    signals = [f"T{k}\n{TYPE_LABELS[t]}" for k in range(6) for t in MAIN_TYPES]
    pivot = valid.pivot(index="ticker", columns="signal_id", values="standardized_effect_bps").reindex(index=ranked, columns=signals)
    q_pivot = valid.pivot(index="ticker", columns="signal_id", values="q_value").reindex(index=ranked, columns=signals)
    finite = np.abs(pivot.to_numpy(float)); scale = float(np.nanpercentile(finite, 95)) if np.isfinite(finite).any() else 1.0
    cell_w, cell_h, left, top = 78, 22, 110, 175
    width, height = left + cell_w * len(signals) + 40, top + cell_h * len(ranked) + 90
    im = Image.new("RGB", (width, height), "white"); draw = ImageDraw.Draw(im)
    title = load_font(24, True); small = load_font(11); tiny = load_font(9)
    draw.text((25, 20), "Standardized effects: pre-open Guardian signals vs market-adjusted intraday returns", font=title, fill="#111")
    draw.text((25, 55), "Top 60 tickers by minimum global q-value; black outline = q < 0.05. Cell values are bps per 1 SD signal increase.", font=small, fill="#444")
    for j, signal in enumerate(signals):
        x = left + j * cell_w
        parts = signal.split("\n")
        draw.text((x + 4, 95), parts[0], font=small, fill="#111")
        label = parts[1].replace(" stance", "").replace(" sentiment", "")
        draw.text((x + 4, 115), label[:11], font=tiny, fill="#333")
    for i, ticker in enumerate(ranked):
        y = top + i * cell_h; draw.text((25, y + 3), ticker, font=small, fill="#222")
        for j, signal in enumerate(signals):
            x = left + j * cell_w; value = pivot.loc[ticker, signal]; q = q_pivot.loc[ticker, signal]
            draw.rectangle((x, y, x + cell_w - 2, y + cell_h - 2), fill=color_for_value(value, scale),
                           outline="#000" if pd.notna(q) and q < 0.05 else "#ddd", width=3 if pd.notna(q) and q < 0.05 else 1)
    draw.text((25, height - 45), f"Color clipped at +/-{scale:.1f} bps. Full 1,000-stock results are in main_univariate_tests.csv.", font=small, fill="#444")
    path.parent.mkdir(parents=True, exist_ok=True); im.save(path)


def draw_coefficient_summary(results: pd.DataFrame, path: Path) -> None:
    valid = results.loc[results["status"].eq("valid") & results["standardized_effect_bps"].notna()].copy()
    significant = valid.loc[valid["q_value"].lt(0.05)].sort_values("q_value")
    width, height = 1500, 900
    im = Image.new("RGB", (width, height), "white"); draw = ImageDraw.Draw(im)
    title = load_font(26, True); font = load_font(15); small = load_font(12)
    if significant.empty:
        draw.text((35, 25), "No globally FDR-significant stock-signal coefficients", font=title, fill="#111")
        draw.text((35, 67), "Distribution of all valid standardized effects; this is not a chart of significant stocks.", font=font, fill="#444")
        values = valid["standardized_effect_bps"].clip(valid["standardized_effect_bps"].quantile(.01), valid["standardized_effect_bps"].quantile(.99)).to_numpy(float)
        counts, edges = np.histogram(values, bins=50); left, top, right, bottom = 80, 140, 980, 770
        maxc = max(counts.max(), 1)
        for i, count in enumerate(counts):
            x0 = left + i * (right-left)/len(counts); x1 = left + (i+1)*(right-left)/len(counts)
            y0 = bottom - count/maxc*(bottom-top)
            draw.rectangle((x0, y0, x1-1, bottom), fill="#4C78A8")
        draw.line((left, bottom, right, bottom), fill="#222", width=2); draw.line((left, top, left, bottom), fill="#222", width=2)
        draw.text((left, bottom+15), f"{edges[0]:.0f} bps", font=small, fill="#333"); draw.text((right-80, bottom+15), f"{edges[-1]:.0f} bps", font=small, fill="#333")
        draw.text((1040, 135), "Smallest q-values (exploratory ranking)", font=font, fill="#111")
        for i, (_, row) in enumerate(valid.sort_values("q_value").head(18).iterrows()):
            draw.text((1040, 175+i*34), f"{row.ticker} T{int(row.topic_id)} {TYPE_LABELS[row.index_type]}: q={row.q_value:.3g}, {row.standardized_effect_bps:.1f} bps", font=small, fill="#333")
    else:
        show = significant.head(30).copy(); draw.text((35, 25), "Globally FDR-significant standardized coefficients", font=title, fill="#111")
        maxabs = float(np.nanmax(np.abs(show[["standardized_ci_low_bps", "standardized_ci_high_bps"]].to_numpy())))
        center, left, right, top, row_h = 750, 350, 1420, 100, 24
        for i, (_, row) in enumerate(show.iterrows()):
            y = top + i*row_h; scale = (right-left)/(2*maxabs); x0 = center + row.standardized_ci_low_bps*scale; x1 = center + row.standardized_ci_high_bps*scale; xb = center + row.standardized_effect_bps*scale
            draw.text((25, y-6), f"{row.ticker} T{int(row.topic_id)} {TYPE_LABELS[row.index_type]}", font=small, fill="#222")
            draw.line((x0,y,x1,y), fill="#333", width=2); draw.ellipse((xb-4,y-4,xb+4,y+4), fill="#B24C3D")
        draw.line((center, top-15, center, top+len(show)*row_h), fill="#888", width=1)
    path.parent.mkdir(parents=True, exist_ok=True); im.save(path)


def draw_scatter(returns: pd.DataFrame, signals: pd.DataFrame, main_results: pd.DataFrame, path: Path) -> None:
    preferred = [("XOM", 3, "topic_sentiment"), ("WMT", 4, "topic_sentiment"), ("JNJ", 2, "topic_sentiment")]
    selected = None
    for ticker, topic, index_type in preferred:
        row = main_results.loc[(main_results.ticker == ticker) & (main_results.topic_id == topic) & (main_results.index_type == index_type) & (main_results.status == "valid")]
        if len(row): selected = row.iloc[0]; break
    if selected is None:
        valid = main_results.loc[main_results.status.eq("valid")].sort_values(["q_value", "ticker"])
        if valid.empty: return
        selected = valid.iloc[0]
    ticker, topic, index_type = selected.ticker, int(selected.topic_id), selected.index_type
    x = signals.loc[(signals.topic_id == topic) & (signals.index_type == index_type)].set_index("target_trading_date")["signal_value"]
    y = returns.loc[returns.ticker.eq(ticker)].copy(); y.index = y.date.dt.strftime("%Y-%m-%d")
    d = pd.concat([x.rename("x"), y["market_adjusted_open_close_return"].rename("y")], axis=1).dropna()
    if d.empty: return
    width, height = 1200, 850; im = Image.new("RGB", (width,height),"white"); draw=ImageDraw.Draw(im)
    title=load_font(24,True); font=load_font(14); left,top,right,bottom=100,120,1120,740
    draw.text((35,25), f"Illustrative pre-specified business mapping: {ticker}, T{topic} {TYPE_LABELS[index_type]}",font=title,fill="#111")
    draw.text((35,65),"Pre-open signal vs same target-day open-to-close return minus SPY; association is not causal.",font=font,fill="#444")
    xmin,xmax=float(d.x.min()),float(d.x.max()); ymin,ymax=float(d.y.min()),float(d.y.max())
    if xmin==xmax: xmax=xmin+1
    if ymin==ymax: ymax=ymin+1
    for _,r in d.iterrows():
        px=left+(r.x-xmin)/(xmax-xmin)*(right-left); py=bottom-(r.y-ymin)/(ymax-ymin)*(bottom-top)
        draw.ellipse((px-4,py-4,px+4,py+4),fill="#2F6B9A")
    xs=np.array([xmin,xmax]); ys=float(selected.alpha)+float(selected.beta)*xs
    pts=[(left+(v-xmin)/(xmax-xmin)*(right-left),bottom-(w-ymin)/(ymax-ymin)*(bottom-top)) for v,w in zip(xs,ys)]
    draw.line(pts,fill="#B24C3D",width=3); draw.rectangle((left,top,right,bottom),outline="#333",width=2)
    draw.text((left,bottom+25),f"Signal range {xmin:.2f} to {xmax:.2f}",font=font,fill="#333")
    draw.text((left+500,bottom+25),f"N={len(d)}, beta={selected.beta:.4g}, global q={selected.q_value:.3g}",font=font,fill="#333")
    path.parent.mkdir(parents=True,exist_ok=True); im.save(path)


def draw_basket_performance(baskets: pd.DataFrame, path: Path) -> None:
    width,height=1500,900; im=Image.new("RGB",(width,height),"white"); draw=ImageDraw.Draw(im)
    title=load_font(25,True); font=load_font(13); draw.text((35,25),"Fixed policy-topic baskets: cumulative market-adjusted intraday performance",font=title,fill="#111")
    draw.text((35,65),"Equal-weight mapped stocks; daily basket open-to-close return minus SPY. Components were fixed before association tests.",font=font,fill="#444")
    left,top,right,bottom=100,120,1430,780; colors=["#2F6B9A","#D97706","#2E8B57","#B24C3D","#7A5195","#555555"]
    ymin=float(baskets.cumulative_market_adjusted_return.min()); ymax=float(baskets.cumulative_market_adjusted_return.max()); pad=max((ymax-ymin)*.1,.005); ymin-=pad;ymax+=pad
    dates=sorted(baskets.date.unique()); xpos={d:left+i*(right-left)/max(1,len(dates)-1) for i,d in enumerate(dates)}
    for topic_id,g in baskets.groupby("topic_id"):
        pts=[]
        for _,r in g.sort_values("date").iterrows():
            y=bottom-(r.cumulative_market_adjusted_return-ymin)/(ymax-ymin)*(bottom-top); pts.append((xpos[r.date],y))
        if len(pts)>1: draw.line(pts,fill=colors[int(topic_id)],width=3)
        ly=120+int(topic_id)*28; draw.line((1120,ly,1160,ly),fill=colors[int(topic_id)],width=4); draw.text((1170,ly-8),f"T{int(topic_id)}",font=font,fill="#222")
    zero=bottom-(0-ymin)/(ymax-ymin)*(bottom-top); draw.line((left,zero,right,zero),fill="#aaa",width=1); draw.rectangle((left,top,right,bottom),outline="#333",width=2)
    draw.text((left,bottom+20),str(pd.Timestamp(dates[0]).date()),font=font,fill="#333"); draw.text((right-100,bottom+20),str(pd.Timestamp(dates[-1]).date()),font=font,fill="#333")
    path.parent.mkdir(parents=True,exist_ok=True); im.save(path)


def draw_oos_bars(metrics: pd.DataFrame, path: Path) -> None:
    show=metrics.loc[metrics.prediction_count.ge(1)].copy().sort_values(["topic_id","index_type"])
    if show.empty: return
    width,height=1500,900; im=Image.new("RGB",(width,height),"white"); draw=ImageDraw.Draw(im); title=load_font(25,True); font=load_font(12)
    draw.text((35,25),"Fixed topic baskets: out-of-sample RMSE vs two benchmarks",font=title,fill="#111")
    draw.text((35,65),"Expanding window, at least 30 training pairs. Fewer than 15 predictions remains insufficient evidence.",font=font,fill="#444")
    show=show.head(24); maxv=float(show[["rmse_model","rmse_zero","rmse_historical_mean"]].max().max()); left,top,right,bottom=280,120,1430,800; row_h=(bottom-top)/len(show)
    colors={"model":"#2F6B9A","zero":"#999999","mean":"#D97706"}
    for i,(_,r) in enumerate(show.iterrows()):
        y=top+i*row_h; draw.text((25,y+3),f"T{int(r.topic_id)} {TYPE_LABELS[r.index_type]} (N={int(r.prediction_count)})",font=font,fill="#222")
        for k,(col,label) in enumerate([("rmse_model","model"),("rmse_zero","zero"),("rmse_historical_mean","mean")]):
            value=r.get(col,np.nan)
            if pd.notna(value): draw.rectangle((left,y+3+k*6,left+(right-left)*value/maxv,y+7+k*6),fill=colors[label])
    draw.text((left,bottom+25),"Blue=model  Gray=zero-return benchmark  Orange=training-mean benchmark",font=font,fill="#333")
    path.parent.mkdir(parents=True,exist_ok=True); im.save(path)


def create_report(out_dir: Path, config: dict[str, Any], main: pd.DataFrame, robust: pd.DataFrame,
                  oos_metrics: pd.DataFrame, basket_tests: pd.DataFrame, basket_oos: pd.DataFrame,
                  testability: pd.DataFrame, signals: pd.DataFrame, validation: dict[str, Any]) -> None:
    valid=main.loc[main.status.eq("valid")]; sig=valid.loc[valid.q_value.lt(.05)].sort_values("q_value")
    exploratory=valid.loc[valid.q_value.ge(.05)&valid.q_value.lt(.10)].sort_values("q_value")
    nominal=valid.loc[valid.raw_p_value.lt(.05)&valid.q_value.ge(.10)].sort_values("raw_p_value")
    bootstrap_supported=sig.loc[sig.get("bootstrap_ci_excludes_zero",False).eq(True)] if len(sig) else sig
    evaluable_oos=oos_metrics.loc[oos_metrics.evidence_status.eq("evaluable")]
    beats=evaluable_oos.loc[evaluable_oos.beats_zero_rmse.eq(True)&evaluable_oos.beats_historical_mean_rmse.eq(True)]
    basket_valid=basket_tests.loc[basket_tests.status.eq("valid")]; basket_sig=basket_valid.loc[basket_valid.q_value.lt(.05)]
    signal_coverage=signals.loc[signals.index_type.isin(MAIN_TYPES)].groupby(["topic_id","index_type"])["signal_value"].apply(lambda s:int(s.notna().sum())).reset_index(name="valid_trading_day_windows")
    top_rows=(sig if len(sig) else valid.sort_values("q_value").head(15))
    table="\n".join(f"| {r.ticker} | T{int(r.topic_id)} | {TYPE_LABELS[r.index_type]} | {int(r.N)} | {r.standardized_effect_bps:.2f} | {r.raw_p_value:.4g} | {r.q_value:.4g} |" for _,r in top_rows.iterrows()) or "| — | — | — | — | — | — | — |"
    conclusion = "本窗口未发现经多重检验校正后显著的关联。" if sig.empty else f"共有 {len(sig)} 个股票—指数关联在全局 BH-FDR q<0.05 后保留。"
    coverage_md="| Topic | Index | Valid pre-open trading-day windows |\n|---:|---|---:|\n"+"\n".join(
        f"| T{int(r.topic_id)} | {TYPE_LABELS[r.index_type]} | {int(r.valid_trading_day_windows)} |" for _,r in signal_coverage.iterrows())
    text=f"""# Guardian 政策情绪与美国股票市场调整后收益：探索性分析报告

生成时间：{utc_now()}

## 设计与覆盖

- 研究窗口：{config['project']['start_date']} 至 {config['project']['end_date_inclusive']}；使用目标交易日开盘前固定新闻窗口。
- 股票池在读取任何收益关联前冻结为 {len(testability)} 只；成功下载 {int((testability.status!='not_downloaded').sum())} 只，其中可检验 {int((testability.status=='testable').sum())} 只；下载失败 {int((testability.status=='not_downloaded').sum())} 只，另有 {int((testability.status=='insufficient').sum())} 只行情样本不足。
- 主指数为 6 个主题 × 3 类实际指数（议题情绪、民主党态度、共和党态度）= 18 条信号；关注度单列补充，不当作情绪。
- 主检验有效 {len(valid)} 个，全部有效原始 p 值在同一个股票×指数家族中统一做 BH-FDR。HAC 固定 maxlags={config['regression']['hac_maxlags']}；区块 bootstrap 长度 {config['regression']['block_bootstrap_length']} 个完整交易日、{config['regression']['block_bootstrap_repetitions']} 次。
- 收益为复权开盘至收盘股票收益减去相同区间 SPY 收益。直接相减隐含 beta=1，预设映射股票另做估计市场 beta 与行业 ETF 控制。

## 主结果

{conclusion}

| Ticker | Topic | Index | N | Effect (bps / 1 SD) | raw p | global q |
|---|---:|---|---:|---:|---:|---:|
{table}

- q<0.05：{len(sig)}；0.05≤q<0.10：{len(exploratory)}；原始 p<0.05 但 q≥0.10：{len(nominal)}。
- q<0.05 的 {len(sig)} 个结果中，{len(bootstrap_supported)} 个的 5 日移动区块 bootstrap 95% 区间也排除 0；其余结果对连续交易日共同冲击和稀疏信号的处理不稳，不能称为稳健发现。
- 效应量是情绪指数增加一个样本标准差对应的市场调整后开盘—收盘收益基点数；保留原始方向，不把统计显著等同于经济或因果重要性。
- 广泛显著结果主要集中在 T2 与 T4 的民主党态度。现有标注中民主党态度几乎没有 `oppose`，许多窗口是 0，少数窗口为正；因此这些回归更接近“少数出现正面民主党表达的日期”与其他日期的比较，容易同时反映共同市场日冲击。

## 信号覆盖

{coverage_md}

## 稳健性、联合模型与异常值

- 预设业务映射共 {len(robust)} 个股票—主题—指数稳健性模型。方向在市场控制后保持的可估计记录数：{int(robust.direction_stable_market.sum())}；加入预设行业 ETF 后保持：{int(robust.direction_stable_sector.sum())}。
- `predetermined_robustness_results.csv` 同时保留完整样本、市场控制、行业控制以及 Cook 距离/极端收益敏感性，不因删除某日后显著而替换主结果。
- 联合模型来自分析前保存的多主题业务映射；样本交集不足时按预设门槛跳过，不强行估计。

## 样本外预测

- 全部预设单指数模型均保留；至少 15 次预测的可评估模型 {len(evaluable_oos)} 个，其中 RMSE 同时低于零收益与历史均值基准 {len(beats)} 个。
- 负的样本外 R² 原样保留。这里是预测检验，不是可成交交易收益；未构造手续费、点差、滑点或做空假设下的盈利曲线。
- 若只展示全样本 q<0.05 后再挑出的模型，会产生事后选择；相关明细虽保留，但不当作独立样本外确认。

## 固定主题篮子

- 六个篮子均由分析前业务映射固定并等权汇总，逐日记录有效成分数；没有使用事后显著股票重新组篮子。
- 篮子有效检验 {len(basket_valid)} 个，单独 BH-FDR 后 q<0.05 为 {len(basket_sig)} 个。篮子样本外可评估模型 {int((basket_oos.evidence_status=='evaluable').sum())} 个。

## 解释限制

1. 仅约三个月、约六十多个交易日；稀疏的 T0/T1 和党派态度信号使很多模型不足 30 个配对。
2. 指数来自 Guardian 政策报道中呈现的评价与党派态度，不代表全体选民或客观政策状态；媒体选题和标注分布均可能带来偏差。
3. IWB、S&P 500 和 Nasdaq-100 名单来自 2026-10 附近的可公开快照而非 7 月 10 日历史成分，存在生存者偏差；IWB 是 Russell 1000 的透明代理而非授权指数成分文件。
4. 大范围 1,000 股票筛选是探索性的；即使 FDR 校正也不能消除所有数据挖掘、行业共同冲击和测量误差。
5. HAC 与区块 bootstrap 在短样本和不规则缺失下仍是近似推断；相关、条件相关或样本外预测表现均不构成因果证据。
6. 既有 Democratic Sweep 不显著结果保留在 `data/sweep_analysis/`，本模块没有修改其方法或输出。

验证状态：`{validation['validation_status']}`。详细检查见 `equity_sentiment_validation_report.json`。
"""
    (out_dir/"equity_sentiment_analysis_report.md").write_text(text,encoding="utf-8")


def validate_outputs(config: dict[str, Any], out_dir: Path, returns: pd.DataFrame, preopen: pd.DataFrame,
                     main: pd.DataFrame, oos_details: pd.DataFrame, testable: list[str], mapping: pd.DataFrame) -> dict[str, Any]:
    start=pd.Timestamp(config["project"]["start_date"]); end=pd.Timestamp(config["project"]["end_date_inclusive"])
    price_unique=not returns.duplicated(["ticker","date"]).any()
    signal_end=pd.to_datetime(preopen["window_end_new_york_exclusive"],utc=True); latest=pd.to_datetime(preopen["latest_any_article_timestamp_new_york"],utc=True,errors="coerce")
    no_late=bool(((latest.isna())|(latest<signal_end)).all())
    valid_main=main.loc[main.status.eq("valid")]
    q_complete=int(valid_main.q_value.notna().sum())==len(valid_main)
    formula = returns["adjusted_close"] / returns["adjusted_open"] - 1
    formula_error = float((formula - returns["open_close_return"]).abs().max())
    ar_error = float((returns["open_close_return"] - returns["spy_open_close_return"] - returns["market_adjusted_open_close_return"]).abs().max())
    return_wide = returns.pivot(index="date", columns="ticker", values="adjusted_close").sort_index()
    prior_missing = return_wide.shift(1).isna()
    ctc_wide = returns.pivot(index="date", columns="ticker", values="close_to_close_return").reindex_like(return_wide)
    cross_missing_violations = int((prior_missing.iloc[1:] & ctc_wide.iloc[1:].notna()).sum().sum())
    end_local = pd.to_datetime(preopen["window_end_new_york_exclusive"], utc=True).dt.tz_convert(config["project"]["timezone"])
    open_local = pd.to_datetime(preopen["target_trading_date"] + " 09:30:00").dt.tz_localize(config["project"]["timezone"])
    oos_causal=True
    if not oos_details.empty:
        oos_causal=bool((pd.to_datetime(oos_details.training_end_date)<pd.to_datetime(oos_details.target_date)).all())
    mapped=set(mapping.ticker); frozen=set(pd.read_csv(out_dir/"frozen_stock_universe.csv",encoding="utf-8-sig").ticker)
    report={"validation_status":"pass","checks":{
        "ticker_date_unique":price_unique,"returns_within_requested_dates":bool(returns.date.between(start,end).all()),
        "positive_adjustment_factors":bool(returns.loc[returns.adjustment_factor.notna(),"adjustment_factor"].gt(0).all()),
        "adjusted_open_close_formula_exact":formula_error<1e-12,"stock_minus_spy_interval_exact":ar_error<1e-12,
        "no_close_to_close_return_across_missing_prior_stock_day":cross_missing_violations==0,
        "window_end_precedes_target_open":bool((end_local<open_local).all()),"no_news_at_or_after_window_end":no_late,
        "missing_signals_remain_nan":bool(preopen.loc[preopen.missing_reason.ne(""),"signal_value"].isna().all()),
        "full_global_fdr_coverage":q_complete,"oos_training_targets_precede_prediction":oos_causal,
        "all_mapped_tickers_in_frozen_universe":mapped.issubset(frozen),"all_testable_tickers_have_returns":set(testable).issubset(set(returns.ticker))},
        "counts":{"return_rows":len(returns),"testable_stocks":len(testable),"preopen_signal_rows":len(preopen),
                  "main_valid_tests":len(valid_main),"main_q_values":int(valid_main.q_value.notna().sum()),"oos_prediction_rows":len(oos_details)},
        "diagnostics":{"max_open_close_formula_abs_error":formula_error,"max_stock_minus_spy_abs_error":ar_error,
                       "cross_missing_day_return_violations":cross_missing_violations},"manual_recalculation":{}}
    # Independently recalculate the first valid univariate slope from raw aligned arrays.
    if len(valid_main):
        r=valid_main.iloc[0]; x=preopen.loc[(preopen.topic_id==r.topic_id)&(preopen.index_type==r.index_type)].set_index("target_trading_date").signal_value
        y=returns.loc[returns.ticker.eq(r.ticker)].copy(); y.index=y.date.dt.strftime("%Y-%m-%d")
        d=pd.concat([x,y.market_adjusted_open_close_return],axis=1).dropna(); xv=d.iloc[:,0].to_numpy(float); yv=d.iloc[:,1].to_numpy(float)
        manual=float(np.sum((xv-xv.mean())*(yv-yv.mean()))/np.sum((xv-xv.mean())**2))
        report["manual_recalculation"]={"ticker":r.ticker,"topic_id":int(r.topic_id),"index_type":r.index_type,"N":len(d),
            "independent_beta":manual,"reported_beta":float(r.beta),"absolute_difference":abs(manual-float(r.beta)),"pass":abs(manual-float(r.beta))<1e-10}
    if not all(report["checks"].values()) or (report["manual_recalculation"] and not report["manual_recalculation"]["pass"]):
        report["validation_status"]="fail"
    write_json(out_dir/"equity_sentiment_validation_report.json",report)
    return report


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="equity_sentiment_config.yaml"); args=parser.parse_args()
    config=yaml.safe_load((ROOT/args.config).read_text(encoding="utf-8")); out_dir=ROOT/config["project"]["output_dir"]; fig_dir=ROOT/config["project"]["figure_dir"]
    out_dir.mkdir(parents=True,exist_ok=True); fig_dir.mkdir(parents=True,exist_ok=True)
    topics=load_topics(ROOT/config["project"]["topic_definitions"])
    returns,testability,testable,study_dates=prepare_returns(config,out_dir)
    raw_prices=pd.read_csv(out_dir/"raw_daily_prices.csv",encoding="utf-8-sig"); all_price_dates=sorted(pd.to_datetime(raw_prices.loc[raw_prices.ticker.eq(config["prices"]["benchmark"]),"date"]).unique())
    preopen,same_day=build_news_signals(config,out_dir,topics,study_dates,[pd.Timestamp(x) for x in all_price_dates])
    build_alignment(returns,testable,preopen,out_dir)

    main_signals=preopen.loc[preopen.index_type.isin(MAIN_TYPES)]
    main_results=run_univariate_family(returns,testable,main_signals,config,"main_preopen_sentiment_family","market_adjusted_open_close_return",True)
    write_csv(out_dir/"main_univariate_tests.csv",main_results)
    write_csv(out_dir/"fdr_significant_results.csv",main_results.loc[main_results.result_category.eq("fdr_significant_q_lt_0.05")])
    write_csv(out_dir/"fdr_significant_bootstrap_supported.csv",main_results.loc[main_results.result_category.eq("fdr_significant_q_lt_0.05") & main_results.get("bootstrap_ci_excludes_zero", False).eq(True)])
    write_csv(out_dir/"exploratory_q_0.05_to_0.10.csv",main_results.loc[main_results.result_category.eq("exploratory_q_0.05_to_0.10")])
    write_csv(out_dir/"nominal_p_lt_0.05_only.csv",main_results.loc[main_results.result_category.eq("nominal_only")])

    attention_results=run_univariate_family(returns,testable,preopen.loc[preopen.index_type.eq("attention")],config,"supplementary_attention_family","market_adjusted_open_close_return",False)
    write_csv(out_dir/"supplementary_attention_tests.csv",attention_results)
    same_results=run_univariate_family(returns,testable,same_day.loc[same_day.index_type.isin(MAIN_TYPES)],config,"same_day_descriptive_family","market_adjusted_close_to_close_return",False)
    write_csv(out_dir/"same_day_descriptive_tests.csv",same_results)

    mapping=pd.read_csv(out_dir/"stock_topic_mapping.csv",encoding="utf-8-sig"); mapping["topic_id"]=mapping.topic_id.astype(int)
    joint=run_joint_models(returns,main_signals,config,out_dir)
    robust=robustness_models(returns,main_signals,mapping,config,out_dir)
    oos_details,oos_metrics=walk_forward(returns,testable,main_signals,config,"market_adjusted_open_close_return","main_preopen_sentiment_family")
    write_csv(out_dir/"walk_forward_predictions.csv.gz",oos_details,compressed=True); write_csv(out_dir/"walk_forward_metrics.csv",oos_metrics)

    baskets,basket_tickers=prepare_baskets(returns,mapping,topics,config,out_dir)
    basket_returns=baskets.rename(columns={"date":"date"}).copy()
    basket_signals=main_signals.copy()
    # Each basket is tested only against the three indices for its own pre-specified topic.
    basket_test_frames=[]; basket_oos_frames=[]; basket_pred_frames=[]
    for topic_id in range(6):
        bt=f"BASKET_T{topic_id}"; br=basket_returns.loc[basket_returns.ticker.eq(bt)].copy()
        bs=basket_signals.loc[basket_signals.topic_id.eq(topic_id)]
        basket_test_frames.append(run_univariate_family(br,[bt],bs,config,"fixed_topic_basket_family","market_adjusted_open_close_return",True))
        pred,met=walk_forward(br,[bt],bs,config,"market_adjusted_open_close_return","fixed_topic_basket_family")
        basket_pred_frames.append(pred); basket_oos_frames.append(met)
    basket_tests=pd.concat(basket_test_frames,ignore_index=True)
    # Reapply BH once across all 18 basket tests, not separately by basket.
    basket_tests=apply_fdr(basket_tests.drop(columns=["q_value","result_category"],errors="ignore"))
    basket_predictions=pd.concat(basket_pred_frames,ignore_index=True); basket_oos=pd.concat(basket_oos_frames,ignore_index=True)
    write_csv(out_dir/"topic_basket_univariate_tests.csv",basket_tests); write_csv(out_dir/"topic_basket_walk_forward_predictions.csv",basket_predictions); write_csv(out_dir/"topic_basket_walk_forward_metrics.csv",basket_oos)

    validation=validate_outputs(config,out_dir,returns,preopen,main_results,oos_details,testable,mapping)
    draw_heatmap(main_results,fig_dir/"standardized_coefficient_heatmap.png")
    draw_coefficient_summary(main_results,fig_dir/"fdr_coefficient_summary.png")
    draw_scatter(returns,preopen,main_results,fig_dir/"representative_scatter.png")
    draw_basket_performance(baskets,fig_dir/"fixed_basket_cumulative_market_adjusted_performance.png")
    draw_oos_bars(basket_oos,fig_dir/"oos_model_vs_benchmarks.png")
    create_report(out_dir,config,main_results,robust,oos_metrics,basket_tests,basket_oos,testability,preopen,validation)
    valid_main=main_results.loc[main_results.status.eq("valid")]; significant=valid_main.loc[valid_main.q_value.lt(.05)]
    evaluable_oos=oos_metrics.loc[oos_metrics.evidence_status.eq("evaluable")]
    write_json(out_dir/"analysis_summary.json",{
        "created_at_utc":utc_now(),"frozen_stocks":len(testability),"downloaded_stocks":int(testability.status.ne("not_downloaded").sum()),
        "testable_stocks":len(testable),"main_index_count":18,"valid_main_tests":len(valid_main),"insufficient_main_tests":int(main_results.status.eq("insufficient").sum()),
        "fdr_significant_q_lt_0_05":len(significant),"fdr_significant_and_block_bootstrap_ci_excludes_zero":int(significant.get("bootstrap_ci_excludes_zero",False).eq(True).sum()),
        "exploratory_q_0_05_to_0_10":int(main_results.result_category.eq("exploratory_q_0.05_to_0.10").sum()),
        "nominal_only":int(main_results.result_category.eq("nominal_only").sum()),
        "oos_evaluable_models":len(evaluable_oos),"oos_beats_both_rmse_benchmarks":int((evaluable_oos.beats_zero_rmse.eq(True)&evaluable_oos.beats_historical_mean_rmse.eq(True)).sum()),
        "basket_fdr_significant":int(basket_tests.q_value.lt(.05).sum()),"validation_status":validation["validation_status"]})
    write_json(out_dir/"reproducibility_manifest.json",{"created_at_utc":utc_now(),"config":args.config,"script":Path(__file__).name,
        "inputs":[config["project"]["article_scores"],config["project"]["topic_definitions"],str((out_dir/"frozen_stock_universe.csv").relative_to(ROOT)),str((out_dir/"raw_daily_prices.csv").relative_to(ROOT))],
        "random_seed":config["regression"]["random_seed"],"validation_status":validation["validation_status"]})
    print(json.dumps({"testable_stocks":len(testable),"valid_main_tests":int(main_results.status.eq('valid').sum()),
        "fdr_significant":int(main_results.q_value.lt(.05).sum()),"oos_predictions":len(oos_details),"validation":validation["validation_status"]},ensure_ascii=False))
    return 0 if validation["validation_status"]=="pass" else 2


if __name__=="__main__":
    raise SystemExit(main())
