#!/usr/bin/env python
"""Freeze a broad US equity universe and download cached daily prices with yfinance.

The universe is fixed from public constituent/holdings sources before any return or
sentiment association is calculated.  No research-window performance is used for
selection.  Yahoo tickers are normalized only for downloading; source tickers are
retained in the frozen-universe file.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml


ROOT = Path(__file__).resolve().parent
CSV_ENCODING = "utf-8-sig"
USER_AGENT = "Assignment5 academic research/1.0"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding=CSV_ENCODING)


def yahoo_ticker(value: str) -> str:
    normalized = str(value).strip().upper().replace(".", "-").replace("/", "-")
    return re.sub(r"\s+", "-", normalized)


def get_cached(url: str, path: Path, refresh: bool, timeout: int = 60) -> tuple[bytes, bool, str]:
    if path.exists() and not refresh:
        return path.read_bytes(), True, utc_now()
    response = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return response.content, False, utc_now()


def parse_sp500(content: bytes) -> pd.DataFrame:
    tables = pd.read_html(io.BytesIO(content))
    table = next(t for t in tables if "Symbol" in t.columns and "Security" in t.columns)
    out = pd.DataFrame({
        "source_ticker": table["Symbol"].astype(str),
        "company": table["Security"].astype(str),
        "sector": table.get("GICS Sector", "").astype(str),
    })
    return out


def parse_nasdaq100(content: bytes) -> tuple[pd.DataFrame, str | None]:
    payload = json.loads(content.decode("utf-8"))
    outer = payload.get("data") or {}
    rows = ((outer.get("data") or {}).get("rows") or [])
    table = pd.DataFrame(rows)
    if table.empty or not {"symbol", "companyName"}.issubset(table.columns):
        raise ValueError("Could not identify Nasdaq-100 rows in official JSON response")
    sector = table["sector"].astype(str) if "sector" in table.columns else pd.Series([""] * len(table))
    return pd.DataFrame({"source_ticker": table["symbol"].astype(str), "company": table["companyName"].astype(str),
                         "sector": sector}), outer.get("date")


def parse_iwb(content: bytes) -> tuple[pd.DataFrame, str | None]:
    text = content.decode("utf-8-sig", errors="replace")
    lines = text.splitlines()
    header_index = next(i for i, line in enumerate(lines) if line.startswith("Ticker,Name,Sector,Asset Class"))
    as_of_match = re.search(r'Fund Holdings as of,"([^"]+)"', text[:500])
    as_of = as_of_match.group(1) if as_of_match else None
    table = pd.read_csv(io.StringIO("\n".join(lines[header_index:])))
    table = table.loc[table["Asset Class"].astype(str).eq("Equity")].copy()
    table = table.loc[table["Ticker"].notna() & ~table["Ticker"].astype(str).isin(["-", "nan"])].copy()
    weight = pd.to_numeric(table.get("Weight (%)"), errors="coerce")
    return pd.DataFrame({
        "source_ticker": table["Ticker"].astype(str),
        "company": table["Name"].astype(str),
        "sector": table["Sector"].astype(str),
        "iwb_weight_percent": weight,
        "location": table.get("Location", "").astype(str),
        "exchange": table.get("Exchange", "").astype(str),
    }), as_of


def freeze_universe(config: dict[str, Any], refresh: bool = False) -> pd.DataFrame:
    out_dir = ROOT / config["project"]["output_dir"]
    sources_dir = out_dir / "universe_sources"
    mapping_path = out_dir / "stock_topic_mapping.csv"
    mapping = pd.read_csv(mapping_path, encoding="utf-8-sig")
    policy_tickers = set(mapping["ticker"].map(yahoo_ticker))
    source_specs = {
        "sp500": (config["universe"]["sp500_url"], sources_dir / "sp500.html"),
        "nasdaq100": (config["universe"]["nasdaq100_url"], sources_dir / "nasdaq100.json"),
        "iwb": (config["universe"]["russell1000_proxy_url"], sources_dir / "iwb_holdings.csv"),
    }
    payloads: dict[str, bytes] = {}
    source_audit: dict[str, Any] = {}
    for name, (url, path) in source_specs.items():
        content, cache_hit, retrieved_at = get_cached(url, path, refresh)
        payloads[name] = content
        source_audit[name] = {"url": url, "cache_path": str(path.relative_to(ROOT)), "cache_hit": cache_hit,
                              "retrieved_at_utc": retrieved_at, "sha256": hashlib.sha256(content).hexdigest()}

    sp = parse_sp500(payloads["sp500"]); nq, nq_as_of = parse_nasdaq100(payloads["nasdaq100"]); iwb, iwb_as_of = parse_iwb(payloads["iwb"])
    source_audit["sp500"].update({"list_date": "download date/current Wikipedia table", "parsed_rows": len(sp)})
    source_audit["nasdaq100"].update({"list_date": nq_as_of, "parsed_rows": len(nq), "source_type": "official Nasdaq public JSON endpoint"})
    source_audit["iwb"].update({"list_date": iwb_as_of, "parsed_equity_rows": len(iwb),
                                "proxy_for": "Russell 1000 constituents; ETF holdings are not the licensed index list"})

    records: dict[str, dict[str, Any]] = {}
    def merge(frame: pd.DataFrame, membership: str) -> None:
        for _, row in frame.iterrows():
            ticker = yahoo_ticker(row["source_ticker"])
            if not ticker or ticker in {"NAN", "-"}:
                continue
            rec = records.setdefault(ticker, {"ticker": ticker, "source_ticker": str(row["source_ticker"]),
                                              "company": str(row.get("company", "")), "sector": str(row.get("sector", "")),
                                              "sp500": False, "nasdaq100": False, "russell1000_proxy_iwb": False,
                                              "iwb_weight_percent": math.nan})
            rec[membership] = True
            if not rec["company"] or rec["company"] == "nan": rec["company"] = str(row.get("company", ""))
            if not rec["sector"] or rec["sector"] == "nan": rec["sector"] = str(row.get("sector", ""))
            if membership == "russell1000_proxy_iwb" and pd.notna(row.get("iwb_weight_percent")):
                rec["iwb_weight_percent"] = float(row["iwb_weight_percent"])
    merge(sp, "sp500"); merge(nq, "nasdaq100"); merge(iwb, "russell1000_proxy_iwb")
    for ticker in policy_tickers:
        if ticker not in records:
            row = mapping.loc[mapping["ticker"].map(yahoo_ticker).eq(ticker)].iloc[0]
            records[ticker] = {"ticker": ticker, "source_ticker": ticker, "company": row["company"], "sector": "",
                               "sp500": False, "nasdaq100": False, "russell1000_proxy_iwb": False,
                               "iwb_weight_percent": math.nan}
        records[ticker]["policy_mapping"] = True
    frame = pd.DataFrame(records.values())
    frame["policy_mapping"] = frame.get("policy_mapping", False).fillna(False).astype(bool)
    frame["membership_count"] = frame[["sp500", "nasdaq100", "russell1000_proxy_iwb"]].sum(axis=1)
    frame["priority_group"] = (~(frame["policy_mapping"] | frame["sp500"] | frame["nasdaq100"])).astype(int)
    frame = frame.sort_values(["priority_group", "membership_count", "iwb_weight_percent", "ticker"],
                              ascending=[True, False, False, True], na_position="last").reset_index(drop=True)
    maximum = int(config["universe"]["target_max_stocks"])
    frozen = frame.head(maximum).copy()
    frozen.insert(0, "universe_rank", range(1, len(frozen) + 1))
    frozen["selection_rule"] = "all policy-mapped, S&P 500 and Nasdaq-100 names, then IWB proxy holdings by published weight; capped before returns"
    frozen["source_snapshot_note"] = f"IWB holdings as of {iwb_as_of}; S&P/Nasdaq tables current at retrieval; survivorship bias possible"
    write_csv(out_dir / "frozen_stock_universe.csv", frozen)
    write_json(out_dir / "universe_sources.json", {
        "created_at_utc": utc_now(), "frozen_before_price_download": True, "selection_used_research_returns": False,
        "candidate_union_count": len(frame), "frozen_count": len(frozen), "target_max": maximum,
        "policy_mapping_tickers": sorted(policy_tickers), "sources": source_audit,
        "limitations": ["Current/near-current public lists may create survivorship bias.",
                        "IWB holdings are a transparent proxy for, not the licensed Russell 1000 constituent file."],
    })
    return frozen


def normalize_download(data: pd.DataFrame, requested: list[str]) -> pd.DataFrame:
    fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits", "Capital Gains"]
    rows: list[pd.DataFrame] = []
    if data is None or data.empty:
        return pd.DataFrame(columns=["date", "ticker", *fields])
    for ticker in requested:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                level0 = set(map(str, data.columns.get_level_values(0)))
                level1 = set(map(str, data.columns.get_level_values(1)))
                if ticker in level0:
                    sub = data[ticker].copy()
                elif ticker in level1:
                    sub = data.xs(ticker, axis=1, level=1).copy()
                else:
                    continue
            elif len(requested) == 1:
                sub = data.copy()
            else:
                continue
            sub.columns = [str(c) for c in sub.columns]
            sub = sub.reset_index()
            date_col = next(c for c in sub.columns if str(c).lower() in {"date", "datetime"})
            sub = sub.rename(columns={date_col: "date"})
            for field in fields:
                if field not in sub.columns: sub[field] = math.nan
            sub["date"] = pd.to_datetime(sub["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            sub["ticker"] = ticker
            sub = sub.loc[sub[fields].notna().any(axis=1), ["date", "ticker", *fields]]
            rows.append(sub)
        except Exception:
            continue
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["date", "ticker", *fields])


def download_batch(tickers: list[str], config: dict[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    import yfinance as yf
    p = config["prices"]; q = config["project"]
    attempts: list[dict[str, Any]] = []
    collected: list[pd.DataFrame] = []
    remaining = list(tickers)
    for attempt in range(1, int(p["max_retries"]) + 1):
        try:
            data = yf.download(tickers=remaining, start=q["price_download_start"], end=q["yfinance_end_exclusive"],
                               interval=p["interval"], auto_adjust=bool(p["auto_adjust"]), actions=bool(p["actions"]),
                               repair=bool(p["repair"]), group_by="ticker", multi_level_index=True,
                               threads=min(10, len(remaining)), timeout=int(p["timeout_seconds"]), progress=False, keepna=True)
            frame = normalize_download(data, remaining)
            got = sorted(frame["ticker"].unique()) if not frame.empty else []
            if not frame.empty: collected.append(frame)
            attempts.append({"attempt": attempt, "at_utc": utc_now(), "requested_tickers": list(remaining),
                             "returned_tickers": got, "error": None})
            remaining = sorted(set(remaining) - set(got))
            if not remaining: break
        except Exception as exc:
            attempts.append({"attempt": attempt, "at_utc": utc_now(), "requested_tickers": list(remaining), "returned_tickers": [],
                             "error": f"{type(exc).__name__}: {exc}"})
        if attempt < int(p["max_retries"]):
            time.sleep(float(p["backoff_seconds"]) * (2 ** (attempt - 1)))
    result = pd.concat(collected, ignore_index=True) if collected else pd.DataFrame(
        columns=["date", "ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits", "Capital Gains"]
    )
    if not result.empty: result = result.drop_duplicates(["ticker", "date"], keep="last")
    return result, attempts


def download_prices(config: dict[str, Any]) -> pd.DataFrame:
    import yfinance as yf
    out_dir = ROOT / config["project"]["output_dir"]
    universe_path = out_dir / "frozen_stock_universe.csv"
    if not universe_path.exists():
        raise FileNotFoundError("Freeze the universe first with --stage universe")
    universe = pd.read_csv(universe_path, encoding="utf-8-sig")
    benchmark = config["prices"]["benchmark"]
    etfs = list(dict.fromkeys(str(v) for v in config["prices"]["sector_etfs"].values()))
    stocks = universe["ticker"].astype(str).tolist()
    all_tickers = list(dict.fromkeys([benchmark, *etfs, *stocks]))
    cache_dir = out_dir / "raw_batches"; cache_dir.mkdir(parents=True, exist_ok=True)
    batch_size = int(config["prices"]["batch_size"])
    batches = [all_tickers[i:i + batch_size] for i in range(0, len(all_tickers), batch_size)]
    frames: list[pd.DataFrame] = []; manifest: list[dict[str, Any]] = []
    for index, batch in enumerate(batches, 1):
        key = hashlib.sha256(("|".join(batch) + json.dumps(config["project"], sort_keys=True)).encode()).hexdigest()[:12]
        cache_path = cache_dir / f"batch_{index:03d}_{key}.csv"
        audit_path = cache_dir / f"batch_{index:03d}_{key}.json"
        if cache_path.exists():
            frame = pd.read_csv(cache_path, encoding="utf-8-sig")
            attempts = [{"cache_hit": True, "at_utc": utc_now()}]
        else:
            frame, attempts = download_batch(batch, config)
            write_csv(cache_path, frame)
            write_json(audit_path, {"batch": index, "tickers": batch, "attempts": attempts})
        frames.append(frame)
        got = set(frame["ticker"].astype(str)) if not frame.empty else set()
        manifest.append({"batch": index, "cache_file": str(cache_path.relative_to(ROOT)), "requested": len(batch),
                         "returned": len(got), "missing_tickers": sorted(set(batch) - got), "attempts": attempts})
        print(f"batch {index}/{len(batches)}: {len(got)}/{len(batch)} tickers", flush=True)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    initially_got = set(combined["ticker"].astype(str)) if not combined.empty else set()
    initially_missing = sorted(set(all_tickers) - initially_got)
    if initially_missing:
        retry_key = hashlib.sha256(("|".join(initially_missing) + json.dumps(config["project"], sort_keys=True)).encode()).hexdigest()[:12]
        retry_cache = cache_dir / f"missing_retry_{retry_key}.csv"
        retry_audit = cache_dir / f"missing_retry_{retry_key}.json"
        if retry_cache.exists():
            retry_frame = pd.read_csv(retry_cache, encoding="utf-8-sig")
            retry_attempts = [{"cache_hit": True, "at_utc": utc_now()}]
        else:
            retry_frame, retry_attempts = download_batch(initially_missing, config)
            write_csv(retry_cache, retry_frame)
            write_json(retry_audit, {"tickers": initially_missing, "attempts": retry_attempts})
        if not retry_frame.empty:
            combined = pd.concat([combined, retry_frame], ignore_index=True)
        manifest.append({"batch": "missing_retry", "cache_file": str(retry_cache.relative_to(ROOT)),
                         "requested": len(initially_missing), "returned": int(retry_frame["ticker"].nunique()) if not retry_frame.empty else 0,
                         "missing_tickers": sorted(set(initially_missing) - set(retry_frame.get("ticker", pd.Series(dtype=str)).astype(str))),
                         "attempts": retry_attempts})
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    combined = combined.sort_values(["ticker", "date"])
    duplicates = combined.duplicated(["ticker", "date"], keep=False)
    conflicts = combined.loc[duplicates].copy()
    if not conflicts.empty:
        numeric = [c for c in combined.columns if c not in {"ticker", "date"}]
        counts = conflicts.groupby(["ticker", "date"])[numeric].nunique(dropna=False).max(axis=1)
        conflict_keys = set(counts[counts > 1].index)
        conflicts = conflicts.loc[conflicts.set_index(["ticker", "date"]).index.isin(conflict_keys)]
    write_csv(out_dir / "price_duplicate_conflicts.csv", conflicts)
    combined = combined.drop_duplicates(["ticker", "date"], keep="last").reset_index(drop=True)
    write_csv(out_dir / "raw_daily_prices.csv", combined)
    got = set(combined["ticker"].astype(str))
    missing = sorted(set(all_tickers) - got)
    failures = pd.DataFrame({"ticker": missing, "reason": "no rows returned by yfinance after cached/retried batch download"})
    write_csv(out_dir / "price_download_failures.csv", failures)
    coverage = combined.groupby("ticker").agg(first_date=("date", "min"), last_date=("date", "max"), observations=("date", "size")).reset_index()
    write_csv(out_dir / "price_download_coverage.csv", coverage)
    report = {
        "downloaded_at_utc": utc_now(), "yfinance_version": yf.__version__, "request": {
            "start": config["project"]["price_download_start"], "end_exclusive": config["project"]["yfinance_end_exclusive"],
            "interval": config["prices"]["interval"], "auto_adjust": config["prices"]["auto_adjust"],
            "actions": config["prices"]["actions"], "repair": config["prices"]["repair"],
        }, "frozen_stock_candidates": len(stocks), "requested_total_including_benchmark_etfs": len(all_tickers),
        "returned_total": len(got), "returned_stocks": len(set(stocks) & got), "missing_count": len(missing),
        "benchmark_returned": benchmark in got, "sector_etfs_returned": sorted(set(etfs) & got),
        "duplicate_conflict_rows": len(conflicts), "batches": manifest,
        "data_source_note": "yfinance is an unofficial research wrapper for Yahoo Finance; data are for academic use and were validated locally.",
    }
    write_json(out_dir / "price_download_report.json", report)
    return combined


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="equity_sentiment_config.yaml")
    parser.add_argument("--stage", choices=["universe", "prices", "all"], default="all")
    parser.add_argument("--refresh-universe", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    if args.stage in {"universe", "all"}:
        frozen = freeze_universe(config, refresh=args.refresh_universe)
        print(f"Frozen universe: {len(frozen)} stocks")
    if args.stage in {"prices", "all"}:
        frame = download_prices(config)
        print(f"Saved {len(frame)} ticker-date price rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
