"""Does FinLab revise published history beyond the universe and alpha? No LLM call, no network.

    .venv/bin/python -m eval.twstock.finlab_revision_scan

Reads finlab FileStorage files straight from disk (pandas, never finlab.data.get, so no cache
refresh is triggered) and compares every older local snapshot against the experiment cache,
cell by cell, on the (date, stock) cells both hold.

  adjusted prices (etl:adj_*)  compared as day-over-day ratios: a backward adjustment rescales a
                               stock's whole history at every ex-dividend date, which changes
                               levels without changing any return, so levels are not evidence.
  everything else              compared as stored values.

A cell counts as revised when both values are finite and |new - old| > 1e-6 * max(|old|, 1), and as
materially revised above 1e-3 relative. Cells that turned from a value into NaN or back are
counted apart. The known 09-11 vintage (the alpha panel the known-answer run wrote on 2026-09-13)
is also compared on raw h-day returns, which isolates prices from the universe change already
recorded (results/feedback_null/diagnose_20260914_215534.json).
"""

import datetime as _dt
import glob
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import records

CURRENT = "/home/ubuntu/finlab_cache_ace"
SNAPSHOTS = {
    "win_finlab_db": "/mnt/c/Users/User/finlab_db",
    "goldenai_lite": "/mnt/c/Users/User/Desktop/goldenai-lite/finlab_db",
    # WSL gets EACCES on the goldenai-lite originals (also on copies left under /mnt/c); this is a byte copy
    # written from Windows into the WSL filesystem on 2026-09-15 (Copy-Item keeps LastWriteTime 2026-07-16).
    # The md5 of every file read is recorded in scan.json.
    "goldenai_lite_copy": os.environ.get("SCAN_GOLDENAI_COPY", "/tmp/finlab_snapshot_goldenai_20260716"),
    "stock_analysis": "/home/ubuntu/stock-analysis/finlab_db",     # root-owned 0600 files: unreadable here
}
DATASETS = {
    "etl#adj_open": "ratio", "etl#adj_close": "ratio",
    "price#開盤價": "value", "price#收盤價": "value", "price#成交股數": "value",
    "etl#market_value": "value", "price_earning_ratio#本益比": "value",
    "financial_statement#每股盈餘": "value", "fundamental_features#ROE稅後": "value",
    "fundamental_features#營業利益率": "value", "monthly_revenue#當月營收": "value",
    "monthly_revenue#去年同月增減(%)": "value",
    "institutional_investors_trading_summary#外陸資買賣超股數(不含外資自營商)": "value",
}
WINDOWS = {"pre_2025-01-02_04-30_plus_h40": ("2025-01-02", "2025-07-01"),
           "post_2026-04-27_08-26_plus_h40": ("2026-04-27", "2026-12-31")}
REL, MATERIAL = 1e-6, 1e-3
KNOWN_PANEL = records.REPO_ROOT / "results" / "eval_layer" / "known_answer_20260913_172942" / "random_actual" / "decisions.parquet"
KNOWN_FACTORS = records.REPO_ROOT / "results" / "eval_layer" / "known_answer_20260913_173748"   # same 09-11 cache


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def locate(folder, key):
    for ext in (".feather", ".pickle"):
        p = Path(folder) / (key + ext)
        if p.exists():
            return p
    return None


def load(path):
    df = pd.read_feather(path) if path.suffix == ".feather" else pd.read_pickle(path)
    df = pd.DataFrame(df)
    if "date" in df.columns:
        df = df.set_index("date")
    try:
        df.index = pd.to_datetime(df.index)
    except (ValueError, TypeError):
        df.index = df.index.astype(str)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    # some files label columns "1101 台泥" (code and name); compare on the code
    df.columns = [str(c).split(" ")[0] for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated(keep="last")]
    return df.apply(pd.to_numeric, errors="coerce")


def ratio(df):
    return df / df.shift(1)


def date_str(x):
    return str(x.date()) if isinstance(x, pd.Timestamp) else str(x)


def compare(old, new, kind):
    if kind == "ratio":
        old, new = ratio(old), ratio(new)
        old = old.iloc[1:]
    rows = old.index.intersection(new.index)
    cols = old.columns.intersection(new.columns)
    a = old.loc[rows, cols].to_numpy(dtype=float)
    b = new.loc[rows, cols].to_numpy(dtype=float)
    both = np.isfinite(a) & np.isfinite(b)
    diff = np.where(both, np.abs(b - a), 0.0)
    scale = np.maximum(np.abs(a), 1.0)
    rel = np.where(both, diff / scale, 0.0)
    revised = both & (diff > REL * scale)
    material = both & (rel > MATERIAL)
    to_nan = np.isfinite(a) & ~np.isfinite(b)
    from_nan = ~np.isfinite(a) & np.isfinite(b)

    def where(mask):
        r_idx, c_idx = np.nonzero(mask)
        dates = pd.Index(rows[r_idx])
        by_date = pd.Series(1, index=dates).groupby(level=0).sum().sort_values(ascending=False)
        out = {"cells": int(mask.sum()), "dates": int(len(by_date)), "stocks": int(len(set(c_idx)))}
        if len(by_date):
            out["first_date"], out["last_date"] = date_str(dates.min()), date_str(dates.max())
            out["top_dates"] = {date_str(k): int(v) for k, v in by_date.head(8).items()}
            if isinstance(rows, pd.DatetimeIndex):
                out["in_windows"] = {w: int(((dates >= s) & (dates <= e)).sum()) for w, (s, e) in WINDOWS.items()}
        return out

    ex = []
    if revised.any():
        order = np.argsort(-rel[revised])[:5]
        r_idx, c_idx = np.nonzero(revised)
        for k in order:
            i, j = r_idx[k], c_idx[k]
            ex.append({"date": date_str(rows[i]), "stock": str(cols[j]), "old": float(a[i, j]), "new": float(b[i, j])})
    return {"compared_as": kind, "common_dates": int(len(rows)), "common_stocks": int(len(cols)),
            "common_first": date_str(rows.min()) if len(rows) else None,
            "common_last": date_str(rows.max()) if len(rows) else None,
            "cells_finite_both": int(both.sum()),
            "revised": where(revised), "materially_revised": where(material),
            "value_to_nan": where(to_nan), "nan_to_value": where(from_nan),
            "max_rel_change": float(rel.max()) if both.any() else None, "largest_examples": ex}


def top50_changes(old, new, start, end):
    rows = old.index.intersection(new.index)
    rows = rows[(rows >= start) & (rows <= end)]
    changed = []
    for d in rows:
        so, sn = set(old.loc[d].nlargest(50).index), set(new.loc[d].nlargest(50).index)
        if so != sn:
            changed.append({"date": date_str(d), "out": sorted(so - sn), "in": sorted(sn - so)})
    return {"dates_compared": int(len(rows)), "dates_with_different_top50": len(changed), "examples": changed[:10]}


def known_panel_returns(current_adj_open):
    """09-11 vintage raw returns r_h vs the current cache, on the recorded entry/maturity dates."""
    pnl = pd.read_parquet(KNOWN_PANEL)
    out = {"panel": str(KNOWN_PANEL), "columns": list(pnl.columns)}
    px = current_adj_open
    for h in (5, 10, 20, 40):
        rc, mc = f"r_h{h}", f"mature_date_h{h}"
        if rc not in pnl or mc not in pnl:
            continue
        g = pnl[pnl[rc].notna() & pnl[mc].notna()]
        e = px.stack().rename("px")
        entry = e.reindex(pd.MultiIndex.from_arrays([pd.to_datetime(g["entry_date"]), g["stock_id"].astype(str)])).to_numpy()
        mat = e.reindex(pd.MultiIndex.from_arrays([pd.to_datetime(g[mc]), g["stock_id"].astype(str)])).to_numpy()
        live = mat / entry - 1
        old = g[rc].to_numpy(dtype=float)
        ok = np.isfinite(live) & np.isfinite(old)
        d = np.abs(live - old)
        bad = ok & (d > REL)
        dates = pd.to_datetime(g.loc[bad, "decision_date"])
        out[f"h{h}"] = {"cells": int(ok.sum()), "revised_gt_1e-6": int(bad.sum()),
                        "max_abs_change": float(d[ok].max()) if ok.any() else None,
                        "decision_dates": sorted({str(x.date()) for x in dates})[:20],
                        "live_nan_where_recorded": int((np.isfinite(old) & ~np.isfinite(live)).sum())}
    return out


def known_factor_values(current):
    """Raw factor values the known-answer run recorded (data to 2026-09-11) vs the same code on the current
    cache, on the recorded universes: momentum reads etl:adj_close, value reads price_earning_ratio:本益比."""
    from . import factors
    out = {}
    for name, fn, key in (("momentum_20d", factors.momentum, "etl#adj_close"),
                          ("value_earnings_yield", factors.value, "price_earning_ratio#本益比")):
        path = KNOWN_FACTORS / name / "decisions.parquet"
        if key not in current or not path.exists():
            out[name] = {"skipped": f"missing {key if key not in current else path}"}
            continue
        try:
            rec = pd.read_parquet(path)
            rec["stock_id"] = rec["stock_id"].astype(str)
            dates = pd.DatetimeIndex(sorted(rec["decision_date"].unique()))
            U = {d: list(g.sort_values("universe_rank")["stock_id"]) for d, g in rec.groupby("decision_date")}
            cols = sorted({s for v in U.values() for s in v})
            live = fn(dates, U, current[key].reindex(columns=cols))
            live["stock_id"] = live["stock_id"].astype(str)
            m = rec[["decision_date", "stock_id", "raw_value"]].merge(
                live[["decision_date", "stock_id", "raw_value"]], on=["decision_date", "stock_id"], suffixes=("_0911", "_live"))
            a, b = m["raw_value_0911"].to_numpy(float), m["raw_value_live"].to_numpy(float)
            both = np.isfinite(a) & np.isfinite(b)
            diff = np.where(both, np.abs(b - a), 0.0)
            revised = both & (diff > REL * np.maximum(np.abs(a), 1.0))
            dates_rev = pd.to_datetime(m.loc[revised, "decision_date"])
            out[name] = {"source": str(path), "dataset": key, "rows": int(len(m)), "finite_both": int(both.sum()),
                         "revised": int(revised.sum()), "max_abs_change": float(diff.max()) if both.any() else None,
                         "value_to_nan": int((np.isfinite(a) & ~np.isfinite(b)).sum()),
                         "nan_to_value": int((~np.isfinite(a) & np.isfinite(b)).sum()),
                         "revised_dates": sorted({str(x.date()) for x in dates_rev})[:30],
                         "revised_in_windows": {w: int(((dates_rev >= s) & (dates_rev <= e)).sum())
                                                for w, (s, e) in WINDOWS.items()}}
        except Exception as e:
            out[name] = {"error": f"{type(e).__name__}: {e}"}
    return out


def main():
    out_dir =records.REPO_ROOT / "results" / "finlab_revisions" / f"scan_{_dt.datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"created_at": _dt.datetime.now().isoformat(timespec="seconds"), "git": records.git_state(),
              "rule": {"revised": f"|new-old| > {REL} * max(|old|, 1)", "material": f"relative > {MATERIAL}",
                       "adjusted_prices": "compared as day-over-day ratios (levels rescale at ex-dividend dates)"},
              "current": CURRENT, "datasets": {}}
    current = {}
    for key, kind in DATASETS.items():
        p = locate(CURRENT, key)
        entry = {"current_file": None}
        if p is None:
            report["datasets"][key] = entry
            continue
        cur = load(p)
        current[key] = cur
        entry = {"current_file": str(p), "current_mtime": _dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
                 "current_md5": md5(p), "current_last_index": date_str(cur.index.max()), "snapshots": {}}
        for name, folder in SNAPSHOTS.items():
            q = locate(folder, key)
            if q is None:
                continue
            try:
                old = load(q)
            except OSError as e:
                entry["snapshots"][name] = {"file": str(q), "unreadable": f"{type(e).__name__}: {e}"}
                print(f"{key:<45} vs {name:<15} UNREADABLE {type(e).__name__}", flush=True)
                continue
            res = {"file": str(q), "mtime": _dt.datetime.fromtimestamp(q.stat().st_mtime).isoformat(timespec="seconds"),
                   "md5": md5(q), "last_index": date_str(old.index.max())}
            res.update(compare(old, cur, kind))
            if key == "etl#market_value" and isinstance(old.index, pd.DatetimeIndex):
                res["top50"] = {w: top50_changes(old, cur, pd.Timestamp(s), pd.Timestamp(e)) for w, (s, e) in WINDOWS.items()}
            entry["snapshots"][name] = res
            rv = res["revised"]
            print(f"{key:<45} vs {name:<15} last {res['last_index']:<12} common {res['cells_finite_both']:>10} "
                  f"revised {rv['cells']:>8} on {rv['dates']:>5} dates / material {res['materially_revised']['cells']:>7} "
                  f"/ ->NaN {res['value_to_nan']['cells']:>6} / NaN-> {res['nan_to_value']['cells']:>6}", flush=True)
        report["datasets"][key] = entry
    if "etl#adj_open" in current and KNOWN_PANEL.exists():
        report["known_panel_0911_returns"] = known_panel_returns(current["etl#adj_open"])
        print(json.dumps(report["known_panel_0911_returns"], ensure_ascii=False)[:1500])
    report["known_factor_raw_values_0911"] = known_factor_values(current)
    print(json.dumps(report["known_factor_raw_values_0911"], ensure_ascii=False)[:2500])
    records.to_json(report, out_dir / "scan.json")
    print("written to", out_dir / "scan.json")


if __name__ == "__main__":
    main()
