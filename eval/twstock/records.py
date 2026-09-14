"""On-disk record of one arm's decisions (v8 §6.4).

A field not written on the first run can only be recovered by re-running
everything, so the schema is fixed here and every arm writes through it - the
known-answer factors today, A0/A1/A2 later.

    <run_dir>/decisions.parquet      one row per (decision_date, stock_id): the
                                     whole universe with its score, not a selection
    <run_dir>/decision_meta.parquet  one row per decision_date
    <run_dir>/manifest.json          provenance for the run
"""

import datetime as _dt
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from . import ic
from .panel import HORIZONS, UNIVERSE_SIZE

SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[2]

# Filled by LLM arms; left null by the known-answer factors.
LLM_FIELDS = {
    "llm_raw_output": "string",
    "prompt_tokens": "Int64",
    "completion_tokens": "Int64",
    "n_llm_calls": "Int64",
    "latency_s": "Float64",
}


def git_state():
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                               capture_output=True, text=True, timeout=10)
        return {"commit": rev.stdout.strip() or None, "dirty": bool(dirty.stdout.strip())}
    except Exception:
        return {"commit": None, "dirty": None}


def _json_default(o):
    if isinstance(o, (pd.Timestamp, _dt.datetime, _dt.date)):
        return o.isoformat()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, (np.ndarray, pd.Index)):
        return list(o)
    return str(o)


def to_json(obj, path):
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8")


def write_run(run_dir, panel, scores, arm, extra=None, horizons=HORIZONS):
    """Join scores onto the α panel and write the three files."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    keys = ["decision_date", "stock_id"]

    stray = scores[keys].merge(panel[keys], on=keys, how="left", indicator=True)
    n_stray = int((stray["_merge"] == "left_only").sum())
    if n_stray:
        raise ValueError(f"{n_stray} score rows fall outside the point-in-time universe")

    cols = keys + [c for c in ("score", "raw_value") if c in scores.columns]
    decisions = panel.merge(scores[cols], on=keys, how="left", validate="one_to_one")
    decisions.insert(0, "arm", arm)
    decisions.to_parquet(run_dir / "decisions.parquet", index=False)

    per_date = []
    ordered = decisions.sort_values(["decision_date", "universe_rank"])
    for t, g in ordered.groupby("decision_date", sort=True):
        row = {
            "decision_date": t,
            "arm": arm,
            "universe": json.dumps(list(g["stock_id"])),
            "n_universe": int(len(g)),
            "n_scored": int(g["score"].notna().sum()),
            "entry_date": g["entry_date"].iloc[0],
        }
        for h in horizons:
            row[f"mature_date_h{h}"] = g[f"mature_date_h{h}"].iloc[0]
            row[f"bench_r_h{h}"] = g[f"bench_r_h{h}"].iloc[0]
            row[f"bench_n_h{h}"] = int(g[f"bench_n_h{h}"].iloc[0])
        per_date.append(row)
    meta = pd.DataFrame(per_date)
    for name, dtype in LLM_FIELDS.items():
        meta[name] = pd.Series([pd.NA] * len(meta), dtype=dtype)
    meta.to_parquet(run_dir / "decision_meta.parquet", index=False)

    dates = pd.DatetimeIndex(meta["decision_date"])
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "arm": arm,
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "git": git_state(),
        "horizons": list(horizons),
        "universe": f"point-in-time top {UNIVERSE_SIZE} by etl:market_value",
        "price_field": "etl:adj_open",
        "alpha_definition": ("r(i,t,h) = open(i,t+1+h)/open(i,t+1) - 1; "
                             "alpha = r - equal-weight mean of r over members with a valid r"),
        "maturity": "mature_date_h = trading day t+1+h, usable from its close (delay h+1)",
        "cs_min_pairs": ic.CS_MIN_PAIRS,
        "ts_min_obs": ic.TS_MIN_OBS,
        "decision_dates": {"first": dates.min(), "last": dates.max(), "count": len(dates)},
        **(extra or {}),
    }
    to_json(manifest, run_dir / "manifest.json")
    return run_dir


def read_run(run_dir):
    run_dir = Path(run_dir)
    decisions = pd.read_parquet(run_dir / "decisions.parquet")
    meta = pd.read_parquet(run_dir / "decision_meta.parquet")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    return decisions, meta, manifest
