"""On-disk record of one arm's decisions (v8 §6.4).

A field not written on the first run can only be recovered by re-running
everything, so the schema is fixed here and every arm writes through it - the
known-answer factors today, A0/A1/A2 later.

    <run_dir>/decisions.parquet      one row per (decision_date, stock_id): the
                                     whole universe with its score, not a selection
    <run_dir>/decision_meta.parquet  one row per decision_date
    <run_dir>/manifest.json          provenance for the run

Schema 3 carries the output-handling protocol of eval.twstock.arm_protocol:
generation attempts, retries and void kind; for A2 the maturity processed that day
and the Reflector/Curator outcomes, counted apart from generation; for A1 the
decisions actually in the window and the slots left empty by void decisions.
A void date has no score at all; a date that is not void has every member scored.
"""

import datetime as _dt
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from . import ic
from .panel import HORIZONS, UNIVERSE_SIZE

SCHEMA_VERSION = 3
REPO_ROOT = Path(__file__).resolve().parents[2]

# Filled by LLM arms; left null by the known-answer factors.
LLM_FIELDS = {
    # generation
    "llm_raw_output": "string",
    "prompt_tokens": "Int64",
    "completion_tokens": "Int64",
    "n_llm_calls": "Int64",
    "latency_s": "Float64",
    "final_answer_form": "string",
    "n_attempts": "Int64",
    "n_retries": "Int64",
    "voided": "boolean",
    "void_reason": "string",
    "void_kinds": "string",
    "attempts_json": "string",
    # A2 learning, counted apart from generation
    "maturity_status": "string",          # none | processed | skipped_void
    "matured_decision_date": "string",
    "reflector_status": "string",         # ok | failed | not_run
    "reflector_attempts": "Int64",
    "reflector_retries": "Int64",
    "curator_status": "string",           # ok | failed | not_run
    "curator_attempts": "Int64",
    "curator_retries": "Int64",
    "aux_attempts_json": "string",
    # A1 window
    "window_decision_dates": "string",
    "window_skipped_void": "Int64",
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


def _check_learning_fields(meta):
    """A skipped or absent maturity makes no learning call; a failed Reflector means no Curator."""
    ms, rs, cs = meta["maturity_status"], meta.get("reflector_status"), meta.get("curator_status")
    known = ms.notna()
    bad_status = meta.loc[known & ~ms.isin(["none", "processed", "skipped_void"]), "decision_date"]
    if len(bad_status):
        raise ValueError(f"unknown maturity_status on {list(bad_status)[:5]}")
    if rs is None or cs is None:
        raise ValueError("maturity_status needs reflector_status and curator_status")
    no_call = known & ms.isin(["none", "skipped_void"]) & ((rs != "not_run") | (cs != "not_run"))
    if no_call.any():
        raise ValueError(f"learning calls recorded for a maturity that was absent or void: "
                         f"{list(meta.loc[no_call, 'decision_date'])[:5]}")
    processed = known & (ms == "processed")
    bad_refl = processed & ~rs.isin(["ok", "failed"])
    if bad_refl.any():
        raise ValueError(f"processed maturity without a Reflector outcome: {list(meta.loc[bad_refl, 'decision_date'])[:5]}")
    curated_after_fail = processed & (rs == "failed") & (cs != "not_run")
    if curated_after_fail.any():
        raise ValueError(f"Curator recorded after a failed Reflector: {list(meta.loc[curated_after_fail, 'decision_date'])[:5]}")


def _fill_llm_meta(meta, llm_meta):
    """Copy per-date LLM fields onto meta and enforce the protocol against the scores."""
    lm = pd.DataFrame(llm_meta).copy()
    unknown = set(lm.columns) - set(LLM_FIELDS) - {"decision_date"}
    if unknown:
        raise ValueError(f"llm_meta has fields outside the schema: {sorted(unknown)}")
    lm["decision_date"] = pd.to_datetime(lm["decision_date"])
    if lm["decision_date"].duplicated().any():
        raise ValueError("llm_meta has more than one row for a decision date")
    dates = set(pd.to_datetime(meta["decision_date"]))
    if set(lm["decision_date"]) != dates:
        raise ValueError("llm_meta must have exactly one row for every decision date of the run")
    lm = lm.set_index("decision_date")
    keys = pd.to_datetime(meta["decision_date"])
    for name in lm.columns:
        values = [None if not isinstance(v, (list, dict)) and pd.isna(v) else v for v in keys.map(lm[name])]
        meta[name] = pd.array(values, dtype=LLM_FIELDS[name])

    if "voided" in lm.columns:
        voided = meta["voided"].fillna(False).astype(bool)
        known = meta["voided"].notna()
        bad_void = meta.loc[voided & (meta["n_scored"] > 0), "decision_date"]
        if len(bad_void):
            raise ValueError(f"void dates carry scores: {list(bad_void)[:5]}")
        bad_full = meta.loc[known & ~voided & (meta["n_scored"] != meta["n_universe"]), "decision_date"]
        if len(bad_full):
            raise ValueError(f"dates that are not void lack scores for some members: {list(bad_full)[:5]}")
    if "maturity_status" in lm.columns:
        _check_learning_fields(meta)
    return meta


def write_run(run_dir, panel, scores, arm, extra=None, horizons=HORIZONS, llm_meta=None):
    """Join scores onto the α panel and write the three files.

    llm_meta: one row per decision date with LLM_FIELDS (see arm_protocol.meta_row);
    None for the known-answer factors, whose LLM fields stay null.
    """
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
    if llm_meta is not None:
        meta = _fill_llm_meta(meta, llm_meta)
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
        "output_protocol": ("arm_protocol: final_answer object or JSON string accepted and recorded; "
                            "max 2 retries per decision point and per Reflector/Curator call, every attempt "
                            "recorded; a decision point still unusable after retries (incomplete, duplicate "
                            "key or out-of-universe key) is void as a whole; a void decision is skipped at "
                            "maturity without replacement"),
        **(extra or {}),
    }
    if llm_meta is not None:
        if meta["voided"].notna().any():
            manifest["voided_decision_points"] = int(meta["voided"].fillna(False).astype(bool).sum())
        if meta["maturity_status"].notna().any():
            manifest["maturities_skipped_void"] = int((meta["maturity_status"] == "skipped_void").sum())
            manifest["reflector_failed"] = int((meta["reflector_status"] == "failed").sum())
            manifest["curator_failed"] = int((meta["curator_status"] == "failed").sum())
        if meta["window_skipped_void"].notna().any():
            manifest["window_slots_skipped_void"] = int(meta["window_skipped_void"].fillna(0).sum())
    to_json(manifest, run_dir / "manifest.json")
    return run_dir


def read_run(run_dir):
    run_dir = Path(run_dir)
    decisions = pd.read_parquet(run_dir / "decisions.parquet")
    meta = pd.read_parquet(run_dir / "decision_meta.parquet")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    return decisions, meta, manifest
