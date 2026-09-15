"""On-disk record of one arm's decisions (v9 §6.4).

A field not written on the first run can only be recovered by re-running
everything, so the schema is fixed here and every arm writes through it - the
known-answer factors today, A0/A1/A2 later.

    <run_dir>/decisions.parquet      one row per (decision_date, stock_id): the
                                     whole universe with its score, not a selection
    <run_dir>/decision_meta.parquet  one row per decision_date
    <run_dir>/manifest.json          provenance for the run
    <run_dir>/llm_calls.parquet      one row per LLM request attempt (LLM arms only)

Schema 3 carries the output-handling protocol of eval.twstock.arm_protocol:
generation attempts, retries and void kind; for A2 the maturity processed that day
and the Reflector/Curator outcomes, counted apart from generation; for A1 the
decisions actually in the window and the slots left empty by void decisions.
A void date has no score at all; a date that is not void has every member scored.

Schema 4 adds the cost of every call (v9.3 §6.4 "每次 LLM 呼叫的 token 數與成本"): cost_usd and
aux_cost_usd per date, and llm_calls.parquet with tokens, latency, usage.cost and the provider
routing of each attempt. Before schema 4 cost sat only in the provider log and inside attempts_json.
"""

import datetime as _dt
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from . import ic
from .arm_protocol import attempt_cost
from .panel import HORIZONS, UNIVERSE_SIZE

SCHEMA_VERSION = 4
REPO_ROOT = Path(__file__).resolve().parents[2]
COST_SOURCE = ("usage.cost of each answered call as the gateway metered it (ClinePass quota metering at the "
               "reference price, not cash); a call that got no answer carries none - its HTTP status is in the "
               "provider log")

# Filled by LLM arms; left null by the known-answer factors.
LLM_FIELDS = {
    # generation
    "llm_raw_output": "string",
    "prompt_tokens": "Int64",
    "completion_tokens": "Int64",
    "n_llm_calls": "Int64",
    "latency_s": "Float64",
    "cost_usd": "Float64",                # sum over this date's generation attempts
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
    "aux_cost_usd": "Float64",            # sum over the Reflector and Curator attempts
    "aux_attempts_json": "string",
    # A1 window
    "window_decision_dates": "string",
    "window_skipped_void": "Int64",
}

# <run_dir>/llm_calls.parquet: one row per request attempt, unpacked from attempts_json and
# aux_attempts_json, so the per-call rows and the per-date totals come from the same records.
PROVIDER_CALL_FIELDS = ("generation_id", "http_status", "finish_reason", "finalProvider", "resolvedProvider",
                        "modelAttemptCount", "totalProviderAttemptCount")
LLM_CALL_FIELDS = {
    "arm": "string",
    "decision_date": "datetime64[ns]",
    "role": "string",                     # generator | reflector | curator
    "matured_decision_date": "string",    # reflector / curator: the decision whose feedback was processed
    "call_id": "string",
    "attempt": "Int64",
    "ok_call": "boolean",
    "failure": "string",
    "error": "string",
    "prompt_tokens": "Int64",
    "completion_tokens": "Int64",
    "latency_s": "Float64",
    "cost_usd": "Float64",
    "generation_id": "string",
    "http_status": "Int64",
    "finish_reason": "string",
    "finalProvider": "string",
    "resolvedProvider": "string",
    "modelAttemptCount": "Int64",
    "totalProviderAttemptCount": "Int64",
}


# Reserved for the A2 driver (M5a): <run_dir>/playbook_bullets.parquet, one row per bullet
# version, so every playbook entry can be traced to the maturity that produced it (v9 §6.4)
# and classified later for memory health (v9 §6.3). Nothing writes it yet.
#
# specificity_class exists because the Reflector sees realized returns: entries may be
# "memorised answers" bound to one ticker and one period ("TSMC is a good stock") instead of
# transferable judgement rules. It is filled by a later, separate classification pass.
PLAYBOOK_BULLET_FIELDS = {
    "arm": "string",
    "bullet_id": "string",
    "section": "string",
    "content": "string",
    "operation": "string",                   # ADD | UPDATE | DELETE, as applied by the Curator
    "created_decision_date": "string",       # decision date on which the Curator call ran
    "source_matured_decision_date": "string",  # the matured decision whose feedback produced it
    "helpful_count": "Int64",
    "harmful_count": "Int64",
    "deleted_decision_date": "string",
    "specificity_class": "string",           # transferable_rule | ticker_or_period_specific | mixed | unclassified
    "specificity_note": "string",
    "specificity_classified_by": "string",   # who or what assigned the class, and when
}
SPECIFICITY_CLASSES = ("transferable_rule", "ticker_or_period_specific", "mixed", "unclassified")


def empty_bullet_table():
    """Typed empty playbook_bullets table, so every writer starts from the same columns."""
    return pd.DataFrame({k: pd.Series(dtype=v) for k, v in PLAYBOOK_BULLET_FIELDS.items()})


def _na(v):
    return None if not isinstance(v, (list, dict)) and pd.isna(v) else v


def llm_call_table(meta, arm):
    """One row per request attempt of an LLM arm, unpacked from attempts_json and aux_attempts_json."""
    rows = []
    for m in meta.to_dict(orient="records"):
        for col, role in (("attempts_json", "generator"), ("aux_attempts_json", None)):
            js = _na(m.get(col))
            if js is None:
                continue
            for a in json.loads(js):
                prov = a.get("provider") or {}
                rows.append({
                    "arm": arm, "decision_date": m["decision_date"], "role": role or a.get("role"),
                    "matured_decision_date": None if role else _na(m.get("matured_decision_date")),
                    "call_id": a.get("call_id"), "attempt": a.get("attempt"), "ok_call": a.get("ok_call"),
                    "failure": a.get("failure"), "error": a.get("error"),
                    "prompt_tokens": a.get("prompt_tokens"), "completion_tokens": a.get("completion_tokens"),
                    "latency_s": a.get("latency_s"), "cost_usd": attempt_cost(a),
                    **{k: prov.get(k) for k in PROVIDER_CALL_FIELDS}})
    if not rows:
        return pd.DataFrame({k: pd.Series(dtype=v) for k, v in LLM_CALL_FIELDS.items()})
    df = pd.DataFrame(rows)
    return pd.DataFrame({k: pd.to_datetime(df[k]) if dtype.startswith("datetime")
                         else pd.array([_na(v) for v in df[k]], dtype=dtype)
                         for k, dtype in LLM_CALL_FIELDS.items()})


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
        calls = llm_call_table(meta, arm)
        calls.to_parquet(run_dir / "llm_calls.parquet", index=False)
        by_role = calls.groupby("role")["cost_usd"].sum(min_count=1)
        manifest["llm_calls"] = int(len(calls))
        manifest["cost_usd_by_role"] = {k: (None if pd.isna(v) else float(v)) for k, v in by_role.items()}
        manifest["cost_source"] = COST_SOURCE
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
