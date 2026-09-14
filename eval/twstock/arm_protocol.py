"""Per-decision output handling shared by every arm (protocol fixed 2026-09-14).

1. final_answer may be a JSON object or a string holding one; both are accepted and
   the form is recorded as final_answer_form. This is parser tolerance, not an
   experiment parameter - but a form-jitter rate that differs by arm is a finding.
2. A decision point gets at most MAX_RETRIES = 2 retries after its first attempt, the
   same rule for every arm, and every attempt is recorded. An attempt is spent when
   it is not complete: the call raised, the output is not JSON, final_answer is in
   neither accepted form, or it does not give every universe member a number in [-1, 1].
3. When the retries are used up the decision point is void as a whole: no score is
   filled in and no single stock is dropped, so every scored date carries the full
   cross-section and csIC is always computed on the same base. Void counts are
   reported per arm (void_summary).

This must be the only retry layer. Run with ACE_MAX_RETRIES=1 and an OpenAI client
with max_retries=0; otherwise retries happen below the attempts counted here.

Not settled by the protocol, so recorded but not acted on: keys outside the universe
and duplicate keys in an answer that otherwise scores all members.
"""

import json
import re

import pandas as pd

MAX_RETRIES = 2
ACCEPTED_FORMS = ("object", "json_string")

NUM_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)$")


def classify_value(v):
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, (int, float)):
        return "in_range" if -1 <= v <= 1 else "number_out_of_range"
    if v is None:
        return "null"
    if isinstance(v, str):
        s = v.strip()
        if NUM_RE.match(s):
            return "numeric_string"
        if s.endswith("%") and NUM_RE.match(s[:-1].strip()):
            return "percent_string"
        return "text"
    return "nested_" + type(v).__name__


def analyse(raw, universe):
    out = {"json_ok": False, "top_keys": None, "final_answer_form": None, "n_returned": 0,
           "n_valid_ids": 0, "missing": [], "extra_keys": [], "duplicate_keys": [],
           "value_forms": {}, "out_of_range_examples": [], "scores": {}}
    dups = []

    def hook(pairs):
        d = {}
        for k, v in pairs:
            if k in d:
                dups.append(k)
            d[k] = v
        return d

    try:
        obj = json.loads(raw, object_pairs_hook=hook)
    except Exception as e:
        out["json_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        return out
    if not isinstance(obj, dict):
        out["json_error"] = f"top level is {type(obj).__name__}"
        return out
    out["json_ok"] = True
    out["top_keys"] = sorted(obj)
    fa = obj.get("final_answer")
    if isinstance(fa, dict):
        out["final_answer_form"] = "object"
        scores = fa
    elif isinstance(fa, str):
        try:
            inner = json.loads(fa, object_pairs_hook=hook)
            out["final_answer_form"] = "json_string" if isinstance(inner, dict) else "string_other"
            scores = inner if isinstance(inner, dict) else None
        except Exception:
            out["final_answer_form"] = "text"
            scores = None
    elif fa is None:
        out["final_answer_form"] = "missing"
        scores = None
    else:
        out["final_answer_form"] = type(fa).__name__
        scores = None
    out["duplicate_keys"] = dups
    if scores is None:
        return out

    members = set(universe)
    keys = {str(k).strip(): v for k, v in scores.items()}
    out["n_returned"] = len(keys)
    valid = {k: v for k, v in keys.items() if k in members}
    out["n_valid_ids"] = len(valid)
    out["missing"] = [s for s in universe if s not in valid]
    out["extra_keys"] = [k for k in keys if k not in members]
    forms = {}
    for k, v in valid.items():
        f = classify_value(v)
        forms[f] = forms.get(f, 0) + 1
        if f != "in_range" and len(out["out_of_range_examples"]) < 10:
            out["out_of_range_examples"].append({k: v})
        if f == "in_range":
            out["scores"][k] = float(v)
    out["value_forms"] = forms
    return out


def failure_kind(analysis, universe):
    """None when the attempt is usable, else why it is not."""
    if not analysis["json_ok"]:
        return "json"
    if analysis["final_answer_form"] not in ACCEPTED_FORMS:
        return "final_answer_form"
    if analysis["missing"] or len(analysis["scores"]) != len(universe):
        return "incomplete"
    return None


def run_decision(call, universe, max_retries=MAX_RETRIES):
    """Attempt one decision point until complete or out of retries.

    call(attempt_no) returns a dict with the raw text under "response" plus any
    per-attempt facts to keep (tokens, latency, provider fields); it may raise.
    """
    attempts = []
    for attempt_no in range(1, max_retries + 2):
        rec = {"attempt": attempt_no}
        try:
            out = call(attempt_no)
            rec.update(out)
            rec["ok_call"] = True
            analysis = analyse(out.get("response") or "", universe)
            kind = failure_kind(analysis, universe)
        except Exception as e:
            rec["ok_call"] = False
            rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            analysis = analyse("", universe)
            kind = "call_error"
        rec.update({
            "failure": kind,
            "json_ok": analysis["json_ok"],
            "json_error": analysis.get("json_error"),
            "final_answer_form": analysis["final_answer_form"],
            "n_valid_ids": analysis["n_valid_ids"],
            "n_scored_in_range": len(analysis["scores"]),
            "missing": analysis["missing"],
            "extra_keys": analysis["extra_keys"],
            "duplicate_keys": analysis["duplicate_keys"],
            "value_forms": analysis["value_forms"],
        })
        attempts.append(rec)
        if kind is None:
            return {"voided": False, "void_reason": None, "scores": analysis["scores"],
                    "final_answer_form": analysis["final_answer_form"],
                    "n_attempts": attempt_no, "n_retries": attempt_no - 1, "attempts": attempts}
    return {"voided": True, "void_reason": attempts[-1]["failure"], "scores": None,
            "final_answer_form": None, "n_attempts": len(attempts),
            "n_retries": len(attempts) - 1, "attempts": attempts}


def score_rows(decision_date, outcome):
    """Score rows for records.write_run: the whole universe, or nothing for a void date."""
    if outcome["voided"]:
        return pd.DataFrame(columns=["decision_date", "stock_id", "score"])
    return pd.DataFrame({"decision_date": pd.Timestamp(decision_date),
                         "stock_id": list(outcome["scores"]),
                         "score": list(outcome["scores"].values())})


def meta_row(decision_date, outcome):
    """One llm_meta row for records.write_run."""
    atts = outcome["attempts"]

    def total(key):
        vals = [a.get(key) for a in atts if a.get(key) is not None]
        return sum(vals) if vals else None

    return {
        "decision_date": pd.Timestamp(decision_date),
        "final_answer_form": outcome["final_answer_form"],
        "n_attempts": outcome["n_attempts"],
        "n_retries": outcome["n_retries"],
        "voided": outcome["voided"],
        "void_reason": outcome["void_reason"],
        "attempts_json": json.dumps(atts, ensure_ascii=False, default=str),
        "llm_raw_output": atts[-1].get("response"),
        "prompt_tokens": total("prompt_tokens"),
        "completion_tokens": total("completion_tokens"),
        "n_llm_calls": len(atts),
        "latency_s": total("latency_s"),
    }


def void_summary(meta):
    """Per arm: void count and rate, retries spent, and final_answer forms over every attempt."""
    rows = []
    for arm, g in meta.groupby("arm", sort=True):
        forms = {}
        for js in g["attempts_json"].dropna():
            for a in json.loads(js):
                f = a.get("final_answer_form") or "none"
                forms[f] = forms.get(f, 0) + 1
        voided = g["voided"].fillna(False).astype(bool)
        rows.append({
            "arm": arm,
            "decision_points": int(len(g)),
            "voided": int(voided.sum()),
            "void_rate": float(voided.mean()) if len(g) else float("nan"),
            "void_reasons": g.loc[voided, "void_reason"].value_counts().to_dict(),
            "points_needing_retry": int((g["n_retries"].fillna(0) > 0).sum()),
            "retries_spent": int(g["n_retries"].fillna(0).sum()),
            "attempt_forms": forms,
        })
    return pd.DataFrame(rows)
