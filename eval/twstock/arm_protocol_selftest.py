"""Offline self-check of the output-handling protocol and its on-disk record. No LLM, no data.

    .venv/bin/python -m eval.twstock.arm_protocol_selftest

Pass criteria, all must hold:
 1. both final_answer forms are accepted, and the form is recorded
 2. a call that raises twice and then succeeds uses 3 attempts / 2 retries, all recorded
 3. an answer missing one stock is retried; after 3 incomplete attempts the point is void
    with reason "incomplete" and no scores
 4. a numeric string value counts as incomplete (only numbers in [-1, 1] score)
 5. never more than 1 + MAX_RETRIES calls per decision point
 6. write_run stores a void date with no score and a complete date with all 50, and
    refuses a void date that carries scores and a non-void date that lacks some
 7. void_summary counts per arm: points, voids, retries, forms over every attempt
Exit status 1 on the first failure.
"""

import json
import sys
import tempfile

import numpy as np
import pandas as pd

from . import arm_protocol as ap
from . import records

U = [str(1101 + i) for i in range(50)]
HORIZONS = (5, 10, 20, 40)


def answer(form="object", drop=0, override=None):
    scores = {s: round(((i % 21) - 10) / 10, 1) for i, s in enumerate(U[:len(U) - drop])}
    scores.update(override or {})
    fa = scores if form == "object" else json.dumps(scores)
    return json.dumps({"reasoning": "r", "bullet_ids": [], "final_answer": fa})


def scripted(outputs):
    """A call that replays outputs in order; an Exception instance is raised."""
    log = []

    def call(attempt_no):
        log.append(attempt_no)
        item = outputs[min(len(log), len(outputs)) - 1]
        if isinstance(item, Exception):
            raise item
        return {"response": item, "prompt_tokens": 100, "completion_tokens": 10, "latency_s": 1.5}
    return call, log


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  {detail}" if not cond else ""))
    if not cond:
        sys.exit(1)


def synthetic_panel(dates):
    rows = []
    for t in dates:
        for r, s in enumerate(U, 1):
            row = {"decision_date": t, "stock_id": s, "universe_rank": r, "entry_date": t + pd.Timedelta(days=1)}
            for h in HORIZONS:
                row[f"r_h{h}"] = 0.0
                row[f"bench_r_h{h}"] = 0.0
                row[f"bench_n_h{h}"] = 50
                row[f"alpha_h{h}"] = 0.0
                row[f"mature_date_h{h}"] = t + pd.Timedelta(days=h + 1)
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    # 1. forms
    call, _ = scripted([answer("object")])
    o1 = ap.run_decision(call, U)
    call, _ = scripted([answer("json_string")])
    o2 = ap.run_decision(call, U)
    check("1 object accepted and recorded", not o1["voided"] and o1["final_answer_form"] == "object"
          and len(o1["scores"]) == 50)
    check("1 JSON string accepted and recorded", not o2["voided"] and o2["final_answer_form"] == "json_string"
          and o2["scores"] == o1["scores"])

    # 2. call errors then success
    call, log = scripted([RuntimeError("HTTP 500"), RuntimeError("timeout"), answer("object")])
    o3 = ap.run_decision(call, U)
    check("2 raises twice then succeeds: 3 attempts, 2 retries", not o3["voided"] and o3["n_attempts"] == 3
          and o3["n_retries"] == 2 and log == [1, 2, 3])
    check("2 every attempt recorded with its failure",
          [a["failure"] for a in o3["attempts"]] == ["call_error", "call_error", None]
          and "HTTP 500" in o3["attempts"][0]["error"])

    # 3. missing one stock three times -> void
    call, log = scripted([answer("object", drop=1)])
    o4 = ap.run_decision(call, U)
    check("3 incomplete x3 is void, no scores", o4["voided"] and o4["void_reason"] == "incomplete"
          and o4["scores"] is None and o4["n_attempts"] == 3)
    check("3 the missing stock is recorded on each attempt",
          all(a["missing"] == [U[-1]] for a in o4["attempts"]))

    # 4. numeric string is incomplete, then a clean retry
    call, log = scripted([answer("object", override={U[0]: "0.3"}), answer("json_string")])
    o5 = ap.run_decision(call, U)
    check("4 numeric string forces a retry", not o5["voided"] and o5["n_retries"] == 1
          and o5["attempts"][0]["failure"] == "incomplete"
          and o5["attempts"][0]["value_forms"].get("numeric_string") == 1)

    # 5. call cap
    call, log = scripted([RuntimeError("down")])
    o6 = ap.run_decision(call, U)
    check("5 at most 1 + MAX_RETRIES calls", len(log) == 1 + ap.MAX_RETRIES and o6["voided"]
          and o6["void_reason"] == "call_error")
    call, log = scripted(["not json"])
    o7 = ap.run_decision(call, U)
    check("5 non-JSON output is void with reason json", o7["voided"] and o7["void_reason"] == "json")

    # 6. on-disk record
    dates = pd.DatetimeIndex(["2026-04-27", "2026-04-28"])
    pnl = synthetic_panel(dates)
    good, void = o2, o4
    scores = pd.concat([ap.score_rows(dates[0], good), ap.score_rows(dates[1], void)], ignore_index=True)
    meta_rows = [ap.meta_row(dates[0], good), ap.meta_row(dates[1], void)]
    with tempfile.TemporaryDirectory() as tmp:
        records.write_run(tmp, pnl, scores, "A0", llm_meta=meta_rows)
        dec, meta, man = records.read_run(tmp)
        m = meta.set_index("decision_date")
        check("6 complete date: 50 scored, not void, form recorded",
              m.loc[dates[0], "n_scored"] == 50 and not m.loc[dates[0], "voided"]
              and m.loc[dates[0], "final_answer_form"] == "json_string")
        check("6 void date: no score, void, reason and 3 attempts kept",
              m.loc[dates[1], "n_scored"] == 0 and bool(m.loc[dates[1], "voided"])
              and m.loc[dates[1], "void_reason"] == "incomplete"
              and len(json.loads(m.loc[dates[1], "attempts_json"])) == 3
              and dec.loc[dec["decision_date"] == dates[1], "score"].isna().all())
        check("6 manifest counts voids, schema 2", man["voided_decision_points"] == 1
              and man["schema_version"] == 2)
        summary = ap.void_summary(meta)

    with tempfile.TemporaryDirectory() as tmp:
        bad = pd.concat([ap.score_rows(dates[0], good), ap.score_rows(dates[1], good)], ignore_index=True)
        try:
            records.write_run(tmp, pnl, bad, "A0", llm_meta=[ap.meta_row(dates[0], good), ap.meta_row(dates[1], void)])
            refused = False
        except ValueError:
            refused = True
        check("6 refuses a void date that carries scores", refused)
    with tempfile.TemporaryDirectory() as tmp:
        partial = ap.score_rows(dates[0], good).iloc[:49]
        partial = pd.concat([partial, ap.score_rows(dates[1], void)], ignore_index=True)
        try:
            records.write_run(tmp, pnl, partial, "A0", llm_meta=meta_rows)
            refused = False
        except ValueError:
            refused = True
        check("6 refuses a non-void date missing a member's score", refused)

    # 7. summary
    row = summary.iloc[0]
    check("7 void_summary per arm", row["decision_points"] == 2 and row["voided"] == 1
          and row["retries_spent"] == 2 and row["points_needing_retry"] == 1
          and row["attempt_forms"] == {"json_string": 1, "object": 3}, row.to_dict())
    print("ALL PASS")


if __name__ == "__main__":
    main()
