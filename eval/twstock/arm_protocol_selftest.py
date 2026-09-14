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
 8. an out-of-universe key voids the point even with all 50 scored, as its own kind
 9. a duplicated score key voids the point (object form, JSON-string form, and a
    collision after stripping whitespace); a repeated "reasoning" key does not
10. when both apply, the named reason is out_of_universe_key and both kinds are kept
11. Reflector/Curator calls follow the same retry rule and are recorded on their own;
    a Reflector that fails after its retries means no Curator call
12. a void decision is skipped at maturity with no call; A1's window leaves its slot
    empty instead of reaching further back
13. the record refuses learning calls on a skipped maturity; learning_summary counts
    skips, failures and playbook updates apart from generation voids
Exit status 1 on the first failure.
"""

import json
import sys
import tempfile
import warnings

import pandas as pd

from . import arm_protocol as ap
from . import records

warnings.filterwarnings("ignore", category=FutureWarning)

U = [str(1101 + i) for i in range(50)]
HORIZONS = (5, 10, 20, 40)


def score_dict(drop=0, override=None):
    scores = {s: round(((i % 21) - 10) / 10, 1) for i, s in enumerate(U[:len(U) - drop])}
    scores.update(override or {})
    return scores


def answer(form="object", drop=0, override=None):
    scores = score_dict(drop, override)
    fa = scores if form == "object" else json.dumps(scores)
    return json.dumps({"reasoning": "r", "bullet_ids": [], "final_answer": fa})


def scripted(outputs):
    """A call that replays outputs in order; an Exception instance is raised."""
    log = []

    def call(attempt_no, *_):
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


def refuses(fn):
    try:
        fn()
        return False
    except ValueError:
        return True


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
        check("6 manifest counts voids, schema 3", man["voided_decision_points"] == 1
              and man["schema_version"] == 3)
        summary = ap.void_summary(meta)

    with tempfile.TemporaryDirectory() as tmp:
        bad = pd.concat([ap.score_rows(dates[0], good), ap.score_rows(dates[1], good)], ignore_index=True)
        check("6 refuses a void date that carries scores", refuses(lambda: records.write_run(
            tmp, pnl, bad, "A0", llm_meta=[ap.meta_row(dates[0], good), ap.meta_row(dates[1], void)])))
    with tempfile.TemporaryDirectory() as tmp:
        partial = pd.concat([ap.score_rows(dates[0], good).iloc[:49], ap.score_rows(dates[1], void)], ignore_index=True)
        check("6 refuses a non-void date missing a member's score",
              refuses(lambda: records.write_run(tmp, pnl, partial, "A0", llm_meta=meta_rows)))

    # 7. summary
    row = summary.iloc[0]
    check("7 void_summary per arm", row["decision_points"] == 2 and row["voided"] == 1
          and row["retries_spent"] == 2 and row["points_needing_retry"] == 1
          and row["attempt_forms"] == {"json_string": 1, "object": 3}, row.to_dict())

    # 8. out-of-universe key with all 50 scored
    call, log = scripted([answer("object", override={"9999": 0.1})])
    o8 = ap.run_decision(call, U)
    check("8 out-of-universe key voids with its own kind", o8["voided"] and o8["void_reason"] == "out_of_universe_key"
          and o8["void_kinds"] == ["out_of_universe_key"] and o8["attempts"][0]["n_scored_in_range"] == 50
          and len(log) == 3)

    # 9. duplicated keys
    dup_object = answer("object").replace('"final_answer": {"', f'"final_answer": {{"{U[0]}": 0.5, "', 1)
    call, _ = scripted([dup_object])
    o9a = ap.run_decision(call, U)
    inner = json.dumps(score_dict())
    dup_string = json.dumps({"reasoning": "r", "bullet_ids": [], "final_answer": f'{{"{U[0]}": 0.5, ' + inner[1:]})
    call, _ = scripted([dup_string])
    o9b = ap.run_decision(call, U)
    call, _ = scripted([answer("object", override={" " + U[0]: 0.4})])
    o9c = ap.run_decision(call, U)
    call, _ = scripted([answer("object").replace('{"reasoning": "r"', '{"reasoning": "r", "reasoning": "r2"', 1)])
    o9d = ap.run_decision(call, U)
    check("9 duplicate key inside an object answer voids", o9a["voided"] and o9a["void_reason"] == "duplicate_key"
          and o9a["attempts"][0]["duplicate_keys"] == [U[0]])
    check("9 duplicate key inside a JSON-string answer voids", o9b["voided"] and o9b["void_reason"] == "duplicate_key")
    check("9 whitespace collision counts as duplicate", o9c["voided"] and o9c["void_reason"] == "duplicate_key")
    check("9 repeated reasoning key is recorded but usable", not o9d["voided"]
          and o9d["attempts"][0]["other_duplicate_keys"] == ["reasoning"])

    # 10. both non-compliance and incomplete
    call, _ = scripted([answer("object", drop=1, override={"9999": 0.1})])
    o10 = ap.run_decision(call, U)
    check("10 named reason out_of_universe_key, both kinds kept", o10["void_reason"] == "out_of_universe_key"
          and o10["void_kinds"] == ["out_of_universe_key", "incomplete"])

    # 11. learning calls
    refl, refl_log = scripted([RuntimeError("500"), "", json.dumps({"reasoning": "x", "bullet_tags": []})])
    seen = []

    def curate(n, reflection):
        seen.append(reflection)
        return {"response": json.dumps({"operations": []})}
    mat_ok = ap.process_maturity(dates[0], False, refl, curate)
    check("11 Reflector retried under the same rule, then Curator gets its reflection",
          mat_ok["reflector"]["status"] == "ok" and mat_ok["reflector"]["n_retries"] == 2
          and [a["failure"] for a in mat_ok["reflector"]["attempts"]] == ["call_error", "empty", None]
          and mat_ok["curator"]["status"] == "ok" and seen and json.loads(seen[0])["reasoning"] == "x")
    refl_bad, _ = scripted(["not json"])
    cur_log = []
    mat_fail = ap.process_maturity(dates[0], False, refl_bad, lambda n, r: cur_log.append(n) or {"response": "{}"})
    check("11 Reflector failing after retries: no Curator call", mat_fail["reflector"]["status"] == "failed"
          and mat_fail["reflector"]["n_attempts"] == 3 and mat_fail["curator"] is None and cur_log == [])
    cur_bad, cur_bad_log = scripted([RuntimeError("down")])
    refl_one, _ = scripted([json.dumps({"reasoning": "x"})])
    mat_cfail = ap.process_maturity(dates[0], False, refl_one, cur_bad)
    check("11 Curator failure recorded apart", mat_cfail["curator"]["status"] == "failed"
          and len(cur_bad_log) == 3)

    # 12. void at maturity, window slots
    never, never_log = scripted([json.dumps({})])
    mat_skip = ap.process_maturity(dates[0], True, never, never)
    check("12 void decision skipped at maturity, no call", mat_skip["maturity_status"] == "skipped_void"
          and never_log == [])
    check("12 no maturity before date index 11", ap.matured_index(10) is None and ap.matured_index(11) == 0)
    voided = [False, True] + [False] * 30
    kept, skipped = ap.window_indices(16, voided)
    check("12 A1 window leaves the void slot empty", kept == [2, 3, 4, 5] and skipped == [1])
    check("12 A1 window before any maturity is empty", ap.window_indices(10, voided) == ([], []))

    # 13. learning fields on disk
    a2_dates = pd.DatetimeIndex(["2026-05-01", "2026-05-04", "2026-05-05", "2026-05-06"])
    pnl2 = synthetic_panel(a2_dates)
    none_m = ap.process_maturity(None, False, never, never)
    maturities = [none_m, mat_skip, mat_fail, mat_ok]
    rows = [ap.meta_row(d, good, maturity=m) for d, m in zip(a2_dates, maturities)]
    sc = pd.concat([ap.score_rows(d, good) for d in a2_dates], ignore_index=True)
    with tempfile.TemporaryDirectory() as tmp:
        records.write_run(tmp, pnl2, sc, "A2", llm_meta=rows)
        _, meta2, man2 = records.read_run(tmp)
        ls = ap.learning_summary(meta2).iloc[0]
        check("13 learning_summary counts apart from generation",
              ls["maturities_due"] == 3 and ls["maturities_skipped_void"] == 1 and ls["reflector_ok"] == 1
              and ls["reflector_failed"] == 1 and ls["curator_not_run_after_reflector_failure"] == 1
              and ls["playbook_updates"] == 1 and ls["reflector_retries"] == 4, ls.to_dict())
        check("13 manifest counts skipped maturities and learning failures",
              man2["maturities_skipped_void"] == 1 and man2["reflector_failed"] == 1 and man2["curator_failed"] == 0)
    forged = [dict(r) for r in rows]
    forged[1]["reflector_status"] = "ok"
    with tempfile.TemporaryDirectory() as tmp:
        check("13 refuses a learning call on a skipped maturity",
              refuses(lambda: records.write_run(tmp, pnl2, sc, "A2", llm_meta=forged)))
    forged2 = [dict(r) for r in rows]
    forged2[2]["curator_status"] = "ok"
    with tempfile.TemporaryDirectory() as tmp:
        check("13 refuses a Curator after a failed Reflector",
              refuses(lambda: records.write_run(tmp, pnl2, sc, "A2", llm_meta=forged2)))
    a1_rows = [ap.meta_row(d, good, window=([a2_dates[0]], [a2_dates[1]])) for d in a2_dates]
    with tempfile.TemporaryDirectory() as tmp:
        records.write_run(tmp, pnl2, sc, "A1", llm_meta=a1_rows)
        _, meta3, man3 = records.read_run(tmp)
        ls1 = ap.learning_summary(meta3).iloc[0]
        check("13 A1 window skips counted", ls1["window_slots_skipped_void"] == 4
              and man3["window_slots_skipped_void"] == 4)
    print("ALL PASS")


if __name__ == "__main__":
    main()
