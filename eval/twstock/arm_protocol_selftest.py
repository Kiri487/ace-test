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
14. cost (schema 4): meta_row sums usage.cost per date, generation and learning apart; write_run
    writes one llm_calls row per attempt with its cost and provider routing, a call with no answer
    carries no cost; the manifest counts the calls and totals cost by role
15. A1 reflection step (schema 5, added 2026-09-15 before its first run): an empty window makes no
    call; a reflection follows rule 5's retries, keeps its text and completion tokens, fails with
    a1_reflection_missing when "reflection" is absent, and (revised 2026-09-15, second ruling: no cap)
    a long reflection is kept whole with no cap field; the record stores status per date, the reflection's cost as aux_cost_usd and its
    attempts as llm_calls rows with role a1_reflector, and refuses a status that disagrees with the
    window; an LLM run without run_kind is refused; a pilot manifest says usable_as_result false and
    require_result_run refuses it
16. pilot lock and disclosure rate (schema 6, added 2026-09-15 before its first run): learning_summary and
    the manifest give the A1 reflection failure rate (1 failed of 2 attempted = 0.5, over 0.05) and the A2
    Reflector rate over processed maturities; read_run opens a pilot only for "pilot_gate", an offline run
    only for "offline_check", and refuses an unknown purpose; write_run refuses a pilot without a
    pilot_window, a pilot_window on a non-pilot, an unknown window, and a main run whose dates are not
    exactly one replay.CONDITIONS window (this last check reads the trading calendar from the snapshot)
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
OFFLINE = {"run_kind": "offline_check"}
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
        records.write_run(tmp, pnl, scores, "A0", extra=OFFLINE, llm_meta=meta_rows)
        dec, meta, man = records.read_run(tmp, "offline_check")
        m = meta.set_index("decision_date")
        check("6 complete date: 50 scored, not void, form recorded",
              m.loc[dates[0], "n_scored"] == 50 and not m.loc[dates[0], "voided"]
              and m.loc[dates[0], "final_answer_form"] == "json_string")
        check("6 void date: no score, void, reason and 3 attempts kept",
              m.loc[dates[1], "n_scored"] == 0 and bool(m.loc[dates[1], "voided"])
              and m.loc[dates[1], "void_reason"] == "incomplete"
              and len(json.loads(m.loc[dates[1], "attempts_json"])) == 3
              and dec.loc[dec["decision_date"] == dates[1], "score"].isna().all())
        check("6 manifest counts voids, schema 6", man["voided_decision_points"] == 1
              and man["schema_version"] == 6)
        summary = ap.void_summary(meta)

    with tempfile.TemporaryDirectory() as tmp:
        bad = pd.concat([ap.score_rows(dates[0], good), ap.score_rows(dates[1], good)], ignore_index=True)
        check("6 refuses a void date that carries scores", refuses(lambda: records.write_run(
            tmp, pnl, bad, "A0", extra=OFFLINE, llm_meta=[ap.meta_row(dates[0], good), ap.meta_row(dates[1], void)])))
    with tempfile.TemporaryDirectory() as tmp:
        partial = pd.concat([ap.score_rows(dates[0], good).iloc[:49], ap.score_rows(dates[1], void)], ignore_index=True)
        check("6 refuses a non-void date missing a member's score",
              refuses(lambda: records.write_run(tmp, pnl, partial, "A0", extra=OFFLINE, llm_meta=meta_rows)))

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
        records.write_run(tmp, pnl2, sc, "A2", extra=OFFLINE, llm_meta=rows)
        _, meta2, man2 = records.read_run(tmp, "offline_check")
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
              refuses(lambda: records.write_run(tmp, pnl2, sc, "A2", extra=OFFLINE, llm_meta=forged)))
    forged2 = [dict(r) for r in rows]
    forged2[2]["curator_status"] = "ok"
    with tempfile.TemporaryDirectory() as tmp:
        check("13 refuses a Curator after a failed Reflector",
              refuses(lambda: records.write_run(tmp, pnl2, sc, "A2", extra=OFFLINE, llm_meta=forged2)))
    a1_rows = [ap.meta_row(d, good, window=([a2_dates[0]], [a2_dates[1]])) for d in a2_dates]
    with tempfile.TemporaryDirectory() as tmp:
        records.write_run(tmp, pnl2, sc, "A1", extra=OFFLINE, llm_meta=a1_rows)
        _, meta3, man3 = records.read_run(tmp, "offline_check")
        ls1 = ap.learning_summary(meta3).iloc[0]
        check("13 A1 window skips counted", ls1["window_slots_skipped_void"] == 4
              and man3["window_slots_skipped_void"] == 4)

    # 14. cost of every call (schema 4)
    def priced(outputs, cost):
        inner, _ = scripted(outputs)

        def call(n, *rest):
            out = inner(n, *rest)
            return {**out, "call_id": f"call-a{n}",
                    "provider": {"generation_id": f"gen-{n}-{cost}", "finalProvider": "fireworks", "usage": {"cost": cost}}}
        return call
    og = ap.run_decision(priced([RuntimeError("HTTP 500"), answer("object")], 0.01), U)
    mat_c = ap.process_maturity(dates[0], False, priced([json.dumps({"reasoning": "x"})], 0.002),
                                lambda n, r: priced([json.dumps({"operations": []})], 0.003)(n))
    cost_rows = [ap.meta_row(dates[0], og, maturity=mat_c),
                 ap.meta_row(dates[1], good, maturity=ap.process_maturity(None, False, None, None))]
    sc14 = pd.concat([ap.score_rows(dates[0], og), ap.score_rows(dates[1], good)], ignore_index=True)
    with tempfile.TemporaryDirectory() as tmp:
        records.write_run(tmp, pnl, sc14, "A2", extra=OFFLINE, llm_meta=cost_rows)
        _, meta4, man4 = records.read_run(tmp, "offline_check")
        calls = pd.read_parquet(f"{tmp}/llm_calls.parquet")
    m4 = meta4.set_index("decision_date")
    check("14 per-date cost: generation and learning apart; no answer, no cost",
          abs(m4.loc[dates[0], "cost_usd"] - 0.01) < 1e-12 and abs(m4.loc[dates[0], "aux_cost_usd"] - 0.005) < 1e-12
          and pd.isna(m4.loc[dates[1], "cost_usd"]) and pd.isna(m4.loc[dates[1], "aux_cost_usd"]), m4[["cost_usd", "aux_cost_usd"]])
    g0 = calls[(calls["decision_date"] == dates[0]) & (calls["role"] == "generator")].sort_values("attempt")
    aux4 = calls[calls["role"].isin(["reflector", "curator"])]
    check("14 llm_calls: one row per attempt, cost per row, failed call has none",
          len(calls) == 5 and list(g0["ok_call"]) == [False, True] and pd.isna(g0["cost_usd"].iloc[0])
          and "HTTP 500" in g0["error"].iloc[0] and g0["cost_usd"].iloc[1] == 0.01
          and sorted(aux4["role"]) == ["curator", "reflector"] and set(aux4["matured_decision_date"]) == {"2026-04-27"}
          and set(aux4["finalProvider"]) == {"fireworks"}, calls.to_dict(orient="records"))
    check("14 manifest: call count and cost by role", man4["llm_calls"] == 5
          and man4["cost_usd_by_role"] == {"curator": 0.003, "generator": 0.01, "reflector": 0.002}, man4.get("cost_usd_by_role"))

    # 15. A1 reflection step (A1 is two calls), run kinds
    ref_none = ap.process_a1_reflection((), None)
    ref_ok = ap.process_a1_reflection(("slot",), priced([RuntimeError("HTTP 500"), json.dumps({"reflection": "look again"})], 0.004))
    ref_bad = ap.process_a1_reflection(("slot",), priced([json.dumps({"reasoning": "no reflection key"})], 0.001))
    ref_long = ap.process_a1_reflection(("slot",), lambda n: {"response": json.dumps({"reflection": "x" * 20000}),
                                                              "completion_tokens": 9999})
    check("15 empty window: no reflection call", ref_none["status"] == "not_run" and ref_none["attempts"] == []
          and ref_none["reflection"] is None)
    check("15 reflection retried under rule 5, text and tokens kept", ref_ok["status"] == "ok" and ref_ok["n_attempts"] == 2
          and ref_ok["reflection"] == "look again" and ref_ok["completion_tokens"] == 10)
    check("15 reflection without text fails after retries, no text", ref_bad["status"] == "failed" and ref_bad["n_attempts"] == 3
          and ref_bad["reflection"] is None and ref_bad["attempts"][-1]["failure"] == "a1_reflection_missing")
    check("15 no length cap: a long reflection is kept whole, its tokens recorded, no cap field",
          ref_long["status"] == "ok" and len(ref_long["reflection"]) == 20000 and ref_long["completion_tokens"] == 9999
          and "over_cap" not in ref_long)
    rows15 = [ap.meta_row(dates[0], good, window=([], []), a1_reflection=ref_none),
              ap.meta_row(dates[1], good, window=([dates[0]], []), a1_reflection=ref_ok)]
    sc15 = pd.concat([ap.score_rows(dates[0], good), ap.score_rows(dates[1], good)], ignore_index=True)
    with tempfile.TemporaryDirectory() as tmp:
        records.write_run(tmp, pnl, sc15, "A1", extra={"run_kind": "offline_check"}, llm_meta=rows15)
        _, meta5, man5 = records.read_run(tmp, "offline_check")
        calls5 = pd.read_parquet(f"{tmp}/llm_calls.parquet")
    m5 = meta5.set_index("decision_date")
    check("15 A1 record: status per date, reflection cost as aux, llm_calls role a1_reflector",
          list(m5["a1_reflection_status"]) == ["not_run", "ok"] and abs(m5.loc[dates[1], "aux_cost_usd"] - 0.004) < 1e-12
          and int(m5.loc[dates[1], "a1_reflection_tokens"]) == 10
          and sorted(calls5.loc[calls5["role"] == ap.A1_REFLECTION_ROLE, "attempt"]) == [1, 2]
          and man5["cost_usd_by_role"].get(ap.A1_REFLECTION_ROLE) == 0.004
          and {k: man5["a1_reflections"][k] for k in ("ok", "failed", "not_run", "failure_rate", "over_disclosure_rate")}
          == {"ok": 1, "failed": 0, "not_run": 1, "failure_rate": 0.0, "over_disclosure_rate": False}, man5.get("a1_reflections"))
    forged15 = [dict(r) for r in rows15]
    forged15[1]["a1_reflection_status"] = "not_run"
    with tempfile.TemporaryDirectory() as tmp:
        check("15 refuses a reflection status that disagrees with the window",
              refuses(lambda: records.write_run(tmp, pnl, sc15, "A1", extra={"run_kind": "offline_check"}, llm_meta=forged15)))
    with tempfile.TemporaryDirectory() as tmp:
        check("15 refuses an LLM run without run_kind", refuses(lambda: records.write_run(tmp, pnl, sc15, "A1", llm_meta=rows15)))
    with tempfile.TemporaryDirectory() as tmp:
        records.write_run(tmp, pnl, sc15, "A0", extra={"run_kind": "pilot", "pilot": ap.PILOT_A0, "pilot_window": "post_85"},
                          llm_meta=[ap.meta_row(d, good) for d in dates])
        _, _, man_p = records.read_run(tmp, "pilot_gate")
        pilot_refused = [refuses(lambda: records.read_run(tmp, p)) for p in ("result", "offline_check")]
    check("15 pilot manifest is marked and refused as a result", man_p["run_kind"] == "pilot"
          and man_p["usable_as_result"] is False and refuses(lambda: records.require_result_run(man_p))
          and records.require_result_run({"run_kind": "main", "usable_as_result": True}))

    # 16. pilot lock, reflection failure disclosure rate (schema 6)
    rows16 = [ap.meta_row(dates[0], good, window=([dates[0]], []), a1_reflection=ref_ok),
              ap.meta_row(dates[1], good, window=([dates[0]], []), a1_reflection=ref_bad)]
    with tempfile.TemporaryDirectory() as tmp:
        records.write_run(tmp, pnl, sc15, "A1", extra=OFFLINE, llm_meta=rows16)
        _, meta6, man6 = records.read_run(tmp, "offline_check")
        offline_refused = [refuses(lambda: records.read_run(tmp, p)) for p in ("result", "pilot_gate")]
    ls6 = ap.learning_summary(meta6).iloc[0]
    check("16 A1 reflection failure rate against the 5% disclosure rate, per arm, in summary and manifest",
          ls6["a1_reflection_failure_rate"] == 0.5 and bool(ls6["a1_reflection_failure_over_disclosure_rate"])
          and man6["a1_reflections"]["failure_rate"] == 0.5 and man6["a1_reflections"]["over_disclosure_rate"] is True
          and man6["a1_reflections"]["disclosure_rate"] == ap.REFLECTION_FAILURE_DISCLOSURE_RATE == 0.05, ls6.to_dict())
    check("16 A2 Reflector failure rate over processed maturities", abs(ls["reflector_failure_rate"] - 0.5) < 1e-12
          and bool(ls["reflector_failure_over_disclosure_rate"]), ls.to_dict())
    check("16 read_run purposes: a pilot opens only for the gate, an offline run never as a result or for the gate",
          all(pilot_refused) and all(offline_refused) and refuses(lambda: records.read_run(".", "anything")), (pilot_refused, offline_refused))
    with tempfile.TemporaryDirectory() as tmp:
        lock = [refuses(lambda: records.write_run(tmp, pnl, sc15, "A0", extra={"run_kind": "pilot"}, llm_meta=rows15)),
                refuses(lambda: records.write_run(tmp, pnl, sc15, "A0", extra={"run_kind": "offline_check", "pilot_window": "long_2025_05"},
                                                  llm_meta=rows15)),
                refuses(lambda: records.write_run(tmp, pnl, sc15, "A0", extra={"run_kind": "pilot", "pilot_window": "other"},
                                                  llm_meta=rows15)),
                refuses(lambda: records.write_run(tmp, pnl, sc15, "A0", extra={"run_kind": "main"}, llm_meta=rows15))]
    check("16 write refuses a pilot without its window, a window on a non-pilot, an unknown window, and a main run "
          "that is not exactly one CONDITIONS window", all(lock), lock)
    print("ALL PASS")


if __name__ == "__main__":
    main()
