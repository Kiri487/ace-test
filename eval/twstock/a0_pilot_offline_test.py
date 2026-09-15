"""Offline test of the A0 pilot runner (eval/twstock/a0_pilot.py). No LLM call, no network.

    .venv/bin/python -m eval.twstock.a0_pilot_offline_test

The production path runs as in replay_offline: a0_pilot.run_pilot, roles.LiveRoles, the ACE Generator,
llm.timed_llm_call, the OpenAI SDK, utils._ClineUnwrapTransport with its provider log and the runner's request
hook. Only the wire is replay_offline.FakeLLM. MORNING.md is redirected into the test directory.

Pass criteria, fixed 2026-09-15 before the first run; every FAIL is printed, exit status 1:
T1 full flow, all four runs; planned faults post85 index 3 incomplete once, post85 index 5 incomplete x3:
 a status "complete"; date files per run == 17, 17, 85, 324; each run's records manifest has run_kind "pilot",
   usable_as_result false, its pilot_window, gate_sha256 == GATE_SHA256, data_snapshot id finlab_20260915_200558,
   and sampling == arm_protocol.SAMPLING
 b generator requests per (run, date) == that date's recorded attempts; post85 index 3 has 2 attempts and is not void;
   post85 index 5 is void with 3 attempts; no other date is void
 c every abln prompt's context contains "共 0 則" and no headline line; every ablr prompt is identical to the post85
   prompt of the same date
 d first_response_raw.json exists; its request params have model cline-pass/deepseek-v4-flash and temperature 0.0
 e gate_verdict.json has C1 and both windows with a verdict; MORNING.md has sections 1 to 7;
   records.read_run(purpose="result") refuses every pilot records directory
T2 interruption: runs abln, ablr, post85; session 1 returns "interrupted" after 40 dates; session 2 returns "complete";
   no (run, date) is requested in both sessions; date files total 119
T3 S1: run post85 only, indexes 0-9 HTTP 500 on every attempt: "stopped", STOP.json code S1, 10 date files,
   30 requests; a second start without after_stop raises SystemExit before any request
T4 S2: run post85 only, indexes 0, 2, 4 incomplete x3: "stopped", code S2, 10 date files
T5 S3: run post85 only, call_cap 20: "stopped", code S3, 20 requests
T6 S4: the request hook records a body naming another model, and check_stop then returns S4; a preflight with
   roles.MODEL changed raises SystemExit before any request
T7 gate guard: with GATE_SHA256 changed, run_pilot raises SystemExit before any request
"""

import datetime as _dt
import glob
import json
import sys
from pathlib import Path

from . import replay_offline as ro      # sets the offline environment on import

import utils

from . import a0_pilot as P
from . import arm_protocol as ap
from . import records, roles

CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok)))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"  {str(detail)[:800]}"), flush=True)


def setup():
    tok = ro.load_tokenizer()
    calls = [json.loads(x) for x in (ro.TRIAL / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    reasonings = [json.loads(c["response"])["reasoning"] for c in calls if c["ok_call"]]
    envelope = json.load(open(glob.glob(str(ro.RESULTS / "format_trial/identity_20260914/identity_out/*_identity.json"))[0],
                              encoding="utf-8"))["body"]
    fake = ro.FakeLLM(tok, envelope, reasonings, {"reflector": 1000, "curator": 1000, "growth": 180, "a1_reflection": 1300})
    ro._Wire.responder = fake
    utils._ClineUnwrapTransport = ro.OfflineClineTransport
    return fake


def fresh(fake, root, name, faults=None):
    fake.log.clear()
    fake.faults = {}
    for (label, idx, kind) in (faults or []):
        run = next(r for r in P.RUNS if r.label == label)
        day = P.run_dates(run)[idx]
        fake.faults[(label, "A0", day, "gen")] = kind
    out = root / name
    P.MORNING = out / "MORNING.md"
    return out


def gen_requests(fake):
    reqs = {}
    for e in fake.log:
        if e["role"] == "generator":
            reqs.setdefault((e["label"], e["date"]), []).append(e)
    return reqs


def main():
    fake = setup()
    root = records.REPO_ROOT / "results" / "a0_pilot_offline" / f"test_{_dt.datetime.now():%Y%m%d_%H%M%S}"
    label_of = {r.name: r.label for r in P.RUNS}
    POST = next(r for r in P.RUNS if r.label == "post85")

    # T1
    out = fresh(fake, root, "T1", [("post85", 3, {1: "incomplete"}), ("post85", 5, {"all": "incomplete"})])
    status = P.run_pilot(out)
    counts = {r.name: len(list((out / r.name / "dates").glob("*.json"))) for r in P.RUNS}
    mans = {r.name: json.loads((out / r.name / "records" / "manifest.json").read_text(encoding="utf-8")) for r in P.RUNS}
    check("T1a complete, date files per run, pilot manifests", status == "complete"
          and counts == {"news_ablation_nonews": 17, "news_ablation_repeat": 17, "post_85": 85, "long_2025_05": 324}
          and all(m["run_kind"] == "pilot" and m["usable_as_result"] is False and m["pilot_window"] == r.pilot_window
                  and m["gate_sha256"] == P.GATE_SHA256 and m["data_snapshot"]["snapshot_id"] == P.EXPECTED_SNAPSHOT
                  and m["sampling"] == json.loads(json.dumps(ap.SAMPLING)) for r in P.RUNS for m in [mans[r.name]]),
          (status, counts))
    reqs = gen_requests(fake)
    bad, voids = [], []
    for r in P.RUNS:
        for f in (out / r.name / "dates").glob("*.json"):
            rec = json.loads(f.read_text(encoding="utf-8"))
            day = ro.pd.Timestamp(rec["decision_date"])
            if len(reqs.get((r.label, day), [])) != rec["outcome"]["n_attempts"]:
                bad.append((r.name, rec["decision_date"]))
            if rec["voided"]:
                voids.append((r.name, rec["decision_date"], rec["outcome"]["n_attempts"]))
    d3, d5 = [str(P.run_dates(POST)[i].date()) for i in (3, 5)]
    rec3 = json.loads((out / "post_85" / "dates" / f"{d3}.json").read_text(encoding="utf-8"))
    check("T1b requests == attempts; planned retry and void only", not bad and rec3["outcome"]["n_attempts"] == 2
          and not rec3["voided"] and voids == [("post_85", d5, 3)], (bad[:5], voids))
    abln = [e for e in fake.log if e["label"] == "abln"]
    ctx_ok = all("共 0 則" in ro.split(ro.GEN_RE, e["prompt"])["context"]
                 and not any(ln.startswith("[20") for ln in ro.split(ro.GEN_RE, e["prompt"])["context"].split("\n"))
                 for e in abln)
    post_prompt = {e["date"]: e["prompt"] for e in fake.log if e["label"] == "post85" and e["attempt"] == 1}
    ablr_same = all(e["prompt"] == post_prompt[e["date"]] for e in fake.log if e["label"] == "ablr")
    check("T1c no-news contexts empty of headlines; repeat prompts identical to post85", ctx_ok and len(abln) == 17 and ablr_same)
    fr = json.loads((out / "first_response_raw.json").read_text(encoding="utf-8"))
    rp_ = fr["request_params_without_messages"]
    check("T1d first raw response with request params", rp_.get("model") == P.EXPECTED_MODEL and rp_.get("temperature") == 0.0
          and fr.get("response_body_raw"), rp_)
    v = json.loads((out / "gate_verdict.json").read_text(encoding="utf-8"))
    morning = P.MORNING.read_text(encoding="utf-8")
    refused = []
    for r in P.RUNS:
        try:
            records.read_run(out / r.name / "records", "result")
            refused.append(False)
        except ValueError:
            refused.append(True)
    check("T1e verdict, MORNING sections, pilot refused as result",
          "C1_news_dependence_post" in v and all(v["windows"][w].get("verdict") for w in ("post_85", "long_2025_05"))
          and all(f"## {i}." in morning for i in range(1, 8)) and all(refused), (list(v.get("windows", {})), refused))

    # T2
    three = tuple(r for r in P.RUNS if r.label in ("abln", "ablr", "post85"))
    out = fresh(fake, root, "T2")
    s1 = P.run_pilot(out, runs=three, stop_after_dates=40)
    first = set((e["label"], e["date"]) for e in fake.log if e["role"] == "generator")
    fake.log.clear()
    s2 = P.run_pilot(out, runs=three)
    second = set((e["label"], e["date"]) for e in fake.log if e["role"] == "generator")
    total = sum(len(list((out / r.name / "dates").glob("*.json"))) for r in three)
    check("T2 interrupted then resumed: no date requested twice, all 119 dates", s1 == "interrupted" and s2 == "complete"
          and not (first & second) and len(first) == 40 and total == 119, (s1, s2, len(first), len(second), total))

    # T3
    out = fresh(fake, root, "T3", [("post85", i, {"all": "http500"}) for i in range(10)])
    s = P.run_pilot(out, runs=(POST,))
    stop = json.loads((out / "STOP.json").read_text(encoding="utf-8"))
    n_req = len(fake.log)
    n_files = len(list((out / "post_85" / "dates").glob("*.json")))
    fake.log.clear()
    try:
        P.run_pilot(out, runs=(POST,))
        refused = False
    except SystemExit:
        refused = True
    check("T3 S1 after 10 consecutive failed dates; restart refused", s == "stopped" and stop["code"] == "S1"
          and n_files == 10 and n_req == 30 and refused and not fake.log, (s, stop, n_files, n_req, refused))

    # T4
    out = fresh(fake, root, "T4", [("post85", i, {"all": "incomplete"}) for i in (0, 2, 4)])
    s = P.run_pilot(out, runs=(POST,))
    stop = json.loads((out / "STOP.json").read_text(encoding="utf-8"))
    check("T4 S2 at 3 failed of 10", s == "stopped" and stop["code"] == "S2"
          and len(list((out / "post_85" / "dates").glob("*.json"))) == 10, stop)

    # T5
    out = fresh(fake, root, "T5")
    s = P.run_pilot(out, runs=(POST,), call_cap=20)
    stop = json.loads((out / "STOP.json").read_text(encoding="utf-8"))
    check("T5 S3 at the request cap", s == "stopped" and stop["code"] == "S3" and stop["requests"] == 20, stop)

    # T6
    out = fresh(fake, root, "T6")
    out.mkdir(parents=True, exist_ok=True)
    w = P.Wire(out)
    req = ro.httpx.Request("POST", "http://offline.invalid/api/v1/chat/completions",
                           content=json.dumps({"model": "someone/else", "messages": []}).encode())
    w._orig = lambda *a: None
    w.hook(req, 200, {}, b"{}")
    code = P.check_stop(out, w, 730, [])[0]
    saved = roles.MODEL
    roles.MODEL = "someone/else"
    try:
        P.run_pilot(root / "T6b", runs=(POST,))
        pre = False
    except SystemExit:
        pre = True
    finally:
        roles.MODEL = saved
    check("T6 S4 on a request naming another model; preflight refuses a wrong model", code == "S4" and pre and not fake.log,
          (code, pre, len(fake.log)))

    # T7
    fresh(fake, root, "T7")
    saved = P.GATE_SHA256
    P.GATE_SHA256 = "0" * 64
    try:
        P.run_pilot(root / "T7", runs=(POST,))
        g = False
    except SystemExit:
        g = True
    finally:
        P.GATE_SHA256 = saved
    check("T7 gate sha256 mismatch refuses before any request", g and not fake.log)

    failed = [n for n, ok in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} PASS" + ("" if not failed else "; FAILED: " + ", ".join(failed)))
    print("written to", root)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
