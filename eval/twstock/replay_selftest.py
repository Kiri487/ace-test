"""Offline self-check of the replay time structure. No LLM, no ACE role; stand-in callables.

    .venv/bin/python -m eval.twstock.replay_selftest            synthetic + both v9 calendars
    .venv/bin/python -m eval.twstock.replay_selftest --synthetic-only

Pass criteria, fixed before the first run; any failure exits with status 1.

C1 Every decision matures exactly once: each decision with a mature_date is harvested once,
   on that date; a decision whose mature_date lies past the calendar is never harvested and
   is listed as unmatured; nothing is left pending or harvested twice.
C2 mature_date is 11 trading sessions later, checked against an independent day-by-day walk
   over the calendar (not the positional shift under test). On the synthetic calendar also
   against hand-derived dates:
     2026-01-19 (Mon, a Wednesday holiday 01-21 inside)        -> 2026-02-04
     2026-02-09 (Mon, a five-day break 02-16..20 and 02-27)    -> 2026-03-04
     2026-02-13 (Fri, the break starts the next session)       -> 2026-03-10
     2026-03-16 (Mon, last maturity inside the calendar)       -> 2026-03-31
     2026-03-17 (Tue, calendar ends 03-31)                     -> no mature date
C3 A2 on decision date t holds only feedback from decisions at least 11 sessions before t and
   matured by t, checked on every recorded snapshot; the entry count equals the maturities
   processed so far with a successful Curator. An OutcomeView at t raises LookaheadError for a
   decision that matures after t and does not contain it.
C4 Each arm is called exactly once per decision date; every day's order is a permutation of
   the three arms; the same seed reproduces every order, a different seed changes at least one;
   all 6 orders occur and none exceeds 40% of dates; the chosen seed (replay.ARM_ORDER_SEED)
   gives chi-square p > 0.5 on 85 and on 75 dates.
C5 A void decision is skipped at maturity with no Reflector/Curator call; A1's window leaves
   the slot empty and the per-date empty-slot count equals the void A1 decisions among the
   last 5 maturity slots (cross-checked with arm_protocol.window_indices); a Reflector that
   fails after its retries leaves the playbook without that entry and no Curator call is made.
C6 The memories are independent: A0 always empty; A1's window holds A1's own scores; only A2
   decisions reach the Reflector and Curator; the memory objects are distinct.
C7 On the v9 calendars: post-cutoff 85 decision dates and pre-cutoff 75; A2 processes
   n - 11 maturities inside the window (74 / 64 before voids), and each arm's last 11
   decisions mature after the last decision date without touching memory.
"""

import argparse
import datetime as _dt
import json
import sys
from collections import Counter

import numpy as np
import pandas as pd

from . import arm_protocol as ap
from . import replay as rp

U = [str(1101 + i) for i in range(50)]
FAILED = []


def check(label, name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + ("" if cond else f"  {detail}"))
    if not cond:
        FAILED.append(f"{label}: {name}")


def synthetic_days():
    holidays = {pd.Timestamp("2026-01-21")} | set(pd.date_range("2026-02-16", "2026-02-20")) \
        | {pd.Timestamp("2026-02-27")}
    days = [d for d in pd.bdate_range("2026-01-05", "2026-03-31") if d not in holidays]
    return pd.DatetimeIndex(days)


def walk_mature(day, trading, n=rp.DELAY):
    """Independent reference: step one calendar day at a time, count trading days."""
    last = max(trading)
    d, count = pd.Timestamp(day), 0
    while count < n:
        d = d + pd.Timedelta(days=1)
        if d > last:
            return None
        if d in trading:
            count += 1
    return d


class Stand:
    """Stand-in generator / Reflector / Curator with call logs and fault injection."""

    def __init__(self, void_plan=(), failing_reflections=()):
        self.void_plan = set(void_plan)                    # (arm, decision_date)
        self.failing_reflections = set(failing_reflections)  # A2 decision dates
        self.gen_calls = Counter()
        self.snapshots = {}
        self.reflect_calls = []
        self.curate_calls = []

    def generate(self, arm, day, snapshot, attempt_no):
        self.gen_calls[(arm, day)] += 1
        self.snapshots[(arm, day)] = snapshot
        rng = np.random.default_rng([ord(arm[1]), int(day.value // 86_400_000_000_000)])
        scores = {s: round(float(x), 3) for s, x in zip(U, rng.uniform(-1, 1, len(U)))}
        if (arm, day) in self.void_plan:
            scores.pop(U[-1])
        return {"response": json.dumps({"reasoning": "stand-in", "bullet_ids": [], "final_answer": scores})}

    def reflect(self, decision, realized, attempt_no):
        self.reflect_calls.append((decision.arm, decision.decision_date, attempt_no))
        if decision.decision_date in self.failing_reflections:
            return {"response": "not json"}
        return {"response": json.dumps({"reasoning": f"matured {decision.decision_date.date()}",
                                        "mean_alpha": float(np.mean(list(realized.values())))})}

    def curate(self, snapshot, reflection, decision, attempt_no):
        self.curate_calls.append((decision.arm, decision.decision_date, len(snapshot)))
        return {"response": json.dumps({"operations": [
            {"type": "ADD", "content": f"lesson from {decision.decision_date.date()}"}]})}


def outcome_values(dates):
    rng = np.random.default_rng(7)
    return {d: {s: float(x) for s, x in zip(U, rng.normal(0, 0.05, len(U)))} for d in dates}


def run_case(label, trading_days, decision_dates, seed, void_plan, failing):
    cal = rp.TradingCalendar(trading_days)
    stand = Stand(void_plan, failing)
    store = rp.OutcomeStore(outcome_values(decision_dates), cal)
    res = rp.run_replay(cal, decision_dates, U, stand.generate, stand.reflect, stand.curate, store, seed)
    return cal, stand, store, res


def verify(label, trading_days, decision_dates, seed, void_plan, failing, hand_cases=None):
    print(f"\n== {label}: {len(decision_dates)} decision dates "
          f"{decision_dates[0].date()}..{decision_dates[-1].date()}, calendar to {trading_days[-1].date()}")
    cal, stand, store, res = run_case(label, trading_days, decision_dates, seed, void_plan, failing)
    trading = set(trading_days)
    n = len(decision_dates)
    last = decision_dates[-1]
    decs = res.decisions

    # C1
    harvest_count = Counter((h["arm"], h["decision_date"]) for h in res.harvests)
    harvest_day = {(h["arm"], h["decision_date"]): h["harvested_on"] for h in res.harvests}
    matured = [d for d in decs if d.mature_date is not None]
    check(label, "C1 each matured decision harvested exactly once, on its mature_date",
          all(harvest_count[(d.arm, d.decision_date)] == 1 and harvest_day[(d.arm, d.decision_date)] == d.mature_date
              for d in matured) and len(res.harvests) == len(matured))
    check(label, "C1 decisions past the calendar never harvested and listed as unmatured",
          all(harvest_count[(d.arm, d.decision_date)] == 0 for d in res.unmatured)
          and {(d.arm, d.decision_date) for d in res.unmatured} == {(d.arm, d.decision_date) for d in decs if d.mature_date is None})
    check(label, "C1 nothing left pending", not res.unharvested, f"{len(res.unharvested)} pending")

    # C2
    walk_ok = all(d.mature_date == walk_mature(d.decision_date, trading) for d in decs)
    check(label, "C2 mature_date equals the independent day-by-day walk (11 sessions)", walk_ok)
    between_ok = all(sum(1 for x in trading_days if d.decision_date < x < d.mature_date) == rp.DELAY - 1
                     for d in matured)
    check(label, "C2 exactly 10 trading days strictly between decision and mature date", between_ok)
    if hand_cases:
        got = {d.decision_date: d.mature_date for d in decs if d.arm == "A0"}
        for day, expected in hand_cases.items():
            check(label, f"C2 hand-derived {day} -> {expected}",
                  got[pd.Timestamp(day)] == (None if expected is None else pd.Timestamp(expected)),
                  f"got {got[pd.Timestamp(day)]}")

    # C3
    a2_voided = {d.decision_date for d in decs if d.arm == "A2" and d.voided}
    a2_ok, count_ok = True, True
    for g in (g for g in res.generations if g["arm"] == "A2"):
        t, j = g["decision_date"], g["decision_index"]
        limit = cal.position(t) - rp.DELAY
        if any(cal.position(s) > limit for s in g["memory_sources"]) or any(m > t for m in g["memory_matured_on"]):
            a2_ok = False
        expected = sum(1 for i in range(0, max(0, j - rp.DELAY + 1))
                       if decision_dates[i] not in a2_voided and decision_dates[i] not in failing)
        if len(g["memory_sources"]) != expected:
            count_ok = False
    check(label, "C3 A2 snapshot at t holds only feedback from decisions <= t-11 sessions, matured by t", a2_ok)
    check(label, "C3 A2 entry count = maturities processed with a successful Curator", count_ok)
    probe_day = decision_dates[min(n - 1, rp.DELAY + 3)]
    view = store.as_of(probe_day)
    later = decision_dates[cal.position(probe_day) - cal.position(decision_dates[0]) - rp.DELAY + 1] \
        if cal.position(probe_day) - cal.position(decision_dates[0]) - rp.DELAY + 1 < n else None
    raised = False
    if later is not None:
        try:
            view.realized(later)
        except rp.LookaheadError:
            raised = True
    check(label, "C3 OutcomeView at t raises LookaheadError for a decision maturing after t, and lacks it",
          later is not None and raised and later not in view.matured_decision_dates())

    # C4
    per_arm_day = Counter((g["arm"], g["decision_date"]) for g in res.generations)
    check(label, "C4 each arm called exactly once per decision date",
          all(per_arm_day[(a, d)] == 1 for a in rp.ARMS for d in decision_dates) and len(res.generations) == 3 * n)
    check(label, "C4 every order is a permutation of the three arms",
          all(sorted(o) == sorted(rp.ARMS) for o in res.orders.values()) and len(res.orders) == n)
    _, _, _, res_same = run_case(label, trading_days, decision_dates, seed, void_plan, failing)
    _, _, _, res_other = run_case(label, trading_days, decision_dates, seed + 1, void_plan, failing)
    check(label, "C4 same seed reproduces every order", res_same.orders == res.orders)
    check(label, "C4 a different seed changes at least one order", res_other.orders != res.orders)
    freq = Counter(res.orders.values())
    check(label, "C4 all 6 orders occur, none above 40% of dates",
          len(freq) == 6 and max(freq.values()) <= 0.4 * n, dict(Counter({"-".join(k): v for k, v in freq.items()})))

    # C5
    a2_harv = {h["decision_date"]: h for h in res.harvests if h["arm"] == "A2" and h["status"] != "after_last_decision"}
    skipped_ok = all(a2_harv[d]["status"] == "skipped_void" and a2_harv[d]["reflector_status"] == "not_run"
                     for d in a2_voided if d in a2_harv)
    reflected_dates = {c[1] for c in stand.reflect_calls}
    check(label, "C5 void A2 decisions skipped at maturity, no Reflector call",
          skipped_ok and not (reflected_dates & a2_voided))
    fail_ok = all(a2_harv[d]["reflector_status"] == "failed" and a2_harv[d]["curator_status"] == "not_run"
                  for d in failing if d in a2_harv)
    curated_dates = {c[1] for c in stand.curate_calls}
    check(label, "C5 failed Reflector: no Curator call, no playbook entry",
          fail_ok and not (curated_dates & set(failing))
          and not (set(failing) & {e.source_decision_date for e in res.memories["A2"].snapshot()}))
    in_window = sum(1 for i in range(n) if i + rp.DELAY <= n - 1)
    check(label, "C5 Reflector reached exactly the non-void A2 maturities inside the window",
          len(reflected_dates) == in_window - len([d for d in a2_voided if decision_dates.get_loc(d) + rp.DELAY <= n - 1]))
    a1_voided_flags = [(("A1", d) in void_plan) for d in decision_dates]
    win_ok = True
    for g in (g for g in res.generations if g["arm"] == "A1"):
        j = g["decision_index"]
        kept, skipped = ap.window_indices(j, a1_voided_flags)
        if [decision_dates[i] for i in kept] != list(g["memory_sources"]) \
                or [decision_dates[i] for i in skipped] != list(g["window_skipped_void"]):
            win_ok = False
    check(label, "C5 A1 window slots match arm_protocol.window_indices, void slots left empty", win_ok)
    a1_rows = res.meta_rows["A1"]
    expected_skips = sum(len(ap.window_indices(j, a1_voided_flags)[1]) for j in range(n))
    check(label, "C5 A1 empty-slot count recorded per date adds up",
          sum(r["window_skipped_void"] for r in a1_rows) == expected_skips)
    a2_rows = res.meta_rows["A2"]
    check(label, "C5 A2 meta rows count skipped maturities",
          sum(1 for r in a2_rows if r["maturity_status"] == "skipped_void")
          == len([d for d in a2_voided if decision_dates.get_loc(d) + rp.DELAY <= n - 1]))

    # C6
    check(label, "C6 A0 memory empty at every decision", all(g["memory_sources"] == () for g in res.generations if g["arm"] == "A0")
          and res.memories["A0"].snapshot() == ())
    own = {(d.arm, d.decision_date): d.scores for d in decs}
    a1_scores_ok = all(slot.scores == own[("A1", slot.decision_date)]
                       for key, snap in stand.snapshots.items() if key[0] == "A1" for slot in snap)
    check(label, "C6 A1 window holds A1's own scores", a1_scores_ok)
    check(label, "C6 only A2 decisions reach Reflector and Curator",
          all(c[0] == "A2" for c in stand.reflect_calls) and all(c[0] == "A2" for c in stand.curate_calls))
    mems = res.memories
    check(label, "C6 memory objects are distinct", len({id(mems[a]) for a in rp.ARMS}) == 3)

    # C7-style counts, reported for every calendar
    a2_processed = sum(1 for h in res.harvests if h["arm"] == "A2" and h["status"] == "processed")
    after_last = Counter(h["arm"] for h in res.harvests if h["status"] == "after_last_decision")
    return {"n": n, "a2_processed": a2_processed, "a2_voided_in_window": len([d for d in a2_voided if decision_dates.get_loc(d) + rp.DELAY <= n - 1]),
            "after_last": dict(after_last), "unmatured": len(res.unmatured), "orders": Counter(res.orders.values())}


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--synthetic-only", action="store_true")
    args = ap_.parse_args()

    days = synthetic_days()
    dd = days                                   # decide every trading day, so the tail is exercised
    void_plan = {("A0", dd[2]), ("A1", dd[3]), ("A2", dd[5]), ("A2", dd[6]), ("A1", dd[20])}
    failing = {dd[8]}
    hand = {"2026-01-19": "2026-02-04", "2026-02-09": "2026-03-04", "2026-02-13": "2026-03-10",
            "2026-03-16": "2026-03-31", "2026-03-17": None}
    s = verify("synthetic", days, dd, seed=rp.ARM_ORDER_SEED, void_plan=void_plan, failing=failing, hand_cases=hand)
    check("synthetic", "C1 tail: the last 11 decisions have no mature date", s["unmatured"] == 3 * rp.DELAY,
          f"unmatured {s['unmatured']}")

    mf = rp.manifest_fields()
    check("seed", f"C4 arm-order seed {mf['arm_order_seed']} gives chi-square p > 0.5 on 85 and 75 dates",
          all(v['chi2_p'] > 0.5 for v in mf['arm_order_uniformity'].values()), mf['arm_order_uniformity'])
    print(f"  info: arm-order seed {mf['arm_order_seed']}, " +
          ", ".join(f"{k} p={v['chi2_p']:.4f}" for k, v in mf['arm_order_uniformity'].items()))

    if not args.synthetic_only:
        from . import panel
        trading = panel.trading_days()
        for label, (a, b) in rp.CONDITIONS.items():
            dates = panel.decision_dates(a, b)
            plan = {("A1", dates[4]), ("A2", dates[7]), ("A0", dates[30])}
            fail = {dates[15]}
            cal_days = trading[trading >= dates[0]]
            r = verify(f"v9 {label}", cal_days, dates, seed=rp.ARM_ORDER_SEED, void_plan=plan, failing=fail)
            expected_n = {"post": 85, "pre": 75}[label]
            check(label, f"C7 {expected_n} decision dates", r["n"] == expected_n, r["n"])
            check(label, f"C7 A2 processes n-11 = {expected_n - 11} maturities less voids and failed reflections",
                  r["a2_processed"] + r["a2_voided_in_window"] == expected_n - 11,
                  r)
            check(label, "C7 each arm's last 11 decisions mature after the last decision date, no memory update",
                  r["after_last"] == {a: rp.DELAY for a in rp.ARMS} and r["unmatured"] == 0, r)
            print(f"  info: A2 processed {r['a2_processed']} (+{r['a2_voided_in_window']} skipped void), "
                  f"orders {dict((('-'.join(k)), v) for k, v in r['orders'].items())}")

    print("\n" + ("ALL PASS" if not FAILED else f"{len(FAILED)} FAILED:\n  " + "\n  ".join(FAILED)))
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
