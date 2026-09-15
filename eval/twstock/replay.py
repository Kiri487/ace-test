"""Causal replay time structure for the three arms (v9.3 §5.1, §5.2.0, §10.1). M5a, part 1.

Scheduling only: no LLM and no ACE role is called here. Generation, reflection and
curation are injected callables; part 2 (roles.py) plugs in the real Generator / Reflector /
Curator and an ACE text playbook through `memories`, and the A2 memory decides in commit()
what a finished maturity changes.

Every trading day t, in this order:
  1. harvest  decisions whose mature_date == t release their h=10 outcome through an
              OutcomeView bounded at t.
              A2: Reflector -> Curator on its own playbook, via arm_protocol.process_maturity
                  (a void decision is skipped with no call; a failed Reflector means no Curator).
              A1: the matured decision enters its rolling window of the last 5 maturity
                  slots; a void decision occupies its slot empty, nothing older is pulled in.
              A0: no memory.
              Maturities after the last decision date are recorded but update no memory:
              no later decision could use them (this is why A2 learns n - 11 times).
  2. decide   on decision dates only: the three arms in an order drawn from
              default_rng([seed, date index]), each from its own memory; every decision
              goes through arm_protocol.run_decision and into pending with mature_date =
              the trading day 11 sessions after t (open(t+1) plus 10 sessions, v9 §5.1),
              or None when the calendar ends first.

Structural guarantees, enforced here rather than left to callers:
- The memories are separate objects; generation receives an immutable snapshot of its own
  arm's memory only.
- OutcomeStore.as_of(t) builds a view holding only outcomes matured by t; later outcomes are
  not in the view object at all, and asking for one raises LookaheadError.
- mature_date is counted on the trading calendar, never in calendar days.
- Before generating, A2's playbook is asserted to hold only feedback from decisions at
  least 11 sessions before t, and A1's window likewise.
"""

import json
from collections import deque
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
import pandas as pd

from . import arm_protocol as ap

ARMS = ("A0", "A1", "A2")
DELAY = ap.DELAY          # 11 sessions
WINDOW_K = ap.WINDOW_K    # A1 keeps the last 5 maturity slots
# v9.3 §5.0, phase one. The single source for both windows: budget_estimate reads it from here.
# Pre-cutoff condition, decided 2026-09-14: 2025-01-02..04-30 (75 decision points), not all of
# 2025. Most of 2025 lies after the best-estimate knowledge cutoff (mid-April 2025) and would
# dilute the contamination upper bound; the pre-minus-post IC gap's SE rises ~26%, accepted
# because the condition is descriptive. A2 learning steps: 64 here vs 74 post-cutoff.
# (news_selfcheck and news_stats keep all of 2025 on purpose: a leakage check over a superset.)
CONDITIONS = {"post": ("2026-04-27", "2026-08-26"), "pre": ("2025-01-02", "2025-04-30")}

# Arm-order seed, chosen 2026-09-14 by a rule fixed before searching: the smallest
# non-negative integer seed whose arm-order counts give chi-square p > 0.5 on both
# phase-one calendars (date indices 0..84 and 0..74). Seed 0: p = 0.9928 and 0.9942.
# Chosen for arm-order uniformity only; arm order does not touch outcomes, so this is not tuning.
ARM_ORDER_SEED = 0
ARM_ORDER_SEED_RULE = ("smallest non-negative integer seed with chi-square p > 0.5 for the arm-order "
                       "counts on both phase-one calendars (85 and 75 decision dates)")
ARM_ORDER_SEED_REASON = "arm-order uniformity only; arm order does not touch outcomes, so this is not tuning"


def arm_order_uniformity(seed, n_dates, arms=None):
    """Chi-square test of the arm-order counts over date indices 0..n_dates-1."""
    from itertools import permutations
    from scipy.stats import chisquare
    arms = arms or ARMS
    perms = sorted(permutations(arms))
    counts = {p: 0 for p in perms}
    for j in range(n_dates):
        counts[arm_order(seed, j, arms)] += 1
    return {"counts": {"-".join(p): c for p, c in counts.items()},
            "chi2_p": float(chisquare(list(counts.values())).pvalue)}


def manifest_fields(seed=ARM_ORDER_SEED):
    """Arm-order and arm-protocol provenance for records.write_run(extra=...)."""
    return {"arm_order_seed": seed, "arm_order_seed_rule": ARM_ORDER_SEED_RULE,
            "arm_order_seed_reason": ARM_ORDER_SEED_REASON,
            "arm_order_uniformity": {label: arm_order_uniformity(seed, n)
                                     for label, n in (("post_85", 85), ("pre_75", 75))},
            "conditions": CONDITIONS,
            "a1_window": {"k": WINDOW_K, "format": ap.A1_WINDOW_FORMAT,
                          "calls_per_decision": ap.A1_CALLS_PER_DECISION,
                          "decided": "2026-09-15 (user)", "reason": ap.A1_WINDOW_FORMAT_REASON,
                          "tokens_reference": "about 2,455 per decision (mean 2,490), 12,466 for a full window of 5; "
                                              "results/budget/budget_20260914_120638.json"}}


class LookaheadError(LookupError):
    """An outcome was requested before its mature_date."""


class TradingCalendar:
    def __init__(self, days):
        idx = pd.DatetimeIndex(days)
        if idx.has_duplicates or not idx.is_monotonic_increasing:
            raise ValueError("trading days must be strictly increasing")
        self.days = idx
        self._pos = {d: i for i, d in enumerate(idx)}

    def position(self, day):
        try:
            return self._pos[pd.Timestamp(day)]
        except KeyError:
            raise ValueError(f"{day} is not a trading day") from None

    def shift(self, day, n):
        """The trading day n sessions after `day`, or None past the end of the calendar."""
        i = self.position(day) + n
        return self.days[i] if i < len(self.days) else None

    def between(self, start, end):
        return self.days[(self.days >= pd.Timestamp(start)) & (self.days <= pd.Timestamp(end))]


class OutcomeView:
    """Outcomes visible at the close of `as_of`. Holds nothing that matures later."""

    def __init__(self, as_of, visible):
        self.as_of = pd.Timestamp(as_of)
        self._visible = MappingProxyType(dict(visible))

    def realized(self, decision_date):
        d = pd.Timestamp(decision_date)
        if d not in self._visible:
            raise LookaheadError(f"outcome of {d.date()} is not matured as of {self.as_of.date()}")
        return self._visible[d]

    def matured_decision_dates(self):
        return tuple(sorted(self._visible))


class OutcomeStore:
    """h=10 outcomes keyed by decision date, released only through as_of(t)."""

    def __init__(self, outcomes, calendar, delay=DELAY):
        self._outcomes = {pd.Timestamp(k): MappingProxyType(dict(v)) for k, v in outcomes.items()}
        self._calendar = calendar
        self._delay = delay

    def as_of(self, t):
        t = pd.Timestamp(t)
        visible = {}
        for d, v in self._outcomes.items():
            m = self._calendar.shift(d, self._delay)
            if m is not None and m <= t:
                visible[d] = v
        return OutcomeView(t, visible)


@dataclass(frozen=True)
class Decision:
    arm: str
    decision_date: pd.Timestamp
    decision_index: int
    mature_date: object          # pd.Timestamp, or None when the calendar ends first
    voided: bool
    scores: object               # read-only mapping, or None when void
    order_position: int
    response: object = None      # raw text of the usable generation; None when void
    bullet_ids: tuple = ()       # playbook bullets that generation cited


@dataclass(frozen=True)
class Slot:
    """One A1 window slot: a matured decision, or an empty slot left by a void one."""
    decision_date: pd.Timestamp
    voided: bool
    scores: object
    outcome: object
    reasoning: object = None     # the generation's "reasoning" field, when it has one
    matured_on: object = None


@dataclass(frozen=True)
class PlaybookEntry:
    content: str
    source_decision_date: pd.Timestamp
    matured_on: pd.Timestamp
    bullet_id: object = None


class A0Memory:
    arm = "A0"

    def snapshot(self):
        return ()


class A1Window:
    arm = "A1"

    def __init__(self, k=WINDOW_K):
        self._slots = deque(maxlen=k)

    def add(self, slot):
        self._slots.append(slot)

    def snapshot(self):
        """Matured, non-void decisions in the window, oldest first."""
        return tuple(s for s in self._slots if not s.voided)

    def skipped_void(self):
        return tuple(s.decision_date for s in self._slots if s.voided)


class A2Playbook:
    """Stand-in playbook for part 1: entries with provenance. Part 2 wraps the ACE text playbook."""
    arm = "A2"

    def __init__(self):
        self._entries = []

    def apply(self, operations, source_decision_date, matured_on):
        for op in operations:
            if op.get("type") != "ADD":
                raise ValueError(f"part-1 playbook only applies ADD, got {op.get('type')}")
            self._entries.append(PlaybookEntry(str(op.get("content", "")),
                                               pd.Timestamp(source_decision_date), pd.Timestamp(matured_on)))

    def commit(self, maturity, source_decision_date, matured_on):
        """Apply a finished maturity: only a successful Curator changes the playbook."""
        c = maturity["curator"]
        if c is not None and c["status"] == "ok":
            self.apply(json.loads(c["response"]).get("operations", []), source_decision_date, matured_on)

    def snapshot(self):
        return tuple(self._entries)


def arm_order(seed, date_index, arms=ARMS):
    """Arm order for one decision date; reproducible from (seed, date index) alone."""
    rng = np.random.default_rng([int(seed), int(date_index)])
    return tuple(arms[i] for i in rng.permutation(len(arms)))


@dataclass
class ReplayResult:
    decisions: list = field(default_factory=list)       # Decision, in call order
    outcomes: dict = field(default_factory=dict)        # (arm, date) -> run_decision outcome
    generations: list = field(default_factory=list)     # one dict per arm per decision date
    harvests: list = field(default_factory=list)        # one dict per matured decision
    orders: dict = field(default_factory=dict)          # decision date -> arm order
    meta_rows: dict = field(default_factory=lambda: {a: [] for a in ARMS})
    unmatured: list = field(default_factory=list)       # decisions whose mature_date is past the calendar
    memories: dict = field(default_factory=dict)


def _assert_no_future_feedback(arm, snapshot, day, calendar, delay):
    limit = calendar.position(day) - delay
    for item in snapshot:
        source = item.source_decision_date if arm == "A2" else item.decision_date
        if calendar.position(source) > limit:
            raise AssertionError(f"{arm} on {day.date()} holds feedback from {source.date()}, "
                                 f"less than {delay} sessions earlier")
        if arm == "A2" and item.matured_on > day:
            raise AssertionError(f"A2 on {day.date()} holds an entry that matured {item.matured_on.date()}")


def _reasoning_of(response):
    try:
        obj = json.loads(response)
    except (TypeError, ValueError):
        return None
    r = obj.get("reasoning") if isinstance(obj, dict) else None
    return None if r is None else str(r)


def run_replay(calendar, decision_dates, universe, generate, reflect, curate, outcomes, seed,
               delay=DELAY, arms=ARMS, memories=None,
               reflector_check=ap.json_object_check, curator_check=ap.json_object_check):
    """Replay one condition.

    universe: one list for every date, or a mapping decision date -> point-in-time list.
    generate(arm, decision_date, snapshot, attempt_no) -> {"response": str, ...}
    reflect(decision, realized, attempt_no)           -> {"response": str, ...}   (A2 only)
    curate(snapshot, reflection, decision, attempt_no) -> {"response": str, ...}
    memories: fresh {"A0", "A1", "A2"} memory objects; the A2 one must offer commit(maturity,
    source_decision_date, matured_on). Defaults to the part-1 stand-ins.
    """
    decision_dates = pd.DatetimeIndex(decision_dates)
    for d in decision_dates:
        calendar.position(d)
    index_of = {d: i for i, d in enumerate(decision_dates)}
    last_decision = decision_dates[-1]
    if memories is None:
        memories = {"A0": A0Memory(), "A1": A1Window(), "A2": A2Playbook()}
    res = ReplayResult(memories=memories)
    pending = []

    start = calendar.position(decision_dates[0])
    for pos in range(start, len(calendar.days)):
        day = calendar.days[pos]
        if day > last_decision and not any(d.mature_date is not None for d in pending):
            break
        view = outcomes.as_of(day)

        overdue = [d for d in pending if d.mature_date is not None and d.mature_date < day]
        if overdue:
            raise RuntimeError(f"{len(overdue)} decisions passed their mature_date unharvested on {day.date()}")
        due = sorted((d for d in pending if d.mature_date == day), key=lambda d: (d.arm, d.decision_index))
        pending = [d for d in pending if d.mature_date != day]

        maturity_today = {}
        for dec in due:
            event = {"arm": dec.arm, "decision_date": dec.decision_date, "harvested_on": day, "voided": dec.voided}
            if day > last_decision:
                event["status"] = "after_last_decision"
            elif dec.arm == "A2":
                realized = None if dec.voided else view.realized(dec.decision_date)
                mat = ap.process_maturity(
                    dec.decision_date, dec.voided,
                    lambda n, dec=dec, realized=realized: reflect(dec, realized, n),
                    lambda n, reflection, dec=dec: curate(memories["A2"].snapshot(), reflection, dec, n),
                    reflector_check=reflector_check, curator_check=curator_check)
                memories["A2"].commit(mat, dec.decision_date, day)
                maturity_today["A2"] = mat
                event["status"] = mat["maturity_status"]
                event["reflector_status"] = mat["reflector"]["status"] if mat["reflector"] else "not_run"
                event["curator_status"] = mat["curator"]["status"] if mat["curator"] else "not_run"
            elif dec.arm == "A1":
                realized = None if dec.voided else view.realized(dec.decision_date)
                memories["A1"].add(Slot(dec.decision_date, dec.voided, dec.scores, realized,
                                        reasoning=_reasoning_of(dec.response), matured_on=day))
                event["status"] = "skipped_void" if dec.voided else "entered_window"
            else:
                event["status"] = "no_memory"
            res.harvests.append(event)

        if day not in index_of:
            continue
        j = index_of[day]
        order = arm_order(seed, j, arms)
        res.orders[day] = order
        for position, arm in enumerate(order):
            snapshot = memories[arm].snapshot()
            if arm in ("A1", "A2"):
                _assert_no_future_feedback(arm, snapshot, day, calendar, delay)
            members = universe[day] if isinstance(universe, dict) else universe
            outcome = ap.run_decision(
                lambda n, arm=arm, snapshot=snapshot: generate(arm, day, snapshot, n), members)
            mature = calendar.shift(day, delay)
            scores = None if outcome["voided"] else MappingProxyType(dict(outcome["scores"]))
            last = outcome["attempts"][-1]
            dec = Decision(arm, day, j, mature, outcome["voided"], scores, position,
                           response=None if outcome["voided"] else last.get("response"),
                           bullet_ids=() if outcome["voided"] else tuple(last.get("bullet_ids") or ()))
            res.decisions.append(dec)
            res.outcomes[(arm, day)] = outcome
            if mature is None:
                res.unmatured.append(dec)
            else:
                pending.append(dec)

            gen = {"arm": arm, "decision_date": day, "decision_index": j, "order_position": position,
                   "n_attempts": outcome["n_attempts"], "voided": outcome["voided"]}
            window = None
            if arm == "A1":
                gen["memory_sources"] = tuple(s.decision_date for s in snapshot)
                gen["window_skipped_void"] = memories["A1"].skipped_void()
                window = (gen["memory_sources"], gen["window_skipped_void"])
            elif arm == "A2":
                gen["memory_sources"] = tuple(e.source_decision_date for e in snapshot)
                gen["memory_matured_on"] = tuple(e.matured_on for e in snapshot)
            else:
                gen["memory_sources"] = ()
            res.generations.append(gen)

            maturity = None
            if arm == "A2":
                maturity = maturity_today.get("A2") or ap.process_maturity(None, False, None, None)
            res.meta_rows[arm].append(ap.meta_row(day, outcome, maturity=maturity, window=window))

    res.unharvested = [d for d in pending if d.mature_date is not None]
    return res
