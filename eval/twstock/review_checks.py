"""Adversarial checks on the parts of this package most likely to be wrong.

Not a demo - each block exists because a plausible bug would show up here and
nowhere else. Run with the ACE venv.
"""

import json

import pandas as pd

from . import market, rules
from .evaluate import sim_return, direct_return, _position_frame
from .data_processor import DataProcessor


def hr(t):
    print("\n" + "=" * 74)
    print("  " + t)
    print("=" * 74)


def main():
    market.login()
    d = market.decision_dates("2023-01-01", 1)[0]
    members = market.universe(d)
    entry, exit_ = market.holding_window(d)
    vals = market.field_values(d, members)

    hr("A. does sim() actually trade every name it was given?")
    pos = _position_frame(members, entry, exit_)
    from finlab.backtest import sim
    rep = sim(pos, resample=None, trade_at_price="open",
              fee_ratio=1.425/1000, tax_ratio=3/1000,
              position_limit=0.25, upload=False, name="review")
    tr = rep.get_trades()
    print(f"   asked to hold {len(members)} names; sim opened {len(tr)} trades")
    missing = set(members) - set(tr['symbol'].astype(str).str.split().str[0])
    print(f"   names with no trade: {len(missing)} {sorted(missing)[:10]}")
    print("   -> a silent shortfall here would bias every benchmark return")

    hr("B. entry really is the day AFTER the decision date")
    idx = market.trading_days()
    p = idx.searchsorted(d)
    print(f"   decision {str(d)[:10]} is idx[{p}]; entry {str(entry)[:10]} is idx[{idx.searchsorted(entry)}]")
    print(f"   holding bars: {idx.searchsorted(exit_) - idx.searchsorted(entry) + 1} (want {market.HOLD_DAYS})")
    print(f"   -> {'PASS' if idx.searchsorted(entry) == p + 1 else 'FAIL'}")

    hr("C. successive decision dates never overlap")
    ds = market.decision_dates("2023-01-01", 6)
    bad = []
    for a, b in zip(ds, ds[1:]):
        _, a_exit = market.holding_window(a)
        if a_exit >= b:
            bad.append((str(a)[:10], str(a_exit)[:10], str(b)[:10]))
    print(f"   checked {len(ds)-1} consecutive pairs; overlaps: {len(bad)} {bad}")
    print(f"   -> {'PASS' if not bad else 'FAIL'}")

    hr("D. percentile filters are computed over the universe, not the survivors")
    one = rules.select({"filters": [{"field": "roe", "op": "top_pct", "value": 50}],
                        "rank_by": "pe", "direction": "low", "n": 25}, vals)
    two = rules.select({"filters": [{"field": "roe", "op": "top_pct", "value": 50},
                                    {"field": "pe", "op": "bottom_pct", "value": 50}],
                        "rank_by": "pe", "direction": "low", "n": 25}, vals)
    pe_bottom = rules.select({"filters": [{"field": "pe", "op": "bottom_pct", "value": 50}],
                              "rank_by": "pe", "direction": "low", "n": 25}, vals)
    inter = sorted(set(one) & set(pe_bottom))
    print(f"   roe-top50 alone      : {len(one)}")
    print(f"   pe-bottom50 alone    : {len(pe_bottom)}")
    print(f"   both filters together: {len(two)}")
    print(f"   set intersection     : {len(inter)}")
    print(f"   -> {'independent screens (as intended)' if sorted(two) == inter else 'ORDER DEPENDENT - bug'}")

    hr("E. NaN never passes a filter")
    s = vals["pe"].copy()
    n_nan = int(s.isna().sum())
    keep = rules._apply_filter(s, "bottom_pct", 100)
    print(f"   pe has {n_nan} NaN of {len(s)}; bottom_pct(100) kept {len(keep)}")
    print(f"   -> {'PASS' if len(keep) == len(s) - n_nan else 'FAIL - NaN slipped through'}")

    hr("F. the cache really keys on the rule, not just the date")
    proc = DataProcessor()
    tgt = json.dumps({"date": str(d)[:10]})
    r1 = {"filters": [], "rank_by": "roe", "direction": "high", "n": 5}
    r2 = {"filters": [], "rank_by": "roe", "direction": "low", "n": 5}
    a1 = proc.answer_is_correct(str(r1), tgt)
    n_after_1 = len(proc._cache)
    a2 = proc.answer_is_correct(str(r2), tgt)
    n_after_2 = len(proc._cache)
    a1_again = proc.answer_is_correct(str(r1), tgt)
    n_after_3 = len(proc._cache)
    print(f"   rule1 -> {a1} (cache {n_after_1}), rule2 -> {a2} (cache {n_after_2}),"
          f" rule1 again -> {a1_again} (cache {n_after_3})")
    ok = n_after_1 == 1 and n_after_2 == 2 and n_after_3 == 2 and a1 == a1_again
    print(f"   -> {'PASS' if ok else 'FAIL - cache is collapsing distinct rules'}")

    hr("G. key-order and formatting variations hit the same cache entry")
    r1b = {"n": 5, "direction": "high", "rank_by": "roe", "filters": []}
    before = len(proc._cache)
    proc.answer_is_correct(json.dumps(r1b), tgt)
    print(f"   same rule, keys reordered, JSON instead of repr: cache {before} -> {len(proc._cache)}")
    print(f"   -> {'PASS' if len(proc._cache) == before else 'duplicate entry (wasteful, not wrong)'}")

    hr("H. a rule selecting fewer than the minimum is rejected, not silently run")
    tight = {"filters": [{"field": "roe", "op": "top_pct", "value": 2},
                         {"field": "pe", "op": "bottom_pct", "value": 2}],
             "rank_by": "roe", "direction": "high", "n": 10}
    try:
        got = rules.select(tight, vals)
        print(f"   selected {len(got)} names -> {'FAIL' if len(got) < rules.MIN_SELECTED else 'ok, enough passed'}")
    except rules.InvalidRule as e:
        print(f"   InvalidRule: {e}")
        print("   -> PASS")

    hr("I. n larger than the candidate pool")
    wide = {"filters": [], "rank_by": "roe", "direction": "high", "n": 25}
    got = rules.select(wide, vals)
    print(f"   asked for 25 from a {len(members)}-name universe -> got {len(got)}")
    print(f"   -> {'PASS' if len(got) == 25 else 'check: fewer than requested'}")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
