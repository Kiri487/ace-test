"""End-to-end check of the data -> DSL -> backtest path, plus the two gates.

Run before trusting anything this package produces:

    .venv-stock-tests/bin/python -m eval.twstock.selftest

Gate 1 (look-ahead): blanking every observation after the decision date must
not change which stocks the rule selects. If it does, the screen is reading the
future and every downstream number is worthless.

Gate 2 (must-lose): hand the engine a basket that is known to have fallen and
confirm it reports a loss. A backtest that cannot report a loss cannot report a
gain either.
"""

import json
import random

import pandas as pd

from . import market, rules
from .evaluate import evaluate_rule, sim_return, direct_return


def hr(title):
    print("\n" + "=" * 74)
    print(f"  {title}")
    print("=" * 74)


def main():
    hr("1. auth + data")
    market.login()
    idx = market.trading_days()
    print(f"   trading days: {len(idx)}  {str(idx[0])[:10]} .. {str(idx[-1])[:10]}")

    hr("2. decision-date schedule (21 trading days apart)")
    train = market.decision_dates("2023-01-01", 5)
    val = market.decision_dates(market.holding_window(train[-1])[1], 5 + 1)[1:]
    test = market.decision_dates(market.holding_window(val[-1])[1], 5 + 1)[1:]
    for label, dates in (("train", train), ("val", val), ("test", test)):
        print(f"   {label:5} {[str(d)[:10] for d in dates]}")
    gap1 = (val[0] - market.holding_window(train[-1])[1]).days
    gap2 = (test[0] - market.holding_window(val[-1])[1]).days
    print(f"   gap after train: {gap1} calendar days | after val: {gap2}")

    d = train[0]
    hr(f"3. universe + market context at {str(d)[:10]}")
    members = market.universe(d)
    print(f"   universe: {len(members)} names, first 10 = {members[:10]}")
    print(f"   context : {market.market_context(d)}")
    entry, exit_ = market.holding_window(d)
    print(f"   window  : entry {str(entry)[:10]}  exit {str(exit_)[:10]}")

    hr("4. field coverage in the universe (NaN would silently shrink a screen)")
    vals = market.field_values(d, members)
    for name, s in vals.items():
        print(f"   {name:12} non-null {s.notna().sum():>3}/{len(s):<3} "
              f"min={s.min():>10.2f} median={s.median():>10.2f} max={s.max():>12.2f}")

    hr("5. a hand-written rule end to end")
    rule = {
        "filters": [
            {"field": "foreign_buy", "op": "positive"},
            {"field": "roe", "op": "top_pct", "value": 50},
        ],
        "rank_by": "rev_yoy", "direction": "high", "n": 10,
    }
    print("   rule:", json.dumps(rule, ensure_ascii=False))
    res = evaluate_rule(rule, d)
    for k in ("n_selected", "rule_return", "benchmark_return", "excess", "correct"):
        print(f"   {k:20} {res.get(k)}")
    print(f"   selected: {res.get('selected')}")

    hr("6. sim() vs direct (should differ by roughly the round-trip cost)")
    print(f"   rule      sim={res['rule_return']:+.5f}  direct={res['rule_return_direct']:+.5f}"
          f"  diff={res['rule_return']-res['rule_return_direct']:+.5f}")
    print(f"   benchmark sim={res['benchmark_return']:+.5f}  direct={res['benchmark_return_direct']:+.5f}"
          f"  diff={res['benchmark_return']-res['benchmark_return_direct']:+.5f}")
    print("   (round trip fee+tax is about 0.44%, so -0.004 is the expected sign)")

    hr("GATE 1 - look-ahead: blanking the future must not change the selection")
    before = rules.select(rule, market.field_values(d, members))
    frames = market._frames()
    saved = {k: frames[k] for k in frames}
    try:
        for k, df in list(frames.items()):
            trimmed = df.copy()
            trimmed.loc[trimmed.index > d] = float("nan")
            frames[k] = trimmed
        after = rules.select(rule, market.field_values(d, members))
    finally:
        frames.update(saved)
    same = before == after
    print(f"   before: {before}")
    print(f"   after : {after}")
    print(f"   -> {'PASS' if same else 'FAIL - the screen is reading the future'}")

    hr("GATE 2 - must-lose: a basket known to have fallen must report a loss")
    px = frames["adj_close"]
    cols = [s for s in members if s in px.columns]
    realised = (px.loc[:exit_, cols].iloc[-1] / px.loc[:entry, cols].iloc[-1] - 1).dropna()
    losers = list(realised.nsmallest(10).index)
    lose_ret, _ = sim_return(losers, entry, exit_)
    bench = res["benchmark_return"]
    print(f"   worst 10 in window: {losers}")
    print(f"   their sim return  : {lose_ret:+.5f}   (direct {direct_return(losers, entry, exit_):+.5f})")
    print(f"   benchmark         : {bench:+.5f}")
    ok = lose_ret < bench
    print(f"   -> {'PASS' if ok else 'FAIL - engine cannot report a loss'}")

    hr("7. random rules (the control arm the DSL exists to make possible)")
    rng = random.Random(42)
    wins = 0
    for i in range(5):
        r = rng and rules.random_rule(rng)
        out = evaluate_rule(r, d)
        mark = "invalid" if not out.get("valid") else f"excess {out['excess']:+.4f}"
        wins += bool(out.get("correct"))
        print(f"   {i+1}. {json.dumps(r, ensure_ascii=False)[:88]}")
        print(f"      -> {mark}")
    print(f"   {wins}/5 beat the benchmark")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
