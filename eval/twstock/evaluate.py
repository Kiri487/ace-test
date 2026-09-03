"""Scoring a rule: run the backtest, compare against the same-window benchmark.

The comparison is deliberately narrow. The rule's basket and the benchmark share
the window, the universe, and equal weighting; the only difference between them
is the screen. So a win means the screen added value, not that the market rose -
which is the whole reason "beat the previous backtest" was rejected as a target.

Two return paths are computed on purpose:

  sim()   - the real engine, with fees and tax, and the thing the report cites.
  direct  - two adjusted-close snapshots, no costs.

They should differ by roughly the round-trip cost. If they ever diverge more
than that, the fast path must not be substituted for sim() in the training loop.
"""

import pandas as pd
from finlab.backtest import sim

from . import market
from .rules import select, InvalidRule

FEE_RATIO = 1.425 / 1000
TAX_RATIO = 3 / 1000
POSITION_LIMIT = 0.25


def _position_frame(stocks, entry, exit_):
    """A position matrix that is False either side of the holding window, so
    sim() has an unambiguous entry and exit bar."""
    idx = market.trading_days()
    p_entry = idx.searchsorted(entry)
    p_exit = idx.searchsorted(exit_)
    lo = max(0, p_entry - 1)
    hi = min(len(idx) - 1, p_exit + 1)
    window = idx[lo:hi + 1]
    pos = pd.DataFrame(False, index=window, columns=list(stocks))
    pos.loc[entry:exit_, :] = True
    return pos


def sim_return(stocks, entry, exit_):
    """Portfolio return over the window from the real backtest engine.

    Positions are equal weight and never rebalanced, so the mean of the
    per-trade returns is the portfolio return.
    """
    pos = _position_frame(stocks, entry, exit_)
    report = sim(
        pos,
        resample=None,
        trade_at_price="open",
        fee_ratio=FEE_RATIO,
        tax_ratio=TAX_RATIO,
        position_limit=POSITION_LIMIT,
        upload=False,        # must be explicit: None defers to an env var
        name="twstock_mvp",
    )
    trades = report.get_trades()
    if trades is None or len(trades) == 0:
        return 0.0, report
    return float(trades["return"].mean()), report


def direct_return(stocks, entry, exit_):
    """Same window, no costs: mean of per-stock adjusted-close returns."""
    px = market._frames()["adj_close"]
    cols = [s for s in stocks if s in px.columns]
    if not cols:
        return 0.0
    start = px.loc[:entry, cols].iloc[-1]
    end = px.loc[:exit_, cols].iloc[-1]
    rets = (end / start - 1).dropna()
    return float(rets.mean()) if len(rets) else 0.0


def evaluate_rule(rule, date, use_sim=True):
    """Score one rule at one decision date.

    Returns a dict. `correct` is the bool the Reflector keys off; `excess` is
    the magnitude the playbook selection metric can aggregate.
    """
    members = market.universe(date)
    entry, exit_ = market.holding_window(date)
    values = market.field_values(date, members)

    out = {
        "date": str(date)[:10],
        "entry": str(entry)[:10],
        "exit": str(exit_)[:10],
        "universe_size": len(members),
    }

    try:
        picked = select(rule, values)
    except InvalidRule as e:
        out.update(valid=False, error=str(e), correct=False)
        return out

    if use_sim:
        rule_ret, _ = sim_return(picked, entry, exit_)
        bench_ret, _ = sim_return(members, entry, exit_)
    else:
        rule_ret = direct_return(picked, entry, exit_)
        bench_ret = direct_return(members, entry, exit_)

    out.update(
        valid=True,
        selected=picked,
        n_selected=len(picked),
        rule_return=round(rule_ret, 6),
        benchmark_return=round(bench_ret, 6),
        excess=round(rule_ret - bench_ret, 6),
        correct=rule_ret > bench_ret,
        rule_return_direct=round(direct_return(picked, entry, exit_), 6),
        benchmark_return_direct=round(direct_return(members, entry, exit_), 6),
    )
    return out
