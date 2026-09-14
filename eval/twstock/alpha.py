"""α(i, t, h): market-adjusted forward open-to-open return (v9 §4.2).

    r(i, t, h) = open(i, t+1+h) / open(i, t+1) - 1
    α(i, t, h) = r(i, t, h) - r(equal_weight_top50, t, h)

t is the decision date, decided after its close. The position is entered at
open(t+1) and the outcome is settled by open(t+1+h), so feedback can reach a
decision at the close of trading day t+1+h: an actual delay of h+1 trading days.
That day is recorded as mature_date_h{h}; the A1/A2 maturity queues read it.

Missing prices stay NaN and are never filled. The benchmark is the equal-weight
mean over the members whose return exists, and that count is stored next to it.
"""

import numpy as np
import pandas as pd

from .panel import HORIZONS


def forward_open_returns(adj_open, h):
    """r(i, t, h) indexed by decision date t.

    Shifts are positional over the trading-day index, so a holiday can never
    stretch a window to h+1 bars.
    """
    return adj_open.shift(-(1 + h)) / adj_open.shift(-1) - 1


def mature_dates(index, h):
    """The trading day whose open settles α(·, t, h): index[t + 1 + h]."""
    return pd.Series(index, index=index).shift(-(1 + h))


def build_alpha_panel(dates, adj_open, universes, horizons=HORIZONS):
    """Long table, one row per (decision date, universe member), all horizons."""
    members_all = sorted(set().union(*(set(universes[t]) for t in dates)))
    opens = adj_open.reindex(columns=members_all)
    idx = opens.index
    entry = pd.Series(idx, index=idx).shift(-1)
    fwd = {h: forward_open_returns(opens, h) for h in horizons}
    mat = {h: mature_dates(idx, h) for h in horizons}

    blocks = []
    for t in dates:
        members = list(universes[t])
        b = pd.DataFrame({
            "decision_date": t,
            "stock_id": members,
            "universe_rank": np.arange(1, len(members) + 1),
            "entry_date": entry.loc[t],
        })
        for h in horizons:
            r = fwd[h].loc[t].reindex(members)
            bench = r.mean()                       # skips NaN members
            b[f"r_h{h}"] = r.to_numpy()
            b[f"bench_r_h{h}"] = bench
            b[f"bench_n_h{h}"] = int(r.notna().sum())
            b[f"alpha_h{h}"] = (r - bench).to_numpy()
            b[f"mature_date_h{h}"] = mat[h].loc[t]
        blocks.append(b)
    return pd.concat(blocks, ignore_index=True)


def missing_report(panel, horizons=HORIZONS):
    """NaN share of α among rows whose window has matured in the data."""
    out = {}
    for h in horizons:
        matured = panel[f"mature_date_h{h}"].notna()
        a = panel.loc[matured, f"alpha_h{h}"]
        out[h] = {
            "matured_decision_dates": int(panel.loc[matured, "decision_date"].nunique()),
            "rows": int(len(a)),
            "nan_rows": int(a.isna().sum()),
            "nan_share": float(a.isna().mean()) if len(a) else float("nan"),
        }
    return out
