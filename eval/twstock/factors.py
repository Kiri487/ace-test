"""Known-answer scores for validating the evaluation layer. Not LLM arms.

Each score is cross-sectional within that date's point-in-time universe and is
mapped to [-1, 1] by rank, the output range of v9 §4.1. Nothing computed from
these is a finding: they exist to check that α and IC behave as expected, over
a period with no leakage control.
"""

import numpy as np
import pandas as pd

MOMENTUM_LOOKBACK = 20


def unit_rank(values):
    """Average-rank a cross-section into [-1, 1]; NaN stays NaN."""
    r = values.rank()
    n = values.notna().sum()
    if n < 2:
        return r * np.nan
    return 2 * (r - 1) / (n - 1) - 1


def _long(dates, universes, row_fn, name):
    blocks = []
    for t in dates:
        members = list(universes[t])
        raw = row_fn(t).reindex(members).astype("float64")
        blocks.append(pd.DataFrame({
            "decision_date": t,
            "stock_id": members,
            "raw_value": raw.to_numpy(),
            "score": unit_rank(raw).to_numpy(),
        }))
    out = pd.concat(blocks, ignore_index=True)
    out["factor"] = name
    return out


def momentum(dates, universes, adj_close, lookback=MOMENTUM_LOOKBACK):
    """Past `lookback`-day close-to-close return, known at the close of t."""
    past = adj_close / adj_close.shift(lookback) - 1
    return _long(dates, universes, lambda t: past.loc[t], f"momentum_{lookback}d")


def value(dates, universes, pe):
    """Earnings yield 1/PE, as last published on or before t. PE <= 0 or
    missing gives NaN, which is left unranked rather than filled."""
    ey = 1.0 / pe.where(pe > 0)
    return _long(dates, universes, lambda t: ey.loc[:t].iloc[-1], "value_earnings_yield")


def random_uniform(dates, universes, seed):
    rng = np.random.default_rng(seed)
    blocks = []
    for t in dates:
        members = list(universes[t])
        blocks.append(pd.DataFrame({
            "decision_date": t,
            "stock_id": members,
            "raw_value": rng.uniform(-1, 1, len(members)),
        }))
    out = pd.concat(blocks, ignore_index=True)
    out["score"] = out["raw_value"]
    out["factor"] = f"random_seed{seed}"
    return out
