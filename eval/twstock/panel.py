"""Price and universe panel for the evaluation layer (v9 §4.1, §4.2).

Deliberately separate from market.py. That module serves the screening-rule
pipeline: a fixed 20-day holding window (HOLD_DAYS) and non-overlapping decision
dates (STEP_DAYS). This layer decides daily, uses overlapping windows and is
parametric in the horizon h, so the only thing it shares is login().
"""

from functools import lru_cache

import pandas as pd
from finlab import data

from . import market

UNIVERSE_SIZE = 50
HORIZONS = (5, 10, 20, 40)

DATASETS = {
    "adj_open": "etl:adj_open",
    "adj_close": "etl:adj_close",
    "market_value": "etl:market_value",
    "pe": "price_earning_ratio:本益比",
    # Unadjusted prices, used only to check the adjustment itself.
    "raw_open": "price:開盤價",
    "raw_close": "price:收盤價",
}


@lru_cache(maxsize=None)
def frame(name):
    """One finlab dataset as a plain, date-sorted DataFrame.

    Converted out of FinlabDataFrame on purpose: that class overrides
    arithmetic and alignment, and every shift in this layer must be a plain
    positional shift over the trading-day index.
    """
    market.login()
    df = pd.DataFrame(data.get(DATASETS[name]))
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    if df.columns.duplicated().any():
        dup = list(df.columns[df.columns.duplicated()][:5])
        raise ValueError(f"{name}: duplicated stock columns, e.g. {dup}")
    return df


def trading_days():
    return frame("adj_open").index


def decision_dates(start, end):
    """Every trading day in [start, end] - v9 §4.1 decides daily."""
    idx = trading_days()
    return idx[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]


def universe(date):
    """Point-in-time top 50 by market cap, from the latest row on or before date.

    market_value also carries weekend rows; reading the latest row at or before
    a trading day makes that irrelevant.
    """
    mv = frame("market_value").loc[:date]
    if mv.empty:
        raise ValueError(f"no market value on or before {date}")
    return list(mv.iloc[-1].nlargest(UNIVERSE_SIZE).index)


def universes(dates):
    return {d: universe(d) for d in dates}


def data_last_dates(names=("adj_open", "adj_close", "market_value", "pe")):
    return {n: str(frame(n).index[-1].date()) for n in names}
