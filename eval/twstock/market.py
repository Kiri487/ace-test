"""Data layer for the TW stock screening task.

Everything the adapter needs from finlab lives here: auth, the cached
FinlabDataFrames, the point-in-time universe, the decision-date schedule, and
the aggregate market context handed to the LLM.

Two rules this module exists to enforce:

1. **No look-ahead.** Every value handed out is taken as `.loc[:date].iloc[-1]`,
   i.e. the latest observation published on or before the decision date. finlab
   indexes monthly revenue and financial statements by publication date, not by
   the period they describe, so this is sufficient - but it is only sufficient
   if nothing slices by position or by period instead.

2. **No stock identity reaches the LLM.** `market_context` returns aggregate
   statistics only. Per-stock values are available to the compiler and the
   backtest, never to the prompt.
"""

import os
from functools import lru_cache

import finlab
from finlab import data
from finlab.data import FileStorage

CACHE_DIR = os.environ.get("TWSTOCK_FINLAB_CACHE", "/home/ubuntu/finlab_cache_ace")
ENV_FILE = os.environ.get("TWSTOCK_ENV_FILE", "/home/ubuntu/stock-analysis/.env")
SESSION_VARS = ("FINLAB_REFRESH_TOKEN", "FINLAB_SESSION_ID", "FINLAB_API_KEY")

UNIVERSE_SIZE = 50
HOLD_DAYS = 20          # trading days, not calendar days
STEP_DAYS = 21          # trading days between decision dates; > HOLD_DAYS so
                        # sample windows never overlap by construction

_logged_in = False


def _load_env_file(path):
    if not os.path.exists(path):
        return
    for raw in open(path, encoding="utf-8"):
        raw = raw.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        k, v = raw.split("=", 1)
        v = v.strip().strip('"').strip("'")
        if k.startswith("FINLAB") and v and k not in os.environ:
            os.environ[k] = v


def login():
    """Authenticate exactly the way stock-analysis/utils/finlab_auth.py does.

    Never calls finlab.login() with no argument: on expired credentials that
    falls back to a browser flow and blocks for 300s in a headless shell.
    """
    global _logged_in
    if _logged_in:
        return
    _load_env_file(ENV_FILE)
    os.makedirs(CACHE_DIR, exist_ok=True)
    data.set_storage(FileStorage(CACHE_DIR))

    if all(os.environ.get(v) for v in SESSION_VARS):
        from finlab.auth import get_id_token
        if get_id_token():
            _logged_in = True
            return
    token = os.environ.get("FINLAB_API_TOKEN")
    if not token:
        raise EnvironmentError(
            "No FinLab credentials. Expected %s or FINLAB_API_TOKEN in %s"
            % (" / ".join(SESSION_VARS), ENV_FILE)
        )
    finlab.login(token)
    _logged_in = True


# --------------------------------------------------------------------------
# Fields. Each entry is (finlab key, one-line description shown to the LLM,
# whether zero is a meaningful boundary for that field).
# --------------------------------------------------------------------------
FIELD_SPECS = {
    "ma20_bias": (None,
                  "股價相對 20 日均線的乖離,正值代表站上均線",
                  True),
    "pe": ("price_earning_ratio:本益比",
           "本益比,越低越便宜",
           False),
    "rev_yoy": ("monthly_revenue:去年同月增減(%)",
                "月營收年增率(%),正值代表營收比去年同月成長",
                True),
    "roe": ("fundamental_features:ROE稅後",
            "股東權益報酬率,衡量公司用股東的錢賺錢的效率",
            False),
    "foreign_buy": ("institutional_investors_trading_summary:外陸資買賣超股數(不含外資自營商)",
                    "外資近 20 個交易日的累計買賣超股數,正值代表買超",
                    True),
}

FIELD_NAMES = tuple(FIELD_SPECS)


def _to_date_index(df):
    """Quarterly datasets arrive indexed by strings like '2023-Q1'.

    index_str_to_date() maps those to the date the filing was actually uploaded,
    which is both what makes `.loc[:decision_date]` work and what keeps the
    value out of reach before it was public.
    """
    import pandas as pd
    if isinstance(df.index, pd.DatetimeIndex):
        return df
    return df.index_str_to_date()


@lru_cache(maxsize=1)
def _frames():
    """Load every DataFrame once. Cached for the life of the process."""
    login()
    adj_close = data.get("etl:adj_close")
    frames = {
        "market_value": data.get("etl:market_value"),
        "adj_close": adj_close,
        # Deviation from the 20-day MA. Zero is "sitting on the average".
        "ma20_bias": adj_close / adj_close.average(20) - 1,
        "pe": _to_date_index(data.get(FIELD_SPECS["pe"][0])),
        "rev_yoy": _to_date_index(data.get(FIELD_SPECS["rev_yoy"][0])),
        "roe": _to_date_index(data.get(FIELD_SPECS["roe"][0])),
        # A single day of net buying is mostly noise; sum over the same horizon
        # as the holding period so the field means what its description says.
        "foreign_buy": _to_date_index(
            data.get(FIELD_SPECS["foreign_buy"][0])).rolling(HOLD_DAYS).sum(),
    }
    return frames


def trading_days():
    """The trading-day index everything else is positioned against."""
    return _frames()["adj_close"].index


def decision_dates(start, count, step=STEP_DAYS):
    """`count` decision dates, `step` trading days apart, starting at or after
    `start`. Spacing is in trading days so holidays cannot make one sample's
    holding window overlap the next."""
    idx = trading_days()
    pos = idx.searchsorted(start)
    out = []
    for i in range(count):
        p = pos + i * step
        # Need HOLD_DAYS of future bars for the window to be evaluable.
        if p + HOLD_DAYS >= len(idx):
            break
        out.append(idx[p])
    return out


def holding_window(date):
    """(entry_date, exit_date) for a decision made at `date`.

    Entry is the *next* trading day's open: the signal is only known once the
    decision date has closed, so entering on the decision date itself would be
    look-ahead.
    """
    idx = trading_days()
    p = idx.searchsorted(date)
    return idx[p + 1], idx[p + 1 + HOLD_DAYS - 1]


def universe(date):
    """Point-in-time top-N by market cap, as of `date`.

    `is_largest` ranks within each row, so membership is recomputed for every
    date - today's index composition never leaks backwards.
    """
    mv = _frames()["market_value"]
    row = mv.loc[:date]
    if row.empty:
        raise ValueError(f"no market cap data on or before {date}")
    return list(row.iloc[-1].nlargest(UNIVERSE_SIZE).index)


def field_values(date, members):
    """{field: Series over `members`} as known on `date`.

    Every field is read as the last observation at or before the decision date.
    Missing values stay NaN and are excluded downstream rather than silently
    passing a filter.
    """
    frames = _frames()
    out = {}
    for name in FIELD_NAMES:
        df = frames[name]
        hist = df.loc[:date]
        if hist.empty:
            raise ValueError(f"no data for {name} on or before {date}")
        row = hist.iloc[-1]
        out[name] = row.reindex(members).astype("float64")
    return out


def market_context(date):
    """Aggregate state of the universe, for the prompt.

    Deliberately contains no stock identity: the two numbers describe how the
    universe as a whole moved and how much its members diverged.
    """
    idx = trading_days()
    p = idx.searchsorted(date)
    start = idx[max(0, p - HOLD_DAYS)]
    px = _frames()["adj_close"]
    members = universe(date)
    prior = px.loc[start:date, members]
    if len(prior) < 2:
        return {"universe_return_pct": 0.0, "dispersion_pct": 0.0}
    rets = (prior.iloc[-1] / prior.iloc[0] - 1).dropna()
    return {
        "universe_return_pct": round(float(rets.mean()) * 100, 2),
        "dispersion_pct": round(float(rets.std()) * 100, 2),
    }
