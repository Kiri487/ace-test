"""News data layer (v8 §4.1): tw_news_cnyes headlines per stock and decision date.

    .venv/bin/python -m eval.twstock.news          build and write stock/market tables

Decisions fixed on 2026-09-13, recorded in news_config.json and every manifest:

* The visibility time is `date`, the cnyes publication time (Taipei local,
  calendar day only from 2025). `key_date` is finlab's ingestion time in UTC and
  is never used for slicing: it follows the vendor's crawl schedule, and a filter
  on it would remove 35% of 2025 but under 1% of the test window, which would
  make the pre- and post-cutoff conditions incomparable.
* finlab's `data.truncate_end` does not cut this dataset (a long table with a
  RangeIndex, verified 2026-09-13), so point-in-time slicing is done by this
  module alone and checked by news_selfcheck.
* One row per cnyes article id. When copies disagree, the latest calendar day is
  kept, ties broken by the latest key_date. Keeping the later day can only move
  a headline into a later window, never an earlier one.
* An article tagged with more than 50 tickers is market-level. It never enters
  per-stock input and is kept in its own table.
* The window for decision date T is trading sessions T-4..T. An article belongs
  to the first session on or after its calendar date, so weekend and holiday
  articles join the next session's window instead of being dropped, and an
  article dated T - including one published after the close - is in T's window.
  Entry is at the open of T+1 (v8 §4.2), after every article in the window.
"""

import datetime as _dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from finlab import data

from . import market, panel, records

CONFIG_PATH = Path(__file__).with_name("news_config.json")
MARKET_LEVEL_MAX_IDS = 50
WINDOW_TRADING_DAYS = 5
AID_PATTERN = r"/news/id/(\d+)"
SAVED_COLUMNS = ["aid", "date", "date_day", "key_date", "title", "url",
                 "stock_ids", "n_ids", "session_pos", "session_date"]


def load_config(path=CONFIG_PATH):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    frozen = {
        "time_field": "date",
        "market_level_max_ids": MARKET_LEVEL_MAX_IDS,
        "window_trading_days": WINDOW_TRADING_DAYS,
    }
    for key, want in frozen.items():
        if cfg.get(key) != want:
            raise ValueError(f"news_config.json: {key} = {cfg.get(key)!r}, frozen at {want!r} "
                             "(decided 2026-09-13; not a tunable parameter)")
    if MARKET_LEVEL_MAX_IDS != panel.UNIVERSE_SIZE:
        raise ValueError("the market-level threshold is tied to the universe size")
    return cfg


def config_hash(cfg):
    blob = json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def raw_news(cfg):
    market.login()
    return pd.DataFrame(data.get(cfg["dataset"]))


def parse_ids(value):
    if value is None or pd.isna(value):
        return ()
    return tuple(dict.fromkeys(t.strip() for t in str(value).split(",") if t.strip()))


def clean(raw):
    """One row per article id, with its parsed tickers. Returns (articles, report)."""
    need = ["date", "key_date", "title", "url", "stock_ids"]
    missing = [c for c in need if c not in raw.columns]
    if missing:
        raise ValueError(f"news frame lacks columns {missing}")
    df = raw[need].copy()
    df["url"] = df["url"].astype(str)
    df["title"] = df["title"].astype(str)
    df["stock_ids"] = df["stock_ids"].astype("string")
    aid = pd.to_numeric(df["url"].str.extract(AID_PATTERN, expand=False), errors="coerce")
    if aid.isna().any():
        raise ValueError(f"{int(aid.isna().sum())} rows carry no cnyes article id in url")
    df["aid"] = aid.astype("int64")
    df["date"] = pd.to_datetime(df["date"])
    df["key_date"] = pd.to_datetime(df["key_date"])
    df["date_day"] = df["date"].dt.normalize()

    copies = df.groupby("aid")["date_day"].agg(["size", "nunique"])
    disagree = copies.index[copies["nunique"] > 1]
    df = (df.sort_values(["aid", "date_day", "key_date"], kind="mergesort")
            .drop_duplicates("aid", keep="last")
            .reset_index(drop=True))
    df["ids"] = df["stock_ids"].map(parse_ids)
    df["n_ids"] = df["ids"].map(len).astype("int64")
    report = {
        "raw_rows": int(len(raw)),
        "articles": int(len(df)),
        "ids_with_copies": int((copies["size"] > 1).sum()),
        "ids_whose_copies_disagree_on_day": int(len(disagree)),
        "disagreeing_ids": [int(x) for x in disagree[:50]],
        "keep_rule": "latest date day, ties by latest key_date",
    }
    return df, report


def assign_sessions(articles, calendar):
    """session_pos = index of the first trading day on or after the article's
    calendar date; -1 when that day is past the end of the calendar."""
    cal = pd.DatetimeIndex(calendar).sort_values()
    pos = cal.searchsorted(articles["date_day"].to_numpy(), side="left")
    valid = pos < len(cal)
    out = articles.copy()
    out["session_pos"] = np.where(valid, pos, -1).astype("int64")
    sd = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
    sd.loc[valid] = cal[pos[valid]].to_numpy()
    out["session_date"] = sd
    return out


def split_market_level(articles, max_ids=MARKET_LEVEL_MAX_IDS):
    is_market = articles["n_ids"] > max_ids
    return (articles[~is_market].reset_index(drop=True),
            articles[is_market].reset_index(drop=True))


class NewsIndex:
    """Headline lookup by decision date. Every read goes through session_range."""

    ARTICLE_FIELDS = ["date", "date_day", "title", "n_ids", "session_date"]

    def __init__(self, stock_level, market_level, calendar, window=WINDOW_TRADING_DAYS):
        self.calendar = pd.DatetimeIndex(calendar).sort_values()
        self.window = int(window)
        self.stock_level = stock_level
        self.market_level = market_level

        live = stock_level[stock_level["session_pos"] >= 0]
        long = (live[["aid", "session_pos", "ids"]].explode("ids")
                .dropna(subset=["ids"]).rename(columns={"ids": "stock_id"}))
        long = long.sort_values(["session_pos", "stock_id", "aid"], kind="mergesort")
        self._long_pos = long["session_pos"].to_numpy()
        self._long = long[["stock_id", "aid"]].reset_index(drop=True)
        self._articles = stock_level.set_index("aid")[self.ARTICLE_FIELDS]

        mk = market_level[market_level["session_pos"] >= 0].sort_values(
            ["session_pos", "aid"], kind="mergesort")
        self._mkt_pos = mk["session_pos"].to_numpy()
        self._mkt = mk[["aid"] + self.ARTICLE_FIELDS + ["stock_ids"]].reset_index(drop=True)

    def session_range(self, T):
        T = pd.Timestamp(T)
        if T not in self.calendar:
            raise ValueError(f"{T.date()} is not a trading day")
        hi = self.calendar.get_loc(T)
        lo = hi - self.window + 1
        if lo < 1:
            raise ValueError(f"{T.date()}: fewer than {self.window} trading days of history")
        return lo, hi

    def window_bounds(self, T):
        lo, hi = self.session_range(T)
        return {
            "first_session": self.calendar[lo],
            "last_session": self.calendar[hi],
            "calendar_days_after": self.calendar[lo - 1],
            "calendar_days_through": self.calendar[hi],
        }

    def headlines(self, T, universe=None):
        """Long frame (stock_id, aid, date, date_day, title, n_ids, session_date).
        universe=None returns every tagged stock."""
        lo, hi = self.session_range(T)
        i0, i1 = np.searchsorted(self._long_pos, [lo, hi + 1], side="left")
        sub = self._long.iloc[i0:i1]
        if universe is not None:
            sub = sub[sub["stock_id"].isin([str(s) for s in universe])]
        rows = sub.join(self._articles, on="aid")
        return rows.sort_values(["stock_id", "date", "aid"], kind="mergesort").reset_index(drop=True)

    def headlines_by_stock(self, T, universe):
        """{stock_id: frame} with every universe member present, empty or not."""
        rows = self.headlines(T, universe)
        empty = rows.iloc[0:0].drop(columns="stock_id")
        groups = {sid: g.drop(columns="stock_id").reset_index(drop=True)
                  for sid, g in rows.groupby("stock_id", sort=False)}
        return {str(s): groups.get(str(s), empty) for s in universe}

    def market_headlines(self, T):
        lo, hi = self.session_range(T)
        i0, i1 = np.searchsorted(self._mkt_pos, [lo, hi + 1], side="left")
        return self._mkt.iloc[i0:i1].reset_index(drop=True)


def build(cfg=None):
    """Load, clean, assign sessions and split. Returns (index, info)."""
    cfg = cfg or load_config()
    raw = raw_news(cfg)
    articles, dedup = clean(raw)
    cal = panel.trading_days()
    articles = assign_sessions(articles, cal)
    stock_level, market_level = split_market_level(articles, cfg["market_level_max_ids"])
    index = NewsIndex(stock_level, market_level, cal, cfg["window_trading_days"])
    info = {
        "config": cfg,
        "config_hash": config_hash(cfg),
        "frozen": {"time_field": "date", "market_level_max_ids": MARKET_LEVEL_MAX_IDS,
                   "window_trading_days": WINDOW_TRADING_DAYS},
        "dedup": dedup,
        "articles": int(len(articles)),
        "stock_level_articles": int(len(stock_level)),
        "market_level_articles": int(len(market_level)),
        "articles_without_tickers": int((articles["n_ids"] == 0).sum()),
        "articles_after_calendar_end": int((articles["session_pos"] < 0).sum()),
        "news_date_range": [articles["date"].min(), articles["date"].max()],
        "key_date_max_utc": articles["key_date"].max(),
        "calendar_range": [cal[0], cal[-1]],
        "finlab_truncate_end": "does not cut this dataset (verified 2026-09-13); slicing is done here",
    }
    return index, info


def save(index, info, out_root=None):
    out_root = Path(out_root or records.REPO_ROOT / "results" / "news_layer")
    run_dir = out_root / f"build_{_dt.datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    index.stock_level[SAVED_COLUMNS].to_parquet(run_dir / "stock_news.parquet", index=False)
    index.market_level[SAVED_COLUMNS].to_parquet(run_dir / "market_news.parquet", index=False)
    records.to_json({
        **info,
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "git": records.git_state(),
        "files": {"stock_news.parquet": "per-stock input candidates (n_ids <= 50)",
                  "market_news.parquet": "market-level articles (n_ids > 50), kept, not in per-stock input"},
    }, run_dir / "manifest.json")
    return run_dir


def main():
    index, info = build()
    run_dir = save(index, info)
    print(json.dumps({k: info[k] for k in ("articles", "stock_level_articles", "market_level_articles",
                                           "articles_without_tickers", "articles_after_calendar_end",
                                           "config_hash")}, indent=2))
    print("dedup:", {k: v for k, v in info["dedup"].items() if k != "disagreeing_ids"})
    print("written to", run_dir)


if __name__ == "__main__":
    main()
