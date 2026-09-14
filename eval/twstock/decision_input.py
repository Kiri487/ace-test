"""One decision point's input object (v8 §4.1; §5.1 step 2 "observe"). Data only.

    .venv/bin/python -m eval.twstock.decision_input --date 2026-06-01

No prompt text lives here. How this object is rendered into the Generator's
`context` slot depends on the A0 prompt definition, which v8 §10.1 (4) leaves
open. `to_record()` is the JSON form, so the exact input of every decision can
be written next to its scores (v8 §6.4).

Market-level articles are carried in their own field and are not part of the
per-stock input; whether they are ever rendered is not decided.
"""

import argparse
import json
from dataclasses import dataclass

import pandas as pd

from . import news, panel


@dataclass(frozen=True)
class Headline:
    aid: int
    date: pd.Timestamp
    title: str
    n_tickers: int


@dataclass
class DecisionInput:
    decision_date: pd.Timestamp
    entry_date: pd.Timestamp            # open of T+1, where alpha starts (v8 §4.2)
    universe: list                      # [{"stock_id", "universe_rank"}], rank 1 = largest cap
    news_window: dict                   # sessions and calendar bounds actually used
    news_by_stock: dict                 # stock_id -> [Headline], every member present
    market_news: list                   # [Headline] with > 50 tickers; not per-stock input
    provenance: dict

    def to_record(self):
        def h(x):
            return {"aid": x.aid, "date": x.date.isoformat(), "title": x.title, "n_tickers": x.n_tickers}
        return {
            "decision_date": self.decision_date.isoformat(),
            "entry_date": self.entry_date.isoformat() if self.entry_date is not None else None,
            "universe": self.universe,
            "news_window": {k: v.isoformat() for k, v in self.news_window.items()},
            "news_by_stock": {s: [h(x) for x in v] for s, v in self.news_by_stock.items()},
            "market_news": [h(x) for x in self.market_news],
            "provenance": self.provenance,
        }


def _headlines(frame):
    return [Headline(int(r.aid), pd.Timestamp(r.date), str(r.title), int(r.n_ids))
            for r in frame.itertuples(index=False)]


def build_decision_input(index, T, cfg):
    T = pd.Timestamp(T)
    cal = index.calendar
    pos = cal.get_loc(T)
    U = panel.universe(T)
    by_stock = index.headlines_by_stock(T, U)
    return DecisionInput(
        decision_date=T,
        entry_date=cal[pos + 1] if pos + 1 < len(cal) else None,
        universe=[{"stock_id": s, "universe_rank": i + 1} for i, s in enumerate(U)],
        news_window=index.window_bounds(T),
        news_by_stock={s: _headlines(f) for s, f in by_stock.items()},
        market_news=_headlines(index.market_headlines(T)),
        provenance={
            "news_config_hash": news.config_hash(cfg),
            "time_field": cfg["time_field"],
            "market_level_max_ids": cfg["market_level_max_ids"],
            "window_trading_days": cfg["window_trading_days"],
            "universe": f"point-in-time top {panel.UNIVERSE_SIZE} by etl:market_value",
        },
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-06-01")
    args = ap.parse_args()
    cfg = news.load_config()
    index, _ = news.build(cfg)
    inp = build_decision_input(index, args.date, cfg)
    rec = inp.to_record()
    counts = {s: len(v) for s, v in rec["news_by_stock"].items()}
    print("decision_date", rec["decision_date"], "entry_date", rec["entry_date"])
    print("news_window", rec["news_window"])
    print("universe size", len(rec["universe"]), "first 5", rec["universe"][:5])
    print("headlines per stock:", counts)
    print("market-level articles in window:", len(rec["market_news"]))
    for s in [u["stock_id"] for u in rec["universe"][:2]]:
        print(f"  {s}:", json.dumps(rec["news_by_stock"][s][:3], ensure_ascii=False))
    print("provenance", rec["provenance"])


if __name__ == "__main__":
    main()
