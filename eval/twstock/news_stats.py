"""Descriptive statistics of the news layer. Nothing here selects anything.

    .venv/bin/python -m eval.twstock.news_stats

Per decision date, for the point-in-time top 50:
  headline counts (unique articles, and stock-article pairs) and their tokens,
  universe members with no headline in the window.
Per period: the share of articles removed by the N = 50 market-level rule and the
median ticker count of those removed; two descriptive integrity numbers carried
over from the 2026-09-13 date/key_date investigation (late ingestion, and
articles dated a day earlier than their id neighbours).

The full-history period exists only to check whether the "about 236 headlines /
5.6k tokens" figure recorded before v8 reproduces, and under which reading.
"""

import argparse
import datetime as _dt
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import news, panel, records

PERIODS = {"test_window": ("2026-04-27", "2026-08-26"),
           "pre_cutoff_2025": ("2025-01-02", "2025-12-31")}
TOKENIZER_CANDIDATES = ("deepseek-ai/DeepSeek-V4-Flash", "deepseek-ai/DeepSeek-V3.2", "deepseek-ai/DeepSeek-V3")


def tokenizers():
    """{label: fn(list[str]) -> list[int]}. cl100k is always included for comparison."""
    out = {}
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "20")
    try:
        from transformers import AutoTokenizer
        for name in TOKENIZER_CANDIDATES:
            try:
                tok = AutoTokenizer.from_pretrained(name)
                out[f"hf:{name}"] = lambda texts, tok=tok: [
                    len(x) for x in tok(list(texts), add_special_tokens=False)["input_ids"]]
                break
            except Exception as e:
                print(f"  tokenizer {name} unavailable: {type(e).__name__}: {str(e)[:150]}")
    except Exception as e:
        print(f"  transformers unavailable: {e}")
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    out["tiktoken:cl100k_base"] = lambda texts: [len(x) for x in enc.encode_batch(list(texts))]
    return out


def summarize(s):
    s = pd.Series(s).astype(float)
    return {"n": int(s.size), "mean": float(s.mean()), "std": float(s.std()),
            "min": float(s.min()), "p10": float(s.quantile(.1)), "p25": float(s.quantile(.25)),
            "median": float(s.median()), "p75": float(s.quantile(.75)), "p90": float(s.quantile(.9)),
            "max": float(s.max())}


def per_date(index, dates, tok_by_aid):
    trading = set(index.calendar)
    rows = []
    for T in dates:
        U = panel.universe(T)
        members = set(U)
        h = index.headlines(T, U)
        uniq = h.drop_duplicates("aid")
        m = index.market_headlines(T)
        # astype(bool): on an empty window map() returns object dtype, which pandas
        # would read as a column selection rather than a row mask
        mask = m["stock_ids"].map(lambda s: any(x in members for x in news.parse_ids(s))).astype(bool)
        touch = m[mask]
        rec = {
            "decision_date": T,
            "n_universe": len(U),
            "headlines_unique": len(uniq),
            "stock_headline_pairs": len(h),
            "title_chars_unique": int(uniq["title"].str.len().sum()),
            "zero_news_stocks": len(U) - h["stock_id"].nunique(),
            "zero_news_stocks_session_T_only": len(U) - h.loc[h["session_date"] == T, "stock_id"].nunique(),
            "market_level_touching_universe": len(touch),
            "headlines_unique_without_N_filter": len(uniq) + len(touch),
            "headlines_unique_trading_dates_only": int(uniq["date_day"].isin(trading).sum()),
            "headlines_unique_without_N_trading_dates_only":
                int(uniq["date_day"].isin(trading).sum()) + int(touch["date_day"].isin(trading).sum()),
        }
        for label, s in tok_by_aid.items():
            short = label.split(":")[-1].split("/")[-1]
            # one title per line: title tokens plus one separator each
            rec[f"tokens_unique[{short}]"] = int(s.reindex(uniq["aid"]).sum()) + len(uniq)
            rec[f"tokens_pairs[{short}]"] = int(s.reindex(h["aid"]).sum()) + len(h)
        rows.append(rec)
    return pd.DataFrame(rows)


def filter_stats(index, a, b, max_ids=news.MARKET_LEVEL_MAX_IDS):
    arts = pd.concat([index.stock_level[["date_day", "n_ids"]], index.market_level[["date_day", "n_ids"]]])
    sel = arts[arts["date_day"].between(pd.Timestamp(a), pd.Timestamp(b))]
    mk = sel["n_ids"] > max_ids
    return {"articles": int(len(sel)), "market_level": int(mk.sum()), "share_removed": float(mk.mean()),
            "median_tickers_removed": float(sel.loc[mk, "n_ids"].median()),
            "p90_tickers_removed": float(sel.loc[mk, "n_ids"].quantile(.9)),
            "median_tickers_all": float(sel["n_ids"].median())}


def late_ingestion(index, a, b):
    """Articles finlab ingested after the next trading day's 09:00 open (key_date is UTC)."""
    cols = ["aid", "date_day", "key_date", "n_ids"]
    arts = pd.concat([index.stock_level[cols], index.market_level[cols]])
    sel = arts[arts["date_day"].between(pd.Timestamp(a), pd.Timestamp(b))]
    cal = index.calendar
    pos = cal.searchsorted(sel["date_day"].to_numpy(), side="right")
    ok = pos < len(cal)
    sel = sel[ok]
    open_ = cal[pos[ok]] + pd.Timedelta(hours=9)
    late = (sel["key_date"] + pd.Timedelta(hours=8)).to_numpy() > open_.to_numpy()
    return {"articles": int(len(sel)), "ingested_after_next_open": int(late.sum()),
            "share": float(late.mean()),
            "by_ingest_day_top": sel.loc[late, "key_date"].dt.normalize().value_counts().head(5)
                                    .rename(lambda x: str(x.date())).to_dict(),
            "note": "descriptive only; not used as a filter (decided 2026-09-13)"}


def id_neighbour_offsets(index, periods, half=50):
    """Articles dated earlier/later than the median day of the `half` ids on each side."""
    arts = pd.concat([index.stock_level[["aid", "date_day"]], index.market_level[["aid", "date_day"]]])
    u = arts.sort_values("aid").reset_index(drop=True)
    days = u["date_day"].to_numpy().astype("datetime64[D]").astype(np.int64).astype(float)
    med = pd.Series(days).rolling(2 * half + 1, center=True, min_periods=half + 1).median().to_numpy()
    u["dev"] = days - med
    out = {}
    for name, (a, b) in periods.items():
        w = u[u["date_day"].between(pd.Timestamp(a), pd.Timestamp(b))]
        out[name] = {"articles": int(len(w)),
                     "earlier_by_1d": int((w["dev"] == -1).sum()),
                     "earlier_by_ge2d": int((w["dev"] <= -2).sum()),
                     "later_by_ge1d": int((w["dev"] >= 1).sum()),
                     "note": "1-day offsets include day-boundary noise; accepted as a known upper bound"}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(records.REPO_ROOT / "results" / "news_layer"))
    ap.add_argument("--history-start", default="2013-01-10")
    args = ap.parse_args()

    index, info = news.build()
    last = index.calendar[-1]
    periods = dict(PERIODS)
    # the universe needs a market value on or before T
    first = max(pd.Timestamp(args.history_start), panel.frame("market_value").index[0])
    periods["full_history"] = (str(first.date()), str(last.date()))

    print("loading tokenizers")
    toks = tokenizers()
    titles = index.stock_level.set_index("aid")["title"]
    tok_by_aid = {label: pd.Series(fn(titles.tolist()), index=titles.index) for label, fn in toks.items()}
    print("  tokenizers used:", list(toks))

    run_dir = Path(args.out) / f"stats_{_dt.datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    result = {"git": records.git_state(), "config_hash": info["config_hash"], "tokenizers": list(toks),
              "build": {k: info[k] for k in ("articles", "stock_level_articles", "market_level_articles",
                                             "articles_without_tickers")},
              "periods": {}}
    frames = []
    for name, (a, b) in periods.items():
        dates = [d for d in panel.decision_dates(a, b)]
        dates = [d for d in dates if index.calendar.get_loc(d) >= index.window]
        df = per_date(index, dates, tok_by_aid)
        df.insert(0, "period", name)
        frames.append(df)
        metric_cols = [c for c in df.columns if c not in ("period", "decision_date", "n_universe")]
        result["periods"][name] = {
            "range": [a, b], "decision_dates": len(df),
            "per_decision": {c: summarize(df[c]) for c in metric_cols},
            "zero_news_stocks_histogram": df["zero_news_stocks"].value_counts().sort_index().to_dict(),
            "zero_news_stocks_session_T_only_histogram":
                df["zero_news_stocks_session_T_only"].value_counts().sort_index().to_dict(),
            "n_filter": filter_stats(index, a, b),
        }
        if name != "full_history":
            result["periods"][name]["late_ingestion"] = late_ingestion(index, a, b)
        p = result["periods"][name]
        print(f"\n== {name} {a}..{b}: {len(df)} decision dates ==")
        for c in metric_cols:
            s = p["per_decision"][c]
            print(f"  {c:48s} mean {s['mean']:8.1f}  median {s['median']:8.1f}  p10 {s['p10']:8.1f}  "
                  f"p90 {s['p90']:8.1f}  min {s['min']:6.0f}  max {s['max']:8.0f}")
        print("  zero-news stocks (window) histogram:", p["zero_news_stocks_histogram"])
        print("  zero-news stocks (session T only) histogram:", p["zero_news_stocks_session_T_only_histogram"])
        print("  N filter:", p["n_filter"])
        if "late_ingestion" in p:
            print("  late ingestion:", p["late_ingestion"])
    result["id_neighbour_offsets"] = id_neighbour_offsets(index, PERIODS)
    print("\nid-neighbour offsets:", result["id_neighbour_offsets"])

    pd.concat(frames).to_csv(run_dir / "per_decision.csv", index=False)
    records.to_json(result, run_dir / "news_stats.json")
    print("\nwritten to", run_dir)


if __name__ == "__main__":
    main()
