"""Leakage self-check of the news layer, before any headline reaches a prompt.

    .venv/bin/python -m eval.twstock.news_selfcheck

1. synthetic      a hand-built calendar with a weekend, a mid-week holiday and
                  articles on every calendar day; each window is compared with a
                  hand-derived answer, not with the calendar arithmetic under test
2. no future      every trading day since 2013, all stocks and market-level: no
                  article dated after T, none dated on or before the session
                  before T-4; broken down by boundary type
3. completeness   all 2025 and test-window decision dates plus 200 random
                  earlier days: the index
                  equals a brute-force selection by calendar dates
4. closures       each market closure since 2024-06: the session before sees none
                  of the articles dated inside it, the session after sees all
5. dedup          the kept copy of every article carries its latest day
6. N filter       nothing with more than 50 tickers in per-stock input; nothing lost
7. halted stocks  universe members without an open price still get the market-
                  calendar window and every member is present in the output

Each criterion is printed before its numbers. Exit status 1 if any check fails.
"""

import argparse
import datetime as _dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import news, panel, records

RESULTS = []
# Periods checked since 2026-09-13: all of 2025 is a superset of the v9 §5.0 pre-cutoff
# window (2025-01-02..04-30), kept deliberately - a leakage check over more dates is stricter.
V8_PERIODS = {"pre_cutoff_2025": ("2025-01-02", "2025-12-31"),
              "test_window": ("2026-04-27", "2026-08-26")}


def check(name, ok, detail=""):
    RESULTS.append({"check": name, "pass": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


# --------------------------------------------------------------------------
# 1. synthetic
# --------------------------------------------------------------------------
SYN_CALENDAR = pd.DatetimeIndex([
    "2030-01-02", "2030-01-03", "2030-01-04",                 # Wed Thu Fri
    "2030-01-07", "2030-01-08",                               # Mon Tue
    "2030-01-11",                                             # Fri (Wed-Thu closed)
    "2030-01-14", "2030-01-15", "2030-01-16", "2030-01-17",   # Mon..Thu, end of data
])
# Derived by hand from the calendar above, not by the code under test:
# T -> (first calendar day, last calendar day) whose articles T may see.
SYN_EXPECTED_DAYS = {
    "2030-01-11": ("2030-01-03", "2030-01-11"),
    "2030-01-14": ("2030-01-04", "2030-01-14"),
    "2030-01-15": ("2030-01-05", "2030-01-15"),
    "2030-01-16": ("2030-01-08", "2030-01-16"),
    "2030-01-17": ("2030-01-09", "2030-01-17"),
}


def _url(aid, q=False):
    return f"https://news.cnyes.com/news/id/{aid}" + ("?exp=a" if q else "")


def synthetic():
    print("\n== 1. synthetic calendar: weekend, mid-week holiday, end of data ==")
    print("  criteria: each window equals the hand-derived article set exactly; a Friday")
    print("  never sees its weekend; articles after the last trading day are in no window;")
    print("  a non-trading T or too little history raises; dedup keeps the later day, ties by")
    print("  key_date; 51 tickers is market-level, 50 is not; repeated tickers count once")
    rows, day_aid = [], {}
    for i, d in enumerate(pd.date_range("2029-12-30", "2030-01-19")):
        aid = 1000 + i
        day_aid[d] = aid
        stamp = d + pd.Timedelta(hours=20, minutes=30) if i % 2 else d   # some after the close
        rows.append({"date": stamp, "key_date": d + pd.Timedelta(hours=16),
                     "title": f"daily {d.date()}", "url": _url(aid), "stock_ids": "A,H"})
    d14, d15 = pd.Timestamp("2030-01-14"), pd.Timestamp("2030-01-15")
    many = ",".join(["A"] + [f"S{k}" for k in range(1, 51)])      # 51 tickers
    fifty = ",".join(["A"] + [f"S{k}" for k in range(1, 50)])     # 50 tickers
    rows += [
        {"date": d14, "key_date": pd.Timestamp("2030-01-20 10:00"), "title": "earlier day, newer ingest",
         "url": _url(5000), "stock_ids": "A"},
        {"date": d15, "key_date": pd.Timestamp("2030-01-19 10:00"), "title": "later day",
         "url": _url(5000, q=True), "stock_ids": "A"},
        {"date": d14 + pd.Timedelta(hours=9), "key_date": pd.Timestamp("2030-01-14 16:00"),
         "title": "same day, older ingest", "url": _url(5001), "stock_ids": "A"},
        {"date": d14, "key_date": pd.Timestamp("2030-01-15 16:00"), "title": "same day, newer ingest",
         "url": _url(5001, q=True), "stock_ids": "B"},
        {"date": d14, "key_date": d14, "title": "51 tickers", "url": _url(6000), "stock_ids": many},
        {"date": d14, "key_date": d14, "title": "50 tickers", "url": _url(6001), "stock_ids": fifty},
        {"date": d14, "key_date": d14, "title": "repeated ticker", "url": _url(6002), "stock_ids": "A, A,A"},
    ]
    raw = pd.DataFrame(rows)
    arts, _ = news.clean(raw)
    arts = news.assign_sessions(arts, SYN_CALENDAR)
    stock, mkt = news.split_market_level(arts)
    idx = news.NewsIndex(stock, mkt, SYN_CALENDAR, window=5)

    by = arts.set_index("aid")
    check("dedup: a later calendar day beats a newer ingest",
          by.at[5000, "date_day"] == d15, f"kept {by.at[5000, 'date_day'].date()}")
    check("dedup: same day, the newer ingest (key_date) wins, not the later clock time",
          by.at[5001, "stock_ids"] == "B", f"kept stock_ids {by.at[5001, 'stock_ids']}")
    check("N filter: 51 tickers -> market-level, 50 -> per-stock",
          6000 in set(mkt["aid"]) and 6000 not in set(stock["aid"]) and 6001 in set(stock["aid"]))
    check("repeated tickers count once", by.at[6002, "n_ids"] == 1)

    all_ok, detail = True, []
    for T, (a, b) in SYN_EXPECTED_DAYS.items():
        days = pd.date_range(a, b)
        daily = {day_aid[d] for d in days}
        has14, has15 = d14 in days, d15 in days
        want = {
            "A": daily | ({6001, 6002} if has14 else set()) | ({5000} if has15 else set()),
            "H": daily,
            "B": {5001} if has14 else set(),
            "Z": set(),
        }
        got_frames = idx.headlines_by_stock(T, ["A", "B", "H", "Z"])
        got = {s: set(f["aid"]) for s, f in got_frames.items()}
        got_mkt = set(idx.market_headlines(T)["aid"])
        ok = got == want and got_mkt == ({6000} if has14 else set())
        all_ok &= ok
        detail.append(f"{T}:{'ok' if ok else 'MISMATCH'}")
        if not ok:
            print(f"    {T} want {want} got {got} market {got_mkt}")
    check("every window equals the hand-derived set (A, H, B, empty Z, market)", all_ok, " ".join(detail))

    fri = idx.headlines("2030-01-11")
    check("Friday 01-11 sees nothing dated on its weekend (01-12, 01-13)",
          not fri["date_day"].isin(pd.to_datetime(["2030-01-12", "2030-01-13"])).any()
          and fri["date"].max() < pd.Timestamp("2030-01-12"))
    tail = {day_aid[pd.Timestamp("2030-01-18")], day_aid[pd.Timestamp("2030-01-19")]}
    seen = set().union(*(set(idx.headlines(T)["aid"]) for T in SYN_EXPECTED_DAYS))
    check("articles dated after the last trading day appear in no window",
          not (tail & seen) and set(arts.loc[arts["session_pos"] < 0, "aid"]) == tail)

    def raises(T):
        try:
            idx.headlines(T)
            return False
        except ValueError:
            return True
    check("a non-trading T (holiday 01-09) raises", raises("2030-01-09"))
    check("T without 5 sessions of history (01-08) raises", raises("2030-01-08"))


# --------------------------------------------------------------------------
# 2. no future, exhaustive over real data
# --------------------------------------------------------------------------
def boundary_type(cal, pos):
    nxt = (cal[pos + 1] - cal[pos]).days if pos + 1 < len(cal) else None
    prv = (cal[pos] - cal[pos - 1]).days
    types = []
    if nxt is None:
        types.append("last_session_in_data")
    elif nxt == 3 and cal[pos].dayofweek == 4:
        types.append("before_weekend")
    elif nxt > 1:
        types.append("before_closure")
    if prv >= 4:
        types.append("after_multi_day_closure")
    elif prv == 3 and cal[pos].dayofweek == 0:
        types.append("after_weekend")
    return types or ["plain"]


def no_future(index, start="2013-01-01"):
    print("\n== 2. no future: every trading day, every stock, market-level too ==")
    print("  criteria: over all T, 0 articles with calendar date > T, 0 with a timestamp")
    print("  at or after T+1 00:00, 0 dated on or before the session before T-4;")
    print("  each boundary type is reported and must have 0 future articles")
    cal = index.calendar
    counts = {"rows": 0, "future_day": 0, "future_stamp": 0, "too_old": 0, "dates": 0}
    by_type = {}
    for pos, T in enumerate(cal):
        if T < pd.Timestamp(start):
            continue
        try:
            lo, hi = index.session_range(T)
        except ValueError:
            continue
        after = cal[lo - 1]
        rows = pd.concat([index.headlines(T)[["date", "date_day"]],
                          index.market_headlines(T)[["date", "date_day"]]])
        fut = int((rows["date_day"] > T).sum())
        counts["rows"] += len(rows)
        counts["dates"] += 1
        counts["future_day"] += fut
        counts["future_stamp"] += int((rows["date"] >= T + pd.Timedelta(days=1)).sum())
        counts["too_old"] += int((rows["date_day"] <= after).sum())
        for t in boundary_type(cal, pos):
            b = by_type.setdefault(t, {"dates": 0, "rows": 0, "future": 0})
            b["dates"] += 1
            b["rows"] += len(rows)
            b["future"] += fut
    check("no article dated after T", counts["future_day"] == 0 and counts["future_stamp"] == 0,
          f"(dates {counts['dates']}, article-rows {counts['rows']}, future by day "
          f"{counts['future_day']}, by timestamp {counts['future_stamp']})")
    check("no article older than the window", counts["too_old"] == 0, f"({counts['too_old']})")
    for t, b in sorted(by_type.items()):
        check(f"boundary {t}: no future article", b["future"] == 0 and b["dates"] > 0,
              f"(dates {b['dates']}, rows {b['rows']}, future {b['future']})")
    return {"totals": counts, "by_boundary": by_type}


# --------------------------------------------------------------------------
# 3. completeness against brute force
# --------------------------------------------------------------------------
def brute_pairs(stock_level, cal, T, universe, window):
    pos = cal.get_loc(T)
    after = cal[pos - window]
    sel = stock_level[(stock_level["date_day"] > after) & (stock_level["date_day"] <= T)]
    members = set(universe)
    return {(sid, int(aid)) for aid, ids in zip(sel["aid"], sel["ids"]) for sid in ids if sid in members}


def completeness(index, n_random=200, seed=20260913):
    print("\n== 3. completeness: index == brute-force selection by calendar dates ==")
    print("  criterion: identical (stock_id, article id) sets on 100% of dates - all")
    print(f"  decision dates (2025, test window) plus {n_random} random trading days 2013-2024")
    cal = index.calendar
    dates = []
    for a, b in V8_PERIODS.values():
        dates += list(panel.decision_dates(a, b))
    # the universe needs a market value on or before T; etl:market_value starts in 2013
    first = max(pd.Timestamp("2013-02-01"), panel.frame("market_value").index[0])
    pool = [d for d in panel.decision_dates(first, "2024-12-31")]
    rng = np.random.default_rng(seed)
    dates += list(pd.DatetimeIndex(rng.choice(pool, size=n_random, replace=False)))
    mism, pairs = [], 0
    for T in dates:
        U = panel.universe(T)
        got = {(s, int(a)) for s, a in zip(*index.headlines(T, U)[["stock_id", "aid"]].to_numpy().T)}
        want = brute_pairs(index.stock_level, cal, T, U, index.window)
        pairs += len(want)
        if got != want:
            mism.append((str(T.date()), len(got - want), len(want - got)))
    check("index equals brute force", not mism,
          f"(dates {len(dates)}, pairs {pairs}, mismatching dates {len(mism)} {mism[:5]})")
    return {"dates": len(dates), "pairs": pairs, "mismatches": mism}


# --------------------------------------------------------------------------
# 4. named closures
# --------------------------------------------------------------------------
def closures(index, start="2024-06-01"):
    print("\n== 4. market closures since 2024-06: before sees none, after sees all ==")
    print("  criterion: for every gap of >= 4 calendar days between sessions, articles dated")
    print("  inside the gap (per-stock with tickers, and market-level) number 0 in the window")
    print("  of the session before and all of them in the window of the session after")
    cal = index.calendar
    arts = pd.concat([index.stock_level[index.stock_level["n_ids"] > 0][["aid", "date_day"]],
                      index.market_level[["aid", "date_day"]]])
    out, ok = [], True
    for pos in range(1, len(cal)):
        prev, T = cal[pos - 1], cal[pos]
        if T < pd.Timestamp(start) or (T - prev).days < 4:
            continue
        inside = set(arts.loc[(arts["date_day"] > prev) & (arts["date_day"] < T), "aid"])
        seen_after = set(index.headlines(T)["aid"]) | set(index.market_headlines(T)["aid"])
        seen_before = set(index.headlines(prev)["aid"]) | set(index.market_headlines(prev)["aid"])
        row = {"before": str(prev.date()), "after": str(T.date()), "gap_days": (T - prev).days,
               "articles_in_gap": len(inside), "seen_by_after": len(inside & seen_after),
               "seen_by_before": len(inside & seen_before)}
        ok &= row["seen_by_after"] == row["articles_in_gap"] and row["seen_by_before"] == 0
        out.append(row)
        print(f"    {row['before']} -> {row['after']}  gap {row['gap_days']}d  in gap {row['articles_in_gap']:4d}  "
              f"seen after {row['seen_by_after']:4d}  seen before {row['seen_by_before']}")
    check("closures: before sees 0, after sees all", ok and len(out) > 0, f"({len(out)} closures)")
    return out


# --------------------------------------------------------------------------
# 5-7
# --------------------------------------------------------------------------
def dedup_direction(raw, index):
    print("\n== 5. dedup keeps the latest day of every article ==")
    print("  criterion: 0 kept articles whose day is earlier than any copy's day;")
    print("  article 6437493 (copies dated 2026-04-29 and 04-30) kept as 2026-04-30")
    aid = pd.to_numeric(raw["url"].astype(str).str.extract(news.AID_PATTERN, expand=False))
    latest = pd.to_datetime(raw["date"]).dt.normalize().groupby(aid.to_numpy()).max()
    kept = pd.concat([index.stock_level, index.market_level]).set_index("aid")["date_day"]
    bad = int((kept < latest.reindex(kept.index)).sum())
    one = kept.get(6437493)
    check("kept day is the latest copy's day", bad == 0 and len(kept) == len(latest),
          f"(articles {len(kept)}, raw ids {len(latest)}, violations {bad})")
    check("6437493 kept as 2026-04-30", one == pd.Timestamp("2026-04-30"), f"(kept {one})")
    return {"violations": bad, "6437493": one}


def n_filter(index, info):
    print("\n== 6. N filter ==")
    print("  criteria: per-stock max tickers <= 50; market-level min tickers >= 51;")
    print("  per-stock + market-level == deduplicated articles")
    s_max = int(index.stock_level["n_ids"].max())
    m_min = int(index.market_level["n_ids"].min())
    total = len(index.stock_level) + len(index.market_level)
    check("per-stock never above 50", s_max <= 50, f"(max {s_max})")
    check("market-level always above 50", m_min >= 51, f"(min {m_min})")
    check("nothing lost in the split", total == info["articles"], f"({total} vs {info['articles']})")


def halted(index):
    print("\n== 7. universe members without an open price on T ==")
    print("  criteria: every universe member is a key of the per-stock output on every checked date;")
    print("  halted members get no article dated after T (the count of halted pairs is reported)")
    ao = panel.frame("adj_open")
    n_pairs, missing_keys, fut, examples = 0, 0, 0, []
    for a, b in V8_PERIODS.values():
        for T in panel.decision_dates(a, b):
            U = panel.universe(T)
            by = index.headlines_by_stock(T, U)
            missing_keys += len(set(U) - set(by))
            row = ao.loc[T].reindex(U)
            for s in row.index[row.isna()]:
                n_pairs += 1
                fut += int((by[s]["date_day"] > T).sum())
                if len(examples) < 10:
                    examples.append((str(T.date()), s, len(by[s])))
    check("every member present in the output", missing_keys == 0, f"(missing {missing_keys})")
    check("halted members: no future article", fut == 0,
          f"(halted member-days {n_pairs}, examples {examples}; synthetic stock H covers the logic)")
    return {"halted_member_days": n_pairs, "examples": examples}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(records.REPO_ROOT / "results" / "news_layer"))
    args = ap.parse_args()

    synthetic()

    cfg = news.load_config()
    raw = news.raw_news(cfg)
    articles, dedup = news.clean(raw)
    cal = panel.trading_days()
    articles = news.assign_sessions(articles, cal)
    stock, mkt = news.split_market_level(articles, cfg["market_level_max_ids"])
    index = news.NewsIndex(stock, mkt, cal, cfg["window_trading_days"])
    info = {"articles": len(articles), "config_hash": news.config_hash(cfg), "dedup": dedup}
    print(f"\nreal data: raw rows {len(raw)}, articles {len(articles)}, per-stock {len(stock)}, "
          f"market-level {len(mkt)}, calendar {cal[0].date()}..{cal[-1].date()}")

    out = {
        "no_future": no_future(index),
        "completeness": completeness(index),
        "closures": closures(index),
        "dedup": dedup_direction(raw, index),
    }
    n_filter(index, info)
    out["halted"] = halted(index)

    run_dir = Path(args.out) / f"selfcheck_{_dt.datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    records.to_json({"git": records.git_state(), "config": cfg, **info, "results": out,
                     "checks": RESULTS}, run_dir / "news_selfcheck.json")
    failed = [r["check"] for r in RESULTS if not r["pass"]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed; written to {run_dir}")
    if failed:
        print("FAILED:", *failed, sep="\n  ")
        sys.exit(1)


if __name__ == "__main__":
    main()
