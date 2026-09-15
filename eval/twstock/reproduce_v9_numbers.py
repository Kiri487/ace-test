"""Recompute v9 numbers that had no saved source when v9 was checked (2026-09-14). No LLM.

    .venv/bin/python -m eval.twstock.reproduce_v9_numbers

Writes results/v9_numbers/<time>/numbers.json. Every number is recomputed from inputs; where
an earlier output exists it is stored beside the recomputation, and the value v9 quotes is
stored as `v9_text` so any mismatch is visible. The thesis cites only numbers with a source.

1. Known-answer factors on the test window 2026-04-27..08-26, h=40: csIC Newey-West t (lags
   h-1, 1.5h, 2h) and tsIC moving-block bootstrap t (block h, 500 draws, seed 0), for momentum
   and value. Computed with the ic primitives, not ic.summarize, because summarize now blanks t
   when N_eff < 20 - and v9 §6.1 quotes these values as the reason for that rule.
   v9 quotes "NW t -13.4, bootstrap t +50.6" as if from one series: -13.4 is momentum at lag
   2h, +50.6 is value's bootstrap t. Both are recorded so the attribution can be corrected.
2. momentum_20d tsIC mean on the test window, h = 5/10/20/40.
   Inputs for 1-2: known_answer_20260913_173748/<factor>/decisions.parquet (alpha as of the
   price cache used on 2026-09-13).
3. Universe members with no headline whose session is T itself (news_stats definition
   zero_news_stocks_session_T_only), median over the test window's decision dates.
4. Distinct key_date calendar days in the raw tw_news_cnyes table (UTC day and Taipei day),
   and the share of rows in the three largest ingestion days.
5. Mean unique headlines per decision on the test window when headlines dated on a
   non-trading day are dropped (news_stats headlines_unique_trading_dates_only).
"""

import datetime as _dt
import json

import numpy as np
import pandas as pd

from . import ic, news, news_stats, panel, records

KNOWN_ANSWER = records.REPO_ROOT / "results" / "eval_layer" / "known_answer_20260913_173748"
NEWS_STATS = records.REPO_ROOT / "results" / "news_layer" / "stats_20260913_223503" / "news_stats.json"
WINDOW = ("2026-04-27", "2026-08-26")
HORIZONS = (5, 10, 20, 40)
FACTORS = ("momentum_20d", "value_earnings_yield")
V9_TEXT = {
    "h40_nw_t": -13.4, "h40_bootstrap_t": 50.6,
    "ts_mean_test_window_momentum": {5: -0.103, 10: -0.119, 20: -0.171, 40: -0.335},
    "zero_news_session_T_only_median": 12,
    "key_date_distinct_days": 518, "key_date_top3_share": 0.92,
    "headlines_trading_dates_only_mean": 212.0,
}


def factor_numbers(name):
    dec, _, _ = records.read_run(KNOWN_ANSWER / name, "offline_check")
    dates = panel.decision_dates(*WINDOW)
    saved = json.loads((KNOWN_ANSWER / "summary.json").read_text(encoding="utf-8"))
    saved = saved["factors"][name]["test_window"]
    S = ic.to_matrix(dec, "score", dates)
    out = {}
    for h in HORIZONS:
        A = ic.to_matrix(dec, f"alpha_h{h}", dates)
        matured = A.notna().any(axis=1)
        Sm, Am = S.loc[matured], A.loc[matured]
        cs, _ = ic.cs_ic(Sm, Am)
        ts, _ = ic.ts_ic(Sm, Am)
        boot = ic.ts_ic_bootstrap(Sm, Am, block=h, n_boot=500, seed=0)
        ts_mean = float(ts.mean())
        se = float(np.nanstd(boot, ddof=1))
        sv = saved[str(h)]["test_window"]
        out[h] = {
            "matured_dates": int(matured.sum()),
            "n_eff": int(cs.notna().sum()) / h,
            "cs_mean": float(cs.mean()),
            "cs_t_newey_west": {k: ic.newey_west(cs, lag)["t"] for k, lag in ic.nw_lags(h).items()},
            "ts_mean": ts_mean,
            "ts_t_block_bootstrap": ts_mean / se if se > 0 else None,
            "saved_2026_09_13": {
                "matured_dates": sv["n_matured_dates"],
                "cs_t_newey_west": {k: v["t"] for k, v in sv["cs"]["nw_variants"].items()},
                "ts_mean": sv["ts"]["mean"], "ts_t_block_bootstrap": sv["ts"]["t_block_bootstrap"]},
        }
    return out


def news_numbers():
    cfg = news.load_config()
    index, _ = news.build(cfg)
    dates = [d for d in panel.decision_dates(*WINDOW) if index.calendar.get_loc(d) >= index.window]
    df = news_stats.per_date(index, dates, {})
    saved = json.loads(NEWS_STATS.read_text(encoding="utf-8"))["periods"]["test_window"]["per_decision"]

    raw = news.raw_news(cfg)
    kd = pd.to_datetime(raw["key_date"])
    utc_day = kd.dt.normalize()
    tpe_day = (kd + pd.Timedelta(hours=8)).dt.normalize()
    top3 = utc_day.value_counts().head(3)
    return {
        "decision_dates": len(df),
        "zero_news_session_T_only": {
            "median": float(df["zero_news_stocks_session_T_only"].median()),
            "mean": float(df["zero_news_stocks_session_T_only"].mean()),
            "saved_median": saved["zero_news_stocks_session_T_only"]["median"]},
        "headlines_trading_dates_only": {
            "mean": float(df["headlines_unique_trading_dates_only"].mean()),
            "all_dates_mean": float(df["headlines_unique"].mean()),
            "saved_mean": saved["headlines_unique_trading_dates_only"]["mean"]},
        "key_date": {
            "rows": int(len(raw)),
            "distinct_days_utc": int(utc_day.nunique()),
            "distinct_days_taipei": int(tpe_day.nunique()),
            "top3_days_utc": {str(k.date()): int(v) for k, v in top3.items()},
            "top3_share": float(top3.sum() / len(raw)),
        },
    }


def main():
    run_dir = records.REPO_ROOT / "results" / "v9_numbers" / f"run_{_dt.datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    out = {"git": records.git_state(), "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
           "inputs": {"known_answer": str(KNOWN_ANSWER), "news_stats": str(NEWS_STATS)},
           "v9_text": V9_TEXT, "factors": {f: factor_numbers(f) for f in FACTORS}, "news": news_numbers()}
    records.to_json(out, run_dir / "numbers.json")
    for f in FACTORS:
        x = out["factors"][f][40]
        print(f"{f} h=40 test window: matured {x['matured_dates']}, n_eff {x['n_eff']:.2f}, NW t "
              + ", ".join(f"{k} {v:+.2f}" for k, v in x["cs_t_newey_west"].items())
              + f", bootstrap t {x['ts_t_block_bootstrap']:+.2f}")
    m = out["factors"]["momentum_20d"]
    print("momentum tsIC, test window:", {h: round(m[h]["ts_mean"], 4) for h in HORIZONS},
          "saved:", {h: round(m[h]["saved_2026_09_13"]["ts_mean"], 4) for h in HORIZONS})
    n = out["news"]
    print("zero-news, session T only:", n["zero_news_session_T_only"])
    print("headlines, trading dates only:", n["headlines_trading_dates_only"])
    print("key_date:", n["key_date"])
    print("written to", run_dir)


if __name__ == "__main__":
    main()
