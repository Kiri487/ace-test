"""Self-check of the evaluation layer, before any factor or LLM number is read.

    .venv/bin/python -m eval.twstock.eval_selfcheck

1. synthetic prices   α formula, benchmark, NaN handling, entry and maturity dates
2. synthetic IC       Spearman against scipy; Newey-West against its analytic
                      expectation
3. alignment, real α  score = α            -> IC must be exactly 1
                      score shifted k days -> IC must fall as the overlap shrinks
                      score = α already realized at t (T3b)
                                           -> descriptive only; T3a is the overlap
                                              test (see eval.twstock.tsic_placebo)
4. ex-dividend        adj_open must not book a dividend as an overnight loss

Each criterion is printed before its numbers. Exit status 1 if any check fails.
"""

import argparse
import datetime as _dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import alpha, ic, panel, records
from .panel import HORIZONS

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append({"check": name, "pass": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


# --------------------------------------------------------------------------
# 1. synthetic prices
# --------------------------------------------------------------------------
def synthetic_alpha():
    print("\n== 1. synthetic prices: alpha formula, benchmark, NaN, dates ==")
    n, h = 60, 5
    idx = pd.bdate_range("2030-01-01", periods=n)
    p = np.arange(n)
    opens = pd.DataFrame({"A": 100 * 1.01 ** p, "B": np.full(n, 50.0), "C": 20 * 1.02 ** p},
                         index=idx)
    opens.iloc[10, 2] = np.nan                      # C has no open on bar 10
    univ = {d: ["A", "B", "C"] for d in idx}
    pnl = alpha.build_alpha_panel(idx, opens, univ, horizons=(h,))

    def get(t, s, col):
        return pnl.loc[(pnl.decision_date == idx[t]) & (pnl.stock_id == s), col].iloc[0]

    rA, rC = 1.01 ** h - 1, 1.02 ** h - 1
    check("r = open(t+1+h)/open(t+1) - 1",
          np.isclose(get(0, "A", "r_h5"), rA) and np.isclose(get(0, "B", "r_h5"), 0)
          and np.isclose(get(0, "C", "r_h5"), rC),
          f"A {get(0, 'A', 'r_h5'):.6f} want {rA:.6f}")
    bench0 = (rA + rC) / 3
    check("benchmark = equal-weight mean over members",
          np.isclose(get(0, "A", "bench_r_h5"), bench0) and get(0, "A", "bench_n_h5") == 3)
    check("alpha = r - benchmark", np.isclose(get(0, "A", "alpha_h5"), rA - bench0))
    check("a missing entry or exit open gives NaN, never a filled value",
          np.isnan(get(9, "C", "r_h5")) and np.isnan(get(4, "C", "r_h5"))
          and not np.isnan(get(3, "C", "r_h5")) and not np.isnan(get(10, "C", "r_h5")),
          "C: t=9 (entry bar 10) and t=4 (exit bar 10) NaN; t=3, t=10 valid")
    check("benchmark skips NaN members and records how many it used",
          get(4, "A", "bench_n_h5") == 2 and np.isclose(get(4, "A", "bench_r_h5"), rA / 2))
    check("entry_date = t+1, mature_date = t+1+h",
          get(0, "A", "entry_date") == idx[1] and get(0, "A", "mature_date_h5") == idx[6])
    last = n - 2 - h
    check("a window running past the data stays NaN",
          not np.isnan(get(last, "A", "r_h5")) and np.isnan(get(last + 1, "A", "r_h5"))
          and pd.isna(get(last + 1, "A", "mature_date_h5")),
          f"last valid t = {last}")


# --------------------------------------------------------------------------
# 2. synthetic IC machinery
# --------------------------------------------------------------------------
def synthetic_ic():
    from scipy.stats import spearmanr
    print("\n== 2. synthetic IC: Spearman vs scipy, Newey-West vs theory ==")
    rng = np.random.default_rng(7)
    S = pd.DataFrame(np.round(rng.normal(size=(40, 15)), 1))       # rounding makes ties
    A = pd.DataFrame(0.4 * S.to_numpy() + rng.normal(size=(40, 15)))
    S = S.mask(rng.random(S.shape) < 0.15)
    A = A.mask(rng.random(A.shape) < 0.15)

    cs, _ = ic.cs_ic(S, A, min_pairs=3)
    ts, _ = ic.ts_ic(S, A, min_obs=3)
    worst = 0.0
    for i in range(S.shape[0]):
        m = S.iloc[i].notna() & A.iloc[i].notna()
        worst = max(worst, abs(spearmanr(S.iloc[i][m], A.iloc[i][m])[0] - cs.iloc[i]))
    for j in range(S.shape[1]):
        m = S.iloc[:, j].notna() & A.iloc[:, j].notna()
        worst = max(worst, abs(spearmanr(S.iloc[:, j][m], A.iloc[:, j][m])[0] - ts.iloc[j]))
    check("csIC/tsIC equal scipy.spearmanr with ties and NaN", worst < 1e-10,
          f"max abs diff {worst:.2e}")

    x = rng.normal(size=500)
    nw0 = ic.newey_west(x, 0)
    check("Newey-West lag 0 equals the iid standard error",
          np.isclose(nw0["se"], x.std(ddof=0) / np.sqrt(len(x))))

    q, n, sims = 9, 2000, 400            # MA(9) with flat weights = overlapping 10-day sums
    means, ses = [], []
    for _ in range(sims):
        e = rng.normal(size=n + q)
        xs = np.convolve(e, np.ones(q + 1), mode="valid")
        r = ic.newey_west(xs, q)
        means.append(r["mean"])
        ses.append(r["se"])
    true_sd = (q + 1) / np.sqrt(n)
    bartlett_lrv = (q + 1) + 2 * sum((1 - k / (q + 1)) * (q + 1 - k) for k in range(1, q + 1))
    bartlett_sd = np.sqrt(bartlett_lrv / n)
    check("Newey-West matches its analytic Bartlett expectation (within 5%)",
          abs(np.mean(ses) / bartlett_sd - 1) < 0.05,
          f"mean NW se {np.mean(ses):.4f}, Bartlett expectation {bartlett_sd:.4f}")
    check("the simulation itself is calibrated (within 12%)",
          abs(np.std(means, ddof=1) / true_sd - 1) < 0.12,
          f"empirical sd of mean {np.std(means, ddof=1):.4f}, true {true_sd:.4f}")
    ratio = bartlett_sd / true_sd
    print(f"  NOTE: for overlapping h-day windows, NW with lag h-1 recovers only "
          f"{ratio:.0%} of the true standard error at h=10 (property of the Bartlett "
          f"kernel, not a bug). t statistics are overstated by ~{1 / ratio - 1:.0%}.")
    return {"nw_bartlett_to_true_se_ratio_h10": ratio}


# --------------------------------------------------------------------------
# 3. alignment on real alpha
# --------------------------------------------------------------------------
T3_BOUND = 0.05


def alignment(dates, U, adj_open):
    print("\n== 3. alignment on real alpha ==")
    print("  criteria, fixed before the numbers:")
    print("   T1  score = alpha                -> every csIC and tsIC equals 1 (|x-1| < 1e-9)")
    print("   T2  score = alpha shifted k days -> mean csIC strictly decreasing for k = 0..h")
    print("   T3a alpha matured at t, rebuilt from open(t)/open(t-h) via mature_date,"
          " equals the stored alpha shifted h+1 rows (max diff < 1e-12, same NaN pattern)")
    print("   T3b score = that already-realized alpha: DESCRIPTIVE, not a check.")
    print(f"       On 2026-09-13 the pre-declared |IC| < {T3_BOUND} failed at h=20 and h=40. The")
    print("       criterion was not relaxed. An i.i.d. placebo through the same code gives")
    print("       csIC ~ 0 (eval.twstock.tsic_placebo), so the deviation is serial dependence")
    print("       in real alpha plus tsIC finite-sample bias, not window overlap; T3a is the")
    print("       overlap test.")
    pnl = alpha.build_alpha_panel(dates, adj_open, U)
    idx = adj_open.index
    pos = pd.Series(np.arange(len(idx)), index=idx)
    cols = sorted(set().union(*(set(U[t]) for t in dates)))
    opens = adj_open.reindex(columns=cols)
    out = {}

    for h in HORIZONS:
        A = ic.to_matrix(pnl, f"alpha_h{h}", dates)
        cs, _ = ic.cs_ic(A, A)
        ts, _ = ic.ts_ic(A, A)
        dev = max(float((cs.dropna() - 1).abs().max()), float((ts.dropna() - 1).abs().max()))
        check(f"h={h:2d} T1 score=alpha gives IC=1", dev < 1e-9,
              f"(dates {int(cs.notna().sum())}, stocks {int(ts.notna().sum())}, max dev {dev:.1e})")

        ks = sorted({-1, 0, 1, 2, h // 2, h - 1, h, h + 1, 2 * h})
        curve = {}
        print(f"   h={h:2d}   k   overlap   mean csIC   mean tsIC")
        for k in ks:
            S = A.shift(k)
            c, _ = ic.cs_ic(S, A)
            t_, _ = ic.ts_ic(S, A)
            overlap = max(0.0, (h - abs(k)) / h)
            curve[k] = {"overlap": overlap, "cs": float(c.mean()), "ts": float(t_.mean())}
            print(f"        {k:4d}   {overlap:6.2f}   {c.mean():+9.4f}   {t_.mean():+9.4f}")
        seq = [curve[k]["cs"] for k in ks if 0 <= k <= h]
        check(f"h={h:2d} T2 IC falls as the shift grows",
              all(a > b for a, b in zip(seq, seq[1:])) and curve[1]["cs"] < 0.999,
              f"csIC k=0..h: {[round(v, 3) for v in seq]}")

        realized = pd.DataFrame(np.nan, index=A.index, columns=A.columns)
        sub = pnl[["decision_date", "stock_id", f"mature_date_h{h}"]].dropna()
        for d0, g in sub.groupby("decision_date"):
            t = g[f"mature_date_h{h}"].iloc[0]
            if t not in realized.index:
                continue
            pt = int(pos[t])
            members = list(g["stock_id"])
            r = opens.iloc[pt].reindex(members) / opens.iloc[pt - h].reindex(members) - 1
            realized.loc[t, members] = (r - r.mean()).to_numpy()
        shifted = A.shift(h + 1)
        both = realized.notna() & shifted.notna()
        diff = float((realized - shifted).abs().where(both).max().max())
        mismatch = int((realized.notna() ^ shifted.notna()).sum().sum())
        check(f"h={h:2d} T3a matured alpha = open(t)/open(t-h) via mature_date",
              diff < 1e-12 and mismatch == 0,
              f"(max diff {diff:.1e}, NaN-pattern mismatches {mismatch}, cells {int(both.sum().sum())})")

        c, _ = ic.cs_ic(shifted, A)
        t_, _ = ic.ts_ic(shifted, A)
        nw = ic.newey_west(c, h - 1)
        print(f"  [DESCRIPTIVE] h={h:2d} T3b already-realized alpha as score  "
              f"(csIC {c.mean():+.4f}, NW t {nw['t']:+.2f}; tsIC {t_.mean():+.4f})")
        out[h] = {"T1_max_dev": dev, "T2_curve": curve, "T3a_max_diff": diff,
                  "T3a_nan_mismatch": mismatch, "T3b_cs": float(c.mean()),
                  "T3b_cs_nw_t": nw["t"], "T3b_ts": float(t_.mean())}
    return out, pnl


# --------------------------------------------------------------------------
# 4. ex-dividend handling of adj_open
# --------------------------------------------------------------------------
def exdiv(dates, U, show_stock="2330"):
    print("\n== 4. ex-dividend: adj_open must not show the dividend as a loss ==")
    print("  an ex-date is where adj_close/raw_close changes by more than 0.1%;")
    print("  implied yield = 1 - c(t-1)/c(t); gaps are measured against the median")
    print("  overnight gap of the other universe members that day. criteria:")
    print("   a) adj_open and adj_close carry the same factor: p99 |c_open/c_close - 1| < 0.005")
    print("   b) raw excess gap vs implied yield: OLS slope in [-1.3, -0.7]"
          " (the raw price really drops by the dividend)")
    print("   c) adj excess gap vs implied yield: OLS slope in [-0.3, 0.3] (adjusted price does not)")
    print("   d) |median adj excess gap| < 0.25 x median implied yield")
    cols = sorted(set().union(*(set(U[d]) for d in dates)))
    ao, ac = panel.frame("adj_open"), panel.frame("adj_close")
    lo = ao.index.searchsorted(dates[0]) - 1
    hi = ao.index.searchsorted(dates[-1], side="right")
    rows = ao.index[lo:hi]
    AO, AC = ao.reindex(index=rows, columns=cols), ac.reindex(index=rows, columns=cols)
    RO = panel.frame("raw_open").reindex(index=rows, columns=cols)
    RC = panel.frame("raw_close").reindex(index=rows, columns=cols)

    member = pd.DataFrame(False, index=rows, columns=cols)
    for d in dates:
        if d in member.index:
            member.loc[d, U[d]] = True
    member_prev = member.shift(1, fill_value=False)

    c_close, c_open = AC / RC, AO / RO
    consist = (c_open / c_close - 1).abs().where(member).stack()
    p99 = float(consist.quantile(0.99))
    check("a) same adjustment factor for open and close", p99 < 0.005,
          f"(median {consist.median():.1e}, p99 {p99:.1e}, n {len(consist)})")

    jump = c_close / c_close.shift(1) - 1
    event = (jump.abs() > 1e-3) & member_prev
    raw_gap = RO / RC.shift(1) - 1
    adj_gap = AO / AC.shift(1) - 1
    mkt = adj_gap.where(member_prev & ~event).median(axis=1)
    implied = 1 - c_close.shift(1) / c_close

    r_i, c_i = np.where(event.to_numpy())
    ev = pd.DataFrame({
        "date": rows[r_i], "stock_id": np.array(cols)[c_i],
        "raw_close_prev": RC.shift(1).to_numpy()[r_i, c_i],
        "raw_open": RO.to_numpy()[r_i, c_i],
        "raw_gap": raw_gap.to_numpy()[r_i, c_i],
        "adj_gap": adj_gap.to_numpy()[r_i, c_i],
        "market_gap": mkt.to_numpy()[r_i],
        "implied_yield": implied.to_numpy()[r_i, c_i],
    })
    ev["raw_excess"] = ev["raw_gap"] - ev["market_gap"]
    ev["adj_excess"] = ev["adj_gap"] - ev["market_gap"]
    ev["implied_cash_per_share"] = ev["raw_close_prev"] * ev["implied_yield"]
    use = ev[(ev["implied_yield"].abs() < 0.3) & ev["raw_excess"].notna() & ev["adj_excess"].notna()]

    slope_raw = float(np.polyfit(use["implied_yield"], use["raw_excess"], 1)[0])
    slope_adj = float(np.polyfit(use["implied_yield"], use["adj_excess"], 1)[0])
    med_y = float(use["implied_yield"].median())
    med_adj = float(use["adj_excess"].median())
    print(f"  events {len(ev)} (used {len(use)}), median implied yield {med_y:.4f}, "
          f"median raw excess gap {use['raw_excess'].median():+.4f}, "
          f"median adj excess gap {med_adj:+.4f}")
    check("b) raw open drops by the implied dividend", -1.3 < slope_raw < -0.7,
          f"(slope {slope_raw:+.3f})")
    check("c) adj open shows no dividend drop", -0.3 < slope_adj < 0.3, f"(slope {slope_adj:+.3f})")
    check("d) median adj excess gap on ex-dates near 0", abs(med_adj) < 0.25 * med_y,
          f"(|{med_adj:+.4f}| vs {0.25 * med_y:.4f})")

    one = ev[ev["stock_id"] == show_stock]
    print(f"  {show_stock} ex-dates in range:")
    for _, r in one.iterrows():
        print(f"    {r['date'].date()}  prev raw close {r['raw_close_prev']:8.2f}  raw open "
              f"{r['raw_open']:8.2f}  raw gap {r['raw_gap']:+.4f}  adj gap {r['adj_gap']:+.4f}  "
              f"market gap {r['market_gap']:+.4f}  implied cash/share {r['implied_cash_per_share']:.2f}")
    return {"n_events": int(len(ev)), "n_used": int(len(use)), "slope_raw": slope_raw,
            "slope_adj": slope_adj, "median_implied_yield": med_y,
            "median_adj_excess": med_adj, "p99_factor_mismatch": p99}, ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-01-02")
    ap.add_argument("--end", default="2026-08-26")
    ap.add_argument("--window-start", default="2026-04-27")
    ap.add_argument("--window-end", default="2026-08-26")
    ap.add_argument("--out", default=str(records.REPO_ROOT / "results" / "eval_layer"))
    args = ap.parse_args()

    for name in panel.DATASETS:
        panel.frame(name)
    last = {n: str(panel.frame(n).index[-1].date()) for n in panel.DATASETS}
    print("data last dates:", last)

    dates = panel.decision_dates(args.start, args.end)
    all_days = panel.decision_dates(args.start, panel.trading_days()[-1])
    U = panel.universes(all_days)
    window = panel.decision_dates(args.window_start, args.window_end)
    idx = panel.trading_days()
    matured = {h: int(sum(idx.get_loc(d) + 1 + h < len(idx) for d in window)) for h in HORIZONS}
    print(f"test window {args.window_start}..{args.window_end}: {len(window)} decision dates; "
          f"matured per h: {matured}")
    print(f"validation period {args.start}..{args.end}: {len(dates)} decision dates")

    synthetic_alpha()
    nw_note = synthetic_ic()
    align, pnl = alignment(dates, {d: U[d] for d in dates}, panel.frame("adj_open"))
    ex, ev = exdiv(all_days, U)

    run_dir = Path(args.out) / f"selfcheck_{_dt.datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ev.to_csv(run_dir / "exdiv_events.csv", index=False)
    records.to_json({
        "git": records.git_state(), "data_last_dates": last,
        "test_window": {"start": args.window_start, "end": args.window_end,
                        "n_decision_dates": len(window), "matured_per_h": matured},
        "validation_period": {"start": args.start, "end": args.end, "n_decision_dates": len(dates)},
        "missing": alpha.missing_report(pnl),
        "checks": RESULTS, "nw": nw_note, "alignment": align, "exdiv": ex,
    }, run_dir / "selfcheck.json")

    failed = [r["check"] for r in RESULTS if not r["pass"]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed; written to {run_dir}")
    if failed:
        print("FAILED:", *failed, sep="\n  ")
        sys.exit(1)


if __name__ == "__main__":
    main()
