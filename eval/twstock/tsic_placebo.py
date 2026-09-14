"""i.i.d. placebo for IC on scores built from a stock's own past returns.

    .venv/bin/python -m eval.twstock.tsic_placebo

Formal evidence for two findings of 2026-09-13.

1. T3b is not a window-overlap test. The same code path run on synthetic opens
   with i.i.d. log returns - per-stock volatility and the real NaN pattern kept,
   serial and cross-sectional correlation removed - gives csIC ~ 0 for
   already-realized alpha. An overlap bug would give positive csIC here too, so
   the positive csIC on real data is serial dependence in alpha. T3a in
   eval_selfcheck is the overlap test.
2. tsIC is biased negative, mechanically, for any score derived from the
   stock's own past returns, and the bias grows with h. This is the baseline to
   report next to any arm's tsIC: LLM arms see prices, so a negative tsIC is
   not by itself evidence that the model's judgement is inverted.

Two score constructions are measured, on real data and on every simulation:
  realized_alpha  alpha(i, t-h-1, h), the alpha that matured at t (the T3b score)
  momentum_20d    rank of the past 20-day return (the known-answer factor)

Descriptive only: no pass criterion. The synthetic draw order is fixed, so the
realized_alpha numbers reproduce the scratch run reported on 2026-09-13.
"""

import argparse
import datetime as _dt
from pathlib import Path

import numpy as np
import pandas as pd

from . import alpha, factors, ic, panel, records
from .panel import HORIZONS

N_SIM = 30
SEED = 12345
LOOKBACK_BARS = 100          # bars kept before the first decision date
SCORES = ("realized_alpha", "momentum_20d")


def _ic_means(opens, closes, dates, U):
    """Mean csIC and tsIC of both score constructions on one price panel."""
    pnl = alpha.build_alpha_panel(dates, opens, U)
    S_mom = ic.to_matrix(factors.momentum(dates, U, closes), "score", dates)
    out = {}
    for h in HORIZONS:
        A = ic.to_matrix(pnl, f"alpha_h{h}", dates)
        res = {}
        for name, S in (("realized_alpha", A.shift(h + 1)), ("momentum_20d", S_mom)):
            c, _ = ic.cs_ic(S, A)
            t_, _ = ic.ts_ic(S, A)
            res[name] = {"cs": float(c.mean()), "ts": float(t_.mean())}
        out[h] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-01-02")
    ap.add_argument("--end", default="2026-08-26")
    ap.add_argument("--n-sim", type=int, default=N_SIM)
    ap.add_argument("--out", default=str(records.REPO_ROOT / "results" / "eval_layer"))
    args = ap.parse_args()

    dates = panel.decision_dates(args.start, args.end)
    U = panel.universes(dates)
    cols = sorted(set().union(*(set(U[t]) for t in dates)))
    idx = panel.trading_days()
    lo = max(idx.searchsorted(dates[0]) - LOOKBACK_BARS, 0)
    real_open = panel.frame("adj_open").reindex(columns=cols).iloc[lo:]
    real_close = panel.frame("adj_close").reindex(columns=cols).iloc[lo:]

    logret = np.log(real_open).diff()
    sigma = logret.loc[dates[0]:dates[-1]].std().fillna(logret.stack().std())
    nan_mask = real_open.isna()

    real = _ic_means(real_open, real_close, dates, U)
    rng = np.random.default_rng(SEED)
    sims = []
    for s in range(args.n_sim):
        shocks = rng.normal(size=real_open.shape) * sigma.to_numpy()[None, :]
        syn = pd.DataFrame(100 * np.exp(np.cumsum(shocks, axis=0)),
                           index=real_open.index, columns=cols).mask(nan_mask)
        sims.append(_ic_means(syn, syn, dates, U))
        print(f"  simulation {s + 1}/{args.n_sim}", flush=True)

    print(f"\ni.i.d. placebo: {args.n_sim} simulations, {len(dates)} decision dates "
          f"({args.start}..{args.end}), same universes, NaN pattern and code path")
    print("   h  score            metric    real   iid mean  iid sd   iid q05  iid q95  real z")
    table = {}
    for h in HORIZONS:
        table[h] = {}
        for score in SCORES:
            table[h][score] = {}
            for metric in ("cs", "ts"):
                a = np.array([sim[h][score][metric] for sim in sims])
                r = real[h][score][metric]
                sd = a.std(ddof=1)
                row = {"real": r, "iid_mean": float(a.mean()), "iid_sd": float(sd),
                       "iid_q05": float(np.quantile(a, 0.05)), "iid_q95": float(np.quantile(a, 0.95)),
                       "real_z": float((r - a.mean()) / sd), "iid_values": a.tolist()}
                table[h][score][metric] = row
                print(f"  {h:2d}  {score:15s}  {metric}IC  {r:+.4f}  {a.mean():+.4f}  {sd:.4f}  "
                      f"{row['iid_q05']:+.4f}  {row['iid_q95']:+.4f}  {row['real_z']:+.2f}")

    run_dir = Path(args.out) / f"tsic_placebo_{_dt.datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    records.to_json({
        "purpose": __doc__, "git": records.git_state(), "data_last_dates": panel.data_last_dates(),
        "period": {"start": args.start, "end": args.end, "n_decision_dates": len(dates)},
        "n_sim": args.n_sim, "seed": SEED, "results": table,
    }, run_dir / "placebo.json")
    print(f"\nwritten to {run_dir}")


if __name__ == "__main__":
    main()
