"""Known-answer test of the evaluation layer. Not a factor study.

    .venv/bin/python -m eval.twstock.known_answer --stages null,factors

Scores whose behaviour is known go through exactly the path an LLM arm will
use - alpha panel, on-disk records, IC - so a wrong alignment shows up here,
before any LLM output exists. The long period exists only to make the signal
strong enough to judge the code. None of these factor numbers are findings:
the period has no leakage control, so they do not belong in the thesis.

  null     random scores: one recorded run, read back from disk, against a
           500-seed null distribution
  factors  momentum (20-day return rank) and value (earnings-yield rank)

Both stages also report the distribution of daily csIC (v8 §10.1), and the run
ends by printing the on-disk record schema.
"""

import argparse
import datetime as _dt
from pathlib import Path

import numpy as np

from . import alpha, factors, ic, panel, records
from .panel import HORIZONS

PURPOSE = ("known-answer validation of the evaluation code; not a factor-effectiveness "
           "result and not for the thesis (no leakage control over this period)")
ACTUAL_SEED = 20260913
NULL_SEEDS = range(1, 501)
N_BOOT = 500
POOLED_DAILY_SEEDS = 50

FACTOR_H = 10
FACTOR_T_LAG = "1.5h"
FACTOR_T_BOUND = 2.0

RESULTS = []


def evaluate_recorded(run_dir, dates, periods, n_boot=N_BOOT):
    """IC for a run as read back from disk, so the record round trip is tested too."""
    dec, _, _ = records.read_run(run_dir)
    S = ic.to_matrix(dec, "score", dates)
    out = {}
    for h in HORIZONS:
        A = ic.to_matrix(dec, f"alpha_h{h}", dates)
        out[h] = {name: ic.summarize(S, A, h, d, n_boot=n_boot)[0] for name, d in periods.items()}
    return out


def print_cs_distribution(title, rows):
    """rows: (label, h, distribution dict) as built by ic.cs_distribution."""
    print(f"\n  daily csIC distribution - {title}")
    print("   label                         h      n    mean    std     q05     q25     q50     q75     q95"
          "   >0   null sd  |x|<.10 |x|<.15 |x|<.20")
    for label, h, d in rows:
        if not d:
            continue
        n = d.get("n_dates", d.get("n"))
        print(f"   {label:28s} {h:2d} {n:6d} {d['mean']:+.3f}  {d['std']:.3f}  {d['q05']:+.3f}  "
              f"{d['q25']:+.3f}  {d['q50']:+.3f}  {d['q75']:+.3f}  {d['q95']:+.3f}  "
              f"{d['share_positive']:.2f}  {d.get('null_std_theory', float('nan')):.3f}    "
              f"{d['inconclusive_abs_lt_0.10']:.2f}    {d['inconclusive_abs_lt_0.15']:.2f}    "
              f"{d['inconclusive_abs_lt_0.20']:.2f}")


def run_null(root, pnl, dates, U, periods, window):
    print("\n== random factor: recorded run vs 500-seed null ==")
    print("  criterion, fixed before the numbers: for each h, the recorded random factor's")
    print("  overall mean csIC and mean tsIC each lie within [q05, q95] of the null.")
    print("  8 cells; under correct code each cell misses with probability 0.10, and the")
    print("  cells share one set of scores, so they are correlated. Any miss is reported")
    print("  and stops the factor stage.")
    run_dir = records.write_run(root / "random_actual", pnl,
                                factors.random_uniform(dates, U, ACTUAL_SEED),
                                arm="known_answer:random",
                                extra={"seed": ACTUAL_SEED, "purpose": PURPOSE})
    actual = evaluate_recorded(run_dir, dates, periods)
    win = evaluate_recorded(run_dir, window, {"test_window": window})

    mask = ic.to_matrix(pnl, "universe_rank", dates).notna()
    A_by_h = {}
    for h in HORIZONS:
        A = ic.to_matrix(pnl, f"alpha_h{h}", dates)
        A_by_h[h] = A.loc[A.notna().any(axis=1)]
    null = {h: {"cs": [], "ts": [], "daily": []} for h in HORIZONS}
    for seed in NULL_SEEDS:
        rng = np.random.default_rng(seed)
        S = mask.astype("float64")
        S[:] = rng.uniform(-1, 1, mask.shape)
        S = S.where(mask)
        for h in HORIZONS:
            A = A_by_h[h]
            Sh = S.loc[A.index]
            cs, _ = ic.cs_ic(Sh, A)
            ts, _ = ic.ts_ic(Sh, A)
            null[h]["cs"].append(float(cs.mean()))
            null[h]["ts"].append(float(ts.mean()))
            if seed <= POOLED_DAILY_SEEDS:
                null[h]["daily"].append(cs.dropna().to_numpy())

    out, all_pass, dist_rows = {}, True, []
    print("\n   h  metric  recorded    null q05    null q95   null mean   null sd  theory sd  in band")
    for h in HORIZONS:
        o = actual[h]["overall"]
        res = {}
        for metric in ("cs", "ts"):
            arr = np.array(null[h][metric])
            q05, q95 = np.quantile(arr, [0.05, 0.95])
            val = o[metric]["mean"]
            ok = bool(q05 <= val <= q95)
            all_pass &= ok
            theory = float("nan")
            if metric == "cs":
                theory = o["cs_distribution"]["null_std_theory"] / np.sqrt(o["cs"]["n_dates_valid"])
            res[metric] = {"recorded": val, "null_q05": float(q05), "null_q95": float(q95),
                           "null_mean": float(arr.mean()), "null_sd": float(arr.std(ddof=1)),
                           "null_mean_z": float(arr.mean() / (arr.std(ddof=1) / np.sqrt(len(arr)))),
                           "theory_sd_of_mean": theory, "in_band": ok}
            print(f"  {h:2d}  {metric}IC   {val:+.5f}   {q05:+.5f}   {q95:+.5f}   "
                  f"{arr.mean():+.5f}   {arr.std(ddof=1):.5f}   {theory:.5f}   {'yes' if ok else 'NO'}")
            RESULTS.append({"check": f"h={h} random {metric}IC within null [q05, q95]", "pass": ok})
        daily = np.concatenate(null[h]["daily"])
        pooled = {"n": int(len(daily)), "mean": float(daily.mean()), "std": float(daily.std(ddof=1)),
                  "share_positive": float((daily > 0).mean())}
        for q in (0.05, 0.25, 0.50, 0.75, 0.95):
            pooled[f"q{int(round(q * 100)):02d}"] = float(np.quantile(daily, q))
        for th in ic.INCONCLUSIVE_THRESHOLDS:
            pooled[f"inconclusive_abs_lt_{th:.2f}"] = float((np.abs(daily) < th).mean())
        res["null_daily_cs_distribution"] = {"seeds": POOLED_DAILY_SEEDS, **pooled}
        res["recorded_summary"] = actual[h]
        res["recorded_test_window"] = win[h]["test_window"]
        out[h] = res
        dist_rows += [("random recorded, 2025-26", h, o["cs_distribution"]),
                      ("random recorded, test window", h, win[h]["test_window"]["cs_distribution"]),
                      (f"random null pooled {POOLED_DAILY_SEEDS} seeds", h, pooled)]
    print_cs_distribution("random", dist_rows)
    out["all_in_band"] = all_pass
    return out


def run_factors(root, pnl, dates, U, periods, window):
    print("\n== momentum and value ==")
    print("  criteria, fixed before the numbers (they test whether this period can confirm")
    print("  each factor's known behaviour through this code; a miss is reported as a miss):")
    print(f"   nonzero      h={FACTOR_H} overall mean csIC with |t| >= {FACTOR_T_BOUND},"
          f" t from Newey-West lag {FACTOR_T_LAG}")
    print(f"   sign-stable  h={FACTOR_H} mean csIC has one sign in cold-start, exploitation, 2025 and 2026")
    cols = sorted(set().union(*(set(U[t]) for t in dates)))
    specs = {
        "momentum_20d": factors.momentum(dates, U, panel.frame("adj_close").reindex(columns=cols)),
        "value_earnings_yield": factors.value(dates, U, panel.frame("pe").reindex(columns=cols)),
    }
    out, dist_rows = {}, []
    for name, scores in specs.items():
        run_dir = records.write_run(root / name, pnl, scores, arm=f"known_answer:{name}",
                                    extra={"purpose": PURPOSE})
        full = evaluate_recorded(run_dir, dates, periods)
        win = evaluate_recorded(run_dir, window, {"test_window": window})
        dec, _, _ = records.read_run(run_dir)
        S = ic.to_matrix(dec, "score", dates)
        by_year = {}
        for h in HORIZONS:
            A = ic.to_matrix(dec, f"alpha_h{h}", dates)
            _, cs, _ = ic.summarize(S, A, h, dates, n_boot=0)
            by_year[h] = {str(y): float(v) for y, v in cs.groupby(cs.index.year).mean().items()}

        print(f"\n  {name}: score NaN share {scores['score'].isna().mean():.4f}")
        print("   h  period          csIC  t(h-1) t(1.5h)  t(2h)    tsIC  boot t  N_eff   csIC by year")
        for h in HORIZONS:
            rows = [(p, full[h][p]) for p in ("cold_start", "exploitation", "overall")]
            rows.append(("test window", win[h]["test_window"]))
            for p, s in rows:
                v = s["cs"]["nw_variants"]
                yr = {k: round(x, 4) for k, x in by_year[h].items()} if p == "overall" else ""
                print(f"  {h:2d}  {p:13s} {s['cs']['mean']:+.4f}  {v['h-1']['t']:+5.2f}  {v['1.5h']['t']:+5.2f}  "
                      f"{v['2h']['t']:+5.2f}  {s['ts']['mean']:+.4f}  {s['ts']['t_block_bootstrap']:+5.2f}  "
                      f"{s['n_eff']:5.1f}   {yr}")
            dist_rows += [(f"{name}, 2025-26", h, full[h]["overall"]["cs_distribution"]),
                          (f"{name}, test window", h, win[h]["test_window"]["cs_distribution"])]

        h10 = full[FACTOR_H]
        t = h10["overall"]["cs"]["nw_variants"][FACTOR_T_LAG]["t"]
        signs = {p: float(np.sign(h10[p]["cs"]["mean"])) for p in ("cold_start", "exploitation")}
        signs.update({y: float(np.sign(v)) for y, v in by_year[FACTOR_H].items()})
        nonzero = bool(abs(t) >= FACTOR_T_BOUND)
        stable = bool(len(set(signs.values())) == 1 and 0.0 not in signs.values())
        RESULTS.append({"check": f"{name} h={FACTOR_H} csIC nonzero", "pass": nonzero,
                        "detail": f"t_NW({FACTOR_T_LAG}) = {t:+.2f}"})
        RESULTS.append({"check": f"{name} h={FACTOR_H} csIC sign stable", "pass": stable,
                        "detail": str(signs)})
        print(f"  [{'PASS' if nonzero else 'MISS'}] {name} nonzero: t_NW({FACTOR_T_LAG}) = {t:+.2f}")
        print(f"  [{'PASS' if stable else 'MISS'}] {name} sign stable: {signs}")
        out[name] = {"score_nan_share": float(scores["score"].isna().mean()),
                     "periods": full, "test_window": win, "cs_mean_by_year": by_year,
                     "criteria": {"nonzero": nonzero, "t": t, "sign_stable": stable, "signs": signs}}
    print_cs_distribution("momentum and value", dist_rows)
    return out


def print_schema(run_dir):
    dec, meta, manifest = records.read_run(run_dir)
    print(f"\n== record schema, as read back from {run_dir.name} ==")
    print(f"  decisions.parquet: {len(dec)} rows")
    for c, t in dec.dtypes.items():
        print(f"    {c:22s} {t}")
    print(f"  decision_meta.parquet: {len(meta)} rows")
    for c, t in meta.dtypes.items():
        print(f"    {c:22s} {t}")
    sample = {k: (str(v)[:70] + "..." if len(str(v)) > 70 else v) for k, v in meta.iloc[0].to_dict().items()}
    print("  first meta row:", sample)
    print("  manifest keys:", list(manifest))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-01-02")
    ap.add_argument("--end", default="2026-08-26")
    ap.add_argument("--window-start", default="2026-04-27")
    ap.add_argument("--window-end", default="2026-08-26")
    ap.add_argument("--stages", default="null,factors")
    ap.add_argument("--out", default=str(records.REPO_ROOT / "results" / "eval_layer"))
    args = ap.parse_args()
    stages = set(args.stages.split(","))

    dates = panel.decision_dates(args.start, args.end)
    U = panel.universes(dates)
    pnl = alpha.build_alpha_panel(dates, panel.frame("adj_open"), U)
    periods = ic.split_periods(dates)
    window = panel.decision_dates(args.window_start, args.window_end)
    root = Path(args.out) / f"known_answer_{_dt.datetime.now():%Y%m%d_%H%M%S}"
    root.mkdir(parents=True, exist_ok=True)

    summary = {
        "purpose": PURPOSE, "git": records.git_state(),
        "data_last_dates": panel.data_last_dates(),
        "validation_period": {"start": args.start, "end": args.end, "n_decision_dates": len(dates),
                              "cold_start_dates": len(periods["cold_start"]),
                              "exploitation_dates": len(periods["exploitation"])},
        "test_window": {"start": args.window_start, "end": args.window_end, "n_decision_dates": len(window)},
        "missing": alpha.missing_report(pnl),
        "cs_min_pairs": ic.CS_MIN_PAIRS, "ts_min_obs": ic.TS_MIN_OBS, "n_boot": N_BOOT,
    }
    print(f"validation period {args.start}..{args.end}: {len(dates)} decision dates "
          f"(cold-start {len(periods['cold_start'])}, exploitation {len(periods['exploitation'])}); "
          f"test window {len(window)} dates")
    print("alpha NaN share among matured rows:",
          {h: round(v["nan_share"], 5) for h, v in summary["missing"].items()})

    if "null" in stages:
        summary["null"] = run_null(root, pnl, dates, U, periods, window)
    if "factors" in stages:
        if "null" in stages and not summary["null"]["all_in_band"]:
            print("\nSTOP: the random factor left its null band; factor stage not run.")
        else:
            summary["factors"] = run_factors(root, pnl, dates, U, periods, window)
    summary["checks"] = RESULTS
    records.to_json(summary, root / "summary.json")

    first = next((root / d for d in ("random_actual", "momentum_20d") if (root / d).exists()), None)
    if first is not None:
        print_schema(first)
    print(f"\nwritten to {root}")


if __name__ == "__main__":
    main()
