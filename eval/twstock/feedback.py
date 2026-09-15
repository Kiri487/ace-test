"""Reflector feedback for A2 (v9.3 §4.4, §10.1 (一)). Three measured items, no verdict.

    .venv/bin/python -m eval.twstock.feedback --build-null     regenerate and verify the pooled null

  (1) every universe member's score on the decision date with its realized h=10 alpha
  (2) the decision date's csIC: Spearman of score and alpha over members with both (ic.cs_ic,
      the evaluation code itself, so the Reflector sees the number the thesis reports)
  (3) that csIC's quantile in the pooled daily csIC of random scores: 50 seeds x 399 decision
      dates (2025-01-02..2026-08-26), h=10, sd 0.144

No threshold, no class, no right/wrong label. Item (1) goes into the Reflector's ground_truth
slot, items (2) and (3) into its environment_feedback slot (roles.py).

The pooled null is the one known_answer.run_null built: seeds 1..50, uniform(-1, 1) scores over
the point-in-time universe mask of the validation period, daily csIC at h=10, NaN dates dropped.
Only its summary was saved (NULL_SUMMARY), so build_null() regenerates the array on the same code
path, and save_null() refuses to write it unless n, mean, sd, share positive and q05/q25/q50/q75/q95
all equal that summary exactly.

It is rebuilt from the alpha panel that run wrote to disk (NULL_PANEL, data to 2026-09-11), not from
live FinLab data: on 2026-09-14 a rebuild from live data missed the summary (sd 0.144034 vs 0.144048)
because FinLab had revised 294 alpha cells and 12 universe rows on 6 decision dates since 09-11, while
the recorded panel reproduces every summary field exactly
(results/feedback_null/diagnose_20260914_215534.json). The null is thus frozen to one data vintage.

Quantile = mid-rank empirical CDF, (#{null < x} + 0.5 #{null == x}) / n: Spearman on 50 stocks takes
discrete values, so ties with the null count half. A csIC that cannot be computed (fewer than
ic.CS_MIN_PAIRS valid pairs) has no quantile and the text says so.
"""

import argparse
import datetime as _dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import alpha, ic, panel, records

H = 10
NULL_SEEDS = range(1, 51)
NULL_PERIOD = ("2025-01-02", "2026-08-26")
NULL_SUMMARY = records.REPO_ROOT / "results" / "eval_layer" / "known_answer_20260913_172942" / "summary.json"
NULL_PANEL = NULL_SUMMARY.parent / "random_actual" / "decisions.parquet"
NULL_PATH = records.REPO_ROOT / "results" / "feedback_null" / "pooled_random_cs_h10.npy"
SUMMARY_KEYS = ("n", "mean", "std", "share_positive", "q05", "q25", "q50", "q75", "q95")


def _stats(daily):
    out = {"n": int(len(daily)), "mean": float(daily.mean()), "std": float(daily.std(ddof=1)),
           "share_positive": float((daily > 0).mean())}
    for q in (0.05, 0.25, 0.50, 0.75, 0.95):
        out[f"q{int(round(q * 100)):02d}"] = float(np.quantile(daily, q))
    return out


def saved_summary():
    s = json.loads(NULL_SUMMARY.read_text(encoding="utf-8"))
    return s["null"][str(H)]["null_daily_cs_distribution"]


def build_null():
    """The pooled daily csIC of random scores, in known_answer.run_null's order, from NULL_PANEL."""
    pnl = pd.read_parquet(NULL_PANEL)
    dates = pd.DatetimeIndex(sorted(pnl["decision_date"].unique()))
    if (str(dates[0].date()), str(dates[-1].date())) != NULL_PERIOD:
        raise RuntimeError(f"{NULL_PANEL} covers {dates[0].date()}..{dates[-1].date()}, not {NULL_PERIOD}")
    mask = ic.to_matrix(pnl, "universe_rank", dates).notna()
    A = ic.to_matrix(pnl, f"alpha_h{H}", dates)
    A = A.loc[A.notna().any(axis=1)]
    daily = []
    for seed in NULL_SEEDS:
        rng = np.random.default_rng(seed)
        S = mask.astype("float64")
        S[:] = rng.uniform(-1, 1, mask.shape)
        S = S.where(mask)
        cs, _ = ic.cs_ic(S.loc[A.index], A)
        daily.append(cs.dropna().to_numpy())
    summary = json.loads(NULL_SUMMARY.read_text(encoding="utf-8"))
    return np.concatenate(daily), {"n_dates": int(len(dates)), "n_dates_with_alpha": int(len(A)),
                                   "source_panel": str(NULL_PANEL),
                                   "source_data_last_dates": summary["data_last_dates"]}


def compare(stats, saved):
    return {k: {"rebuilt": stats[k], "saved": saved[k], "equal": stats[k] == saved[k]} for k in SUMMARY_KEYS}


def save_null(path=NULL_PATH):
    daily, info = build_null()
    cmp = compare(_stats(daily), saved_summary())
    if not all(v["equal"] for v in cmp.values()):
        raise RuntimeError(f"rebuilt null differs from {NULL_SUMMARY}: {json.dumps(cmp, indent=1)}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, daily)
    side = {"created_at": _dt.datetime.now().isoformat(timespec="seconds"), "git": records.git_state(),
            "source_summary": str(NULL_SUMMARY), "seeds": [NULL_SEEDS.start, NULL_SEEDS.stop - 1],
            "period": NULL_PERIOD, "h": H, **info, "comparison_with_saved_summary": cmp,
            "sha256": hashlib.sha256(daily.tobytes()).hexdigest()}
    records.to_json(side, path.with_suffix(".json"))
    return path


def load_null(path=NULL_PATH):
    """Sorted null array; its stats are checked against the saved summary on every load."""
    daily = np.load(Path(path))
    side = json.loads(Path(path).with_suffix(".json").read_text(encoding="utf-8"))
    if hashlib.sha256(daily.tobytes()).hexdigest() != side["sha256"]:
        raise RuntimeError(f"{path} does not match its sha256")
    cmp = compare(_stats(daily), saved_summary())
    if not all(v["equal"] for v in cmp.values()):
        raise RuntimeError(f"{path} no longer matches {NULL_SUMMARY}")
    return np.sort(daily)


def load_or_build_null(path=NULL_PATH):
    if not Path(path).exists():
        save_null(path)
    return load_null(path)


def cs_ic_one(scores, alpha_by_stock, universe):
    """(csIC, valid pairs) for one decision date, through ic.cs_ic."""
    S = pd.DataFrame([[scores.get(s, np.nan) for s in universe]], columns=list(universe), dtype="float64")
    A = pd.DataFrame([[alpha_by_stock.get(s, np.nan) for s in universe]], columns=list(universe), dtype="float64")
    cs, n = ic.cs_ic(S, A)
    return float(cs.iloc[0]), int(n.iloc[0])


def null_quantile(x, null_sorted):
    if x is None or not np.isfinite(x):
        return float("nan")
    lo = np.searchsorted(null_sorted, x, side="left")
    hi = np.searchsorted(null_sorted, x, side="right")
    return float((lo + 0.5 * (hi - lo)) / len(null_sorted))


def _alpha_text(a):
    return "NA" if a is None or pd.isna(a) else f"{a * 100:+.2f}%"


def render(decision_date, entry_date, mature_date, universe, names, scores, alpha_by_stock, null_sorted):
    """The three items as text: {"ground_truth", "environment_feedback", "csic", "n_pairs", "quantile"}."""
    d, e, m = (pd.Timestamp(x).date() for x in (decision_date, entry_date, mature_date))
    csic, n_pairs = cs_ic_one(scores, alpha_by_stock, universe)
    q = null_quantile(csic, null_sorted)
    lines = [f"決策日 {d} 的評分與實現 α（h=10：{e} 開盤進場，{m} 開盤結算，{m} 收盤後揭露；"
             f"α 為該股報酬減股票池 50 檔等權平均報酬）",
             "代號 簡稱 當時評分 實現α(10日)"]
    for sid in universe:
        sc = scores.get(sid)
        sc_s = "NA" if sc is None else f"{sc:+.2f}"
        lines.append(f"{sid} {names.get(sid, '')} {sc_s} {_alpha_text(alpha_by_stock.get(sid))}")
    sd = float(np.std(null_sorted, ddof=1))
    if np.isfinite(csic):
        env = [f"該決策日的 csIC（評分與實現 α 的 Spearman 相關，有效配對 {n_pairs} 檔）：{csic:+.4f}",
               f"此 csIC 在隨機評分分布中的分位數：{q * 100:.1f}%"
               f"（隨機評分的 csIC 低於此值的比例，同值計一半；分布為 {len(NULL_SEEDS)} 個種子 × 399 個決策日、"
               f"h=10、標準差 {sd:.3f}）"]
    else:
        env = [f"該決策日的 csIC：無法計算（有效配對 {n_pairs} 檔，少於 {ic.CS_MIN_PAIRS}）",
               "此 csIC 在隨機評分分布中的分位數：無"]
    return {"ground_truth": "\n".join(lines), "environment_feedback": "\n".join(env),
            "csic": csic, "n_pairs": n_pairs, "quantile": q}


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--build-null", action="store_true")
    args = ap_.parse_args()
    if args.build_null:
        path = save_null()
        print("written", path, "and", path.with_suffix(".json"))
    null = load_null()
    print("loaded", len(null), "values; sd", float(np.std(null, ddof=1)))


if __name__ == "__main__":
    main()
