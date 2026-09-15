"""A0 pilot gate: layer-1 input-dependence checks, layer-2 csIC, and the calibration behind layer 1. No LLM call.

    .venv/bin/python -m eval.twstock.a0_gate --calibrate     reference scorers through every layer-1 check

The criteria live in eval/twstock/a0_pilot_gate.md; its JSON block (between the gate-json markers) is the
single source of every threshold, read by load_gate(). evaluate() is what the pilot runner calls on the four
pilot runs, opened with records.read_run(purpose="pilot_gate"); it records the gate file's sha256.

Layer 1 measures how the scores depend on the input, not how good they are (v9.3 §5.3): necessary, not
sufficient. The five checks, on a date x stock score matrix S:
  C1 news_dependence  17 post dates scored twice more - headlines removed, and unchanged: per date Spearman of the
                      original with each; medians rho_nonews and rho_repeat
  C2 zero_news        KS statistic between the scores of universe members with no headline in their 5-session
                      window and the others, beside its own placebo band (same per-date counts, members drawn at
                      random, PLACEBO_DRAWS draws, seed 0)
  C3 dispersion       daily cross-sectional sd of the scores; share of dates below SD_FLOOR
  C4 persistence      per date Spearman of the scores with the previous decision date's (common members); median
  C5 momentum         per date Spearman of the scores with momentum_5d and momentum_20d; median of each
Calibration runs C2-C5 on reference scorers that ignore headlines (momentum 5/20/60 days, random) over both
pilot windows, and simulates C1 for a scorer with no input dependence (independent random scores).
"""

import argparse
import datetime as _dt
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from . import alpha, factors, ic, news, panel, records
from .arm_protocol import PILOT_NEWS_ABLATION_EVERY
from .replay import PILOT_WINDOWS

GATE_FILE = Path(__file__).with_name("a0_pilot_gate.md")
PLACEBO_DRAWS = 1000
C1_SIMULATIONS = 5000
N_ABLATION_DATES = 17


# ------------------------------------------------------------------ gate file

def gate_sha256():
    return hashlib.sha256(GATE_FILE.read_bytes()).hexdigest()


def load_gate():
    text = GATE_FILE.read_text(encoding="utf-8")
    m = re.search(r"<!-- gate-json -->\s*```json\s*(.*?)```\s*<!-- /gate-json -->", text, re.S)
    if not m:
        raise ValueError(f"{GATE_FILE} has no gate-json block")
    return json.loads(m.group(1))


# ------------------------------------------------------------------ the checks

def dispersion(S, floor):
    sd = S.std(axis=1, ddof=0)
    sd = sd[S.notna().sum(axis=1) > 1]
    return {"median_sd": float(sd.median()), "share_dates_below_floor": float((sd < floor).mean()), "floor": floor,
            "n_dates": int(len(sd))}


def persistence(S):
    rho, n = ic.rank_corr(S, S.shift(1), axis=1, min_n=ic.CS_MIN_PAIRS)
    rho = rho.dropna()
    return {"median_rho_previous_date": float(rho.median()), "n_dates": int(len(rho))}


def momentum_collinearity(S, M):
    out = {}
    for name, Mk in M.items():
        rho, _ = ic.rank_corr(S, Mk, axis=1, min_n=ic.CS_MIN_PAIRS)
        rho = rho.dropna()
        out[name] = {"median_rho": float(rho.median()), "n_dates": int(len(rho))}
    return out


def zero_news_mask(index, dates, universes):
    rows = {}
    for t in dates:
        by = index.headlines_by_stock(t, universes[t])
        rows[t] = {s: len(f) == 0 for s, f in by.items()}
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def zero_news_ks(S, Z, draws=PLACEBO_DRAWS, seed=0):
    Z = Z.reindex(index=S.index, columns=S.columns)
    valid = S.notna()
    zero = (Z == True) & valid                          # noqa: E712
    other = (Z == False) & valid                        # noqa: E712
    x0, x1 = S.values[zero.values], S.values[other.values]
    n0 = int(len(x0))
    stat = float(ks_2samp(x0, x1).statistic) if n0 and len(x1) else None
    rng = np.random.default_rng(seed)
    counts = zero.sum(axis=1).to_numpy()
    rows = [np.flatnonzero(valid.values[i]) for i in range(len(S))]
    vals = S.values
    placebo = []
    for _ in range(draws):
        pick = np.zeros_like(valid.values)
        for i, (c, members) in enumerate(zip(counts, rows)):
            if c:
                pick[i, rng.choice(members, size=c, replace=False)] = True
        p0, p1 = vals[pick & valid.values], vals[~pick & valid.values]
        placebo.append(ks_2samp(p0, p1).statistic if len(p0) else np.nan)
    placebo = np.asarray(placebo, dtype=float)
    return {"ks": stat, "n_zero_news_stock_days": n0, "n_other_stock_days": int(len(x1)),
            "placebo_q50": float(np.nanquantile(placebo, 0.50)), "placebo_q95": float(np.nanquantile(placebo, 0.95)),
            "draws": draws, "seed": seed}


def news_dependence(S_orig, S_nonews, S_repeat):
    r_nn, _ = ic.rank_corr(S_orig, S_nonews, axis=1, min_n=ic.CS_MIN_PAIRS)
    r_rep, _ = ic.rank_corr(S_orig, S_repeat, axis=1, min_n=ic.CS_MIN_PAIRS)
    return {"median_rho_nonews": float(r_nn.dropna().median()), "median_rho_repeat": float(r_rep.dropna().median()),
            "gap": float(r_rep.dropna().median() - r_nn.dropna().median()), "n_dates": int(r_nn.notna().sum())}


def csic_mean(S, pnl, h=10):
    A = ic.to_matrix(pnl, f"alpha_h{h}", S.index)
    cs, _ = ic.cs_ic(S, A)
    return {"mean_csic": float(cs.mean()), "n_computable": int(cs.notna().sum()), "n_dates": int(len(S))}


# ------------------------------------------------------------------ calibration

def _momentum_mats(dates, U, adj_close):
    cols = sorted(set().union(*(set(U[t]) for t in dates)))
    close = adj_close.reindex(columns=cols)
    return {f"momentum_{k}d": ic.to_matrix(factors.momentum(dates, U, close, lookback=k), "score", dates)
            for k in (5, 20, 60)}


def simulate_c1(n_dates=N_ABLATION_DATES, n_stocks=50, sims=C1_SIMULATIONS, seed=0):
    """A scorer with no input dependence: every call draws independent scores."""
    rng = np.random.default_rng(seed)

    def rho(a, b):
        ra, rb = a.argsort(axis=1).argsort(axis=1), b.argsort(axis=1).argsort(axis=1)
        ra = ra - ra.mean(axis=1, keepdims=True)
        rb = rb - rb.mean(axis=1, keepdims=True)
        return (ra * rb).sum(axis=1) / np.sqrt((ra ** 2).sum(axis=1) * (rb ** 2).sum(axis=1))
    gaps, nn = [], []
    for _ in range(sims):
        o, x, r = (rng.uniform(size=(n_dates, n_stocks)) for _ in range(3))
        m_nn, m_rep = np.median(rho(o, x)), np.median(rho(o, r))
        nn.append(m_nn)
        gaps.append(m_rep - m_nn)
    gaps, nn = np.asarray(gaps), np.asarray(nn)
    return {"scorer": "independent uniform scores per call", "n_dates": n_dates, "sims": sims, "seed": seed,
            "gap_q50": float(np.quantile(gaps, 0.5)), "gap_q99": float(np.quantile(gaps, 0.99)),
            "gap_q999": float(np.quantile(gaps, 0.999)), "median_rho_nonews_q99": float(np.quantile(nn, 0.99)),
            "deterministic_news_blind_scorer": {"median_rho_nonews": 1.0, "median_rho_repeat": 1.0, "gap": 0.0}}


def calibrate():
    panel.frame("adj_open")
    cfg = news.load_config()
    index, _ = news.build(cfg)
    adj_close, adj_open = panel.frame("adj_close"), panel.frame("adj_open")
    out = {"created_at": _dt.datetime.now().isoformat(timespec="seconds"), "git": records.git_state(),
           "data_snapshot": __import__("eval.twstock.data_snapshot", fromlist=["provenance"]).provenance(),
           "purpose": "reference numbers for a0_pilot_gate.md layer-1 thresholds; no LLM output involved",
           "windows": {}}
    for wname in ("post_85", "long_2025_05"):
        dates = panel.decision_dates(*PILOT_WINDOWS[wname])
        U = panel.universes(dates)
        M = _momentum_mats(dates, U, adj_close)
        Z = zero_news_mask(index, dates, U)
        pnl = alpha.build_alpha_panel(dates, adj_open, U, horizons=(10,))
        refs = {**M, "random_seed1": ic.to_matrix(factors.random_uniform(dates, U, 1), "score", dates)}
        res = {"n_dates": int(len(dates)), "first": str(dates[0].date()), "last": str(dates[-1].date()),
               "zero_news_stock_days": int(Z.eq(True).values.sum()),
               "zero_news_per_date_mean": float(Z.eq(True).sum(axis=1).mean()),
               "scorers": {}}
        for name, S in refs.items():
            res["scorers"][name] = {
                "C2_zero_news": zero_news_ks(S, Z),
                "C3_dispersion": dispersion(S, 0.05),
                "C4_persistence": persistence(S),
                "C5_momentum": momentum_collinearity(S, {k: M[k] for k in ("momentum_5d", "momentum_20d")}),
                "L2_csic_h10": csic_mean(S, pnl),
            }
            print(wname, name, json.dumps(res["scorers"][name], ensure_ascii=False, default=str)[:600], flush=True)
        out["windows"][wname] = res
    post = panel.decision_dates(*PILOT_WINDOWS["post_85"])
    out["C1_simulation"] = simulate_c1(n_dates=len(post[::PILOT_NEWS_ABLATION_EVERY]))
    print("C1", json.dumps(out["C1_simulation"]), flush=True)
    dest = records.REPO_ROOT / "results" / "a0_gate_calibration" / f"calibration_{_dt.datetime.now():%Y%m%d_%H%M%S}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    records.to_json(out, dest)
    print("written to", dest)
    return out


# ------------------------------------------------------------------ evaluation (called by the pilot runner)

def _scores(run_dir, purpose):
    dec, meta, man = records.read_run(run_dir, purpose)
    dates = pd.DatetimeIndex(sorted(dec["decision_date"].unique()))
    return ic.to_matrix(dec, "score", dates), dec, meta, man


def evaluate(post_dir, long_dir, nonews_dir, repeat_dir, purpose="pilot_gate"):
    """Apply a0_pilot_gate.md to the four pilot runs. Returns the full verdict; decides nothing else."""
    g = load_gate()
    L1, L2 = g["layer1"], g["layer2"]
    cfg = news.load_config()
    index, _ = news.build(cfg)
    adj_close = panel.frame("adj_close")
    verdict = {"gate_sha256": gate_sha256(), "windows": {}}
    if post_dir is not None and nonews_dir is not None and repeat_dir is not None:
        S_post, _, _, _ = _scores(post_dir, purpose)
        S_nn, _, _, _ = _scores(nonews_dir, purpose)
        S_rep, _, _, _ = _scores(repeat_dir, purpose)
        c1 = news_dependence(S_post.reindex(S_nn.index), S_nn, S_rep.reindex(S_nn.index))
        c1["unjudgeable"] = bool(not c1["median_rho_repeat"] >= L1["C1"]["unjudgeable_if_rho_repeat_below"])
        c1["fail"] = bool(not c1["unjudgeable"]
                          and (c1["median_rho_nonews"] >= L1["C1"]["fail_if_median_rho_nonews_at_least"]
                               or c1["gap"] < L1["C1"]["fail_if_gap_below"]))
    else:
        c1 = {"missing": "news_ablation or post_85 not complete", "unjudgeable": True, "fail": False}
    verdict["C1_news_dependence_post"] = c1
    for wname, d in (("post_85", post_dir), ("long_2025_05", long_dir)):
        if d is None:
            verdict["windows"][wname] = {"missing": "run not complete", "verdict": "未完成"}
            continue
        S, dec, meta, man = _scores(d, purpose)
        dates = S.index
        U = {t: list(dec.loc[dec["decision_date"] == t, "stock_id"]) for t in dates}
        M = _momentum_mats(dates, U, adj_close)
        Z = zero_news_mask(index, dates, U)
        void = int(meta["voided"].fillna(False).astype(bool).sum())
        c2 = zero_news_ks(S, Z)
        c2["evaluable"] = c2["n_zero_news_stock_days"] >= L1["C2"]["min_zero_news_stock_days"]
        c2["fail"] = bool(c2["evaluable"] and c2["ks"] <= c2["placebo_q95"])
        c3 = dispersion(S, L1["C3"]["sd_floor"])
        c3["fail"] = bool(c3["share_dates_below_floor"] > L1["C3"]["fail_if_share_of_dates_below_floor_over"])
        c4 = persistence(S)
        c4["fail"] = bool(c4["median_rho_previous_date"] >= L1["C4"]["fail_if_median_rho_at_least"])
        c5 = momentum_collinearity(S, {k: M[k] for k in ("momentum_5d", "momentum_20d")})
        c5["fail"] = bool(any(abs(v["median_rho"]) >= L1["C5"]["fail_if_abs_median_rho_at_least"]
                              for v in c5.values() if isinstance(v, dict)))
        l2 = csic_mean(S, dec)
        w = L2["windows"][wname]
        l2["voided_dates"] = void
        l2["invalid"] = bool(void > w["invalid_if_voided_over"] or l2["n_computable"] < w["invalid_if_computable_below"])
        l2["pass"] = None if l2["invalid"] else bool(l2["mean_csic"] > L2["pass_if_mean_csic_above"])
        extra_h = {}
        for h in (5, 10, 20, 40):
            if f"alpha_h{h}" in dec:
                A = ic.to_matrix(dec, f"alpha_h{h}", S.index)
                cs, _ = ic.cs_ic(S, A)
                ts, _ = ic.ts_ic(S, A)
                extra_h[h] = {"mean_csic": float(cs.mean()), "sd_daily_csic": float(cs.std()),
                              "mean_tsic": float(ts.mean()), "n_csic_dates": int(cs.notna().sum())}
        l1_fail = bool(c2["fail"] or c3["fail"] or c4["fail"] or c5["fail"] or (wname == "post_85" and c1["fail"]))
        l1_unjudgeable = bool(not c2["evaluable"] or (wname == "post_85" and c1["unjudgeable"]))
        if l1_fail:
            w_verdict = "不通過"
        elif l1_unjudgeable or l2["invalid"]:
            w_verdict = "不可判"
        else:
            w_verdict = "通過" if l2["pass"] else "不通過"
        verdict["windows"][wname] = {"C2_zero_news": c2, "C3_dispersion": c3, "C4_persistence": c4, "C5_momentum": c5,
                                     "layer1_fail": l1_fail, "layer1_unjudgeable": l1_unjudgeable,
                                     "layer2_csic_h10": l2, "by_horizon_reported_only": extra_h,
                                     "tsic_note": "LLM-score placebo baseline undefined; not comparable with 0 or -0.189",
                                     "verdict": w_verdict}
    return verdict


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--calibrate", action="store_true", required=True)
    ap_.parse_args()
    calibrate()


if __name__ == "__main__":
    main()
