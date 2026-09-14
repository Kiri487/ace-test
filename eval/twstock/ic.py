"""csIC, tsIC and their uncertainty (v8 §6.1, §6.5). Spearman throughout.

Everything works on date x stock matrices: rows are consecutive decision dates,
columns are stocks, NaN where a stock is outside that date's universe or a value
is missing. Score and α are masked jointly before ranking, so a NaN on one side
never shifts the ranks on the other.

Uncertainty:
  csIC  one value per date, so clustering by date is the plain standard error
        over dates; overlapping windows are handled by Newey-West, lag h-1.
  tsIC  per stock over time, averaged over stocks. Stocks share dates, so the
        standard error comes from a moving-block bootstrap over dates with
        block length h.
"""

import math

import numpy as np
import pandas as pd

CS_MIN_PAIRS = 20    # a date with fewer valid (score, α) pairs gets csIC = NaN
TS_MIN_OBS = 20      # a stock with fewer valid in-universe dates gets tsIC = NaN
INCONCLUSIVE_THRESHOLDS = (0.10, 0.15, 0.20)


def to_matrix(panel, col, dates=None):
    m = panel.pivot(index="decision_date", columns="stock_id", values=col).sort_index()
    if dates is not None:
        m = m.reindex(pd.DatetimeIndex(dates))
    return m.astype("float64")


def rank_corr(S, A, axis, min_n):
    """Spearman along `axis` (1 = across stocks per date, 0 = across dates per
    stock). Returns (correlations, number of valid pairs)."""
    S, A = S.align(A, join="outer")
    M = S.notna() & A.notna()
    s = S.where(M).rank(axis=axis)
    a = A.where(M).rank(axis=axis)
    other = 1 - axis
    s = s.sub(s.mean(axis=axis), axis=other)
    a = a.sub(a.mean(axis=axis), axis=other)
    num = (s * a).sum(axis=axis)
    den = np.sqrt((s ** 2).sum(axis=axis) * (a ** 2).sum(axis=axis))
    n = M.sum(axis=axis)
    out = num / den
    out[(n < min_n) | ~(den > 0)] = np.nan
    return out, n


def cs_ic(S, A, min_pairs=CS_MIN_PAIRS):
    return rank_corr(S, A, axis=1, min_n=min_pairs)


def ts_ic(S, A, min_obs=TS_MIN_OBS):
    return rank_corr(S, A, axis=0, min_n=min_obs)


def newey_west(x, lag):
    """Mean of a series with its Bartlett-kernel HAC standard error.

    NaNs are dropped before lagging; `interior_nan` counts the ones that sat
    between valid values, since those break the lag structure.
    """
    s = pd.Series(x, dtype="float64")
    valid = s.notna().to_numpy()
    v = s.dropna().to_numpy()
    n = len(v)
    interior = 0
    if n:
        first = int(np.argmax(valid))
        last = len(valid) - 1 - int(np.argmax(valid[::-1]))
        interior = (last - first + 1) - n
    nan = float("nan")
    if n < 2:
        return {"mean": nan, "se": nan, "t": nan, "n": n, "lag": lag, "interior_nan": interior}
    e = v - v.mean()
    var = e @ e / n
    for k in range(1, min(lag, n - 1) + 1):
        var += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    se = math.sqrt(var / n) if var > 0 else nan
    t = float(v.mean() / se) if se == se else nan
    return {"mean": float(v.mean()), "se": se, "t": t, "n": n, "lag": lag, "interior_nan": interior}


def nw_lags(h):
    """Newey-West lags reported side by side. h-1 is the v8 §6.5 setting copied
    from FinEvolveBench; 1.5h and 2h are conservative alternatives, because the
    Bartlett kernel at lag h-1 under-covers overlapping h-day windows."""
    return {"h-1": h - 1, "1.5h": int(math.ceil(1.5 * h)), "2h": 2 * h}


def bartlett_recovery(h, lag):
    """Share of the true standard error of a mean that Bartlett NW recovers at
    `lag`, for a flat MA(h-1) - the autocorrelation of overlapping h-day sums of
    i.i.d. returns. h=10: lag 9 -> 0.82, lag 15 -> 0.89, lag 20 -> 0.92."""
    lrv = h + 2 * sum((1 - k / (lag + 1)) * (h - k) for k in range(1, min(lag, h - 1) + 1))
    return math.sqrt(lrv) / h


def ts_ic_bootstrap(S, A, block, n_boot=500, min_obs=TS_MIN_OBS, seed=0):
    """Moving-block bootstrap over dates of the mean tsIC.

    Whole date blocks are resampled, which keeps both the overlap inside a
    block and the cross-stock correlation on each date.
    """
    S, A = S.align(A, join="outer")
    T = len(S)
    L = max(int(block), 1)
    if T < L:
        return np.array([])
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(T / L)
    Sv, Av = S.to_numpy(), A.to_numpy()
    stats = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, T - L + 1, size=n_blocks)
        rows = (starts[:, None] + np.arange(L)[None, :]).ravel()[:T]
        ts, _ = ts_ic(pd.DataFrame(Sv[rows], columns=S.columns),
                      pd.DataFrame(Av[rows], columns=A.columns), min_obs)
        stats[b] = ts.mean()
    return stats


def split_periods(dates):
    """Cold-Start = the first floor(N/3) decision dates, Exploitation = the
    rest, Overall = all (FinEvolveBench §A.4). N counts every decision date of
    the run, matured or not, because the split describes the learning timeline."""
    dates = pd.DatetimeIndex(sorted(dates))
    cut = len(dates) // 3
    return {"cold_start": dates[:cut], "exploitation": dates[cut:], "overall": dates}


def cs_distribution(cs, n_pairs):
    """Distribution of daily csIC, for choosing the feedback form (v8 §10.1)."""
    v = cs.dropna()
    if v.empty:
        return {}
    q = v.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
    pairs = n_pairs.reindex(v.index).astype("float64")
    out = {
        "n_dates": int(len(v)),
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
        "q05": float(q.loc[0.05]), "q25": float(q.loc[0.25]), "q50": float(q.loc[0.50]),
        "q75": float(q.loc[0.75]), "q95": float(q.loc[0.95]),
        "share_positive": float((v > 0).mean()),
        "mean_pairs": float(pairs.mean()),
        "null_std_theory": float((1 / np.sqrt(pairs - 1)).mean()),
    }
    for th in INCONCLUSIVE_THRESHOLDS:
        out[f"inconclusive_abs_lt_{th:.2f}"] = float((v.abs() < th).mean())
    return out


def summarize(S, A, h, dates, n_boot=500, seed=0):
    """Every number v8 §6.1 / §6.5 asks for, for one score, horizon and period.

    Returns (summary dict, daily csIC series, per-stock tsIC series).
    """
    nan = float("nan")
    dates = pd.DatetimeIndex(dates)
    S, A = S.align(A, join="outer")
    S, A = S.reindex(dates), A.reindex(dates)
    matured = A.notna().any(axis=1)
    S, A = S.loc[matured], A.loc[matured]

    cs, n_pairs = cs_ic(S, A)
    nw = newey_west(cs, lag=h - 1)
    iid = newey_west(cs, lag=0)
    ts, n_obs = ts_ic(S, A)
    boot = ts_ic_bootstrap(S, A, block=h, n_boot=n_boot, seed=seed) if n_boot else np.array([])
    ts_mean = float(ts.mean()) if ts.notna().any() else nan
    ts_se = float(np.nanstd(boot, ddof=1)) if boot.size > 1 else nan
    n_valid = int(cs.notna().sum())
    variants = {}
    for name, lag in nw_lags(h).items():
        r = newey_west(cs, lag=lag)
        variants[name] = {"lag": lag, "se": r["se"], "t": r["t"],
                          "bartlett_recovery_flat_ma": bartlett_recovery(h, lag)}

    summary = {
        "h": h,
        "n_decision_dates_in_period": int(len(dates)),
        "n_matured_dates": int(matured.sum()),
        "n_eff": n_valid / h,
        "cs": {
            "mean": nw["mean"],
            "se_newey_west": nw["se"], "t_newey_west": nw["t"], "nw_lag": h - 1,
            "nw_variants": variants,
            "se_date_cluster": iid["se"], "t_date_cluster": iid["t"],
            "n_dates_valid": n_valid, "interior_nan": nw["interior_nan"],
        },
        "ts": {
            "mean": ts_mean,
            "se_block_bootstrap": ts_se,
            "t_block_bootstrap": ts_mean / ts_se if ts_se == ts_se and ts_se > 0 else nan,
            "block_length": h, "n_boot": int(boot.size),
            "n_stocks_valid": int(ts.notna().sum()),
            "n_stocks_below_min_obs": int(((n_obs > 0) & (n_obs < TS_MIN_OBS)).sum()),
            "min_obs": TS_MIN_OBS,
        },
        "cs_distribution": cs_distribution(cs, n_pairs),
    }
    return summary, cs, ts
