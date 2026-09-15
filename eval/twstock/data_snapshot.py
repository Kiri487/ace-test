"""Frozen FinLab data snapshot, read by every experiment and every offline check (user, 2026-09-15).

    .venv/bin/python -m eval.twstock.data_snapshot --refetch   download every dataset in DATASETS once, fresh,
                                                              diff against the old cache, freeze read-only
    .venv/bin/python -m eval.twstock.data_snapshot --verify    md5 of every file == MANIFEST.json, read-only

Why: the old cache was a patchwork fetched on different days (2026-09-03 .. 09-15), and FinLab revises
history (etl:market_value, v9.3 §5.0). A multi-hour three-arm run must not read two data versions.

Reading. market.login() calls activate() unless TWSTOCK_FINLAB_LIVE=1. finlab's FileStorage opens a lock
file next to every data file and may rewrite expiry.pkl, so it cannot read a read-only directory. activate()
checks every file's md5 and read-only mode against MANIFEST.json, builds a per-process view directory whose
data files are symlinks to the snapshot files, points finlab at it and sets data.use_local_data_only: finlab's
own read path (_finalize, stock-id refinement) runs unchanged, nothing is downloaded, a write to a data file
fails. provenance() goes into every manifest (records.write_run, budget_estimate).

The feedback null is not rebuilt from the snapshot (v9.3 §5.0): FEEDBACK_NULL records the vintage it came from.
"""

import argparse
import contextlib
import datetime as _dt
import hashlib
import io
import json
import os
import re
import shutil
import stat
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd
from finlab import data
from finlab.data import FileStorage

SNAPSHOT_ROOT = Path(os.environ.get("TWSTOCK_SNAPSHOT_ROOT", "/home/ubuntu/finlab_snapshots"))
SNAPSHOT_ID = "finlab_20260915_200558"   # frozen 2026-09-15 by --refetch; results/finlab_snapshot/<id>/refetch_report.json

# Every dataset the code in eval/twstock reads (data.get call sites and news_config.json, 2026-09-15).
DATASETS = {
    "tw_news_cnyes": "prompt: headlines (news.raw_news, news_config.json)",
    "etl:market_value": "prompt and evaluation: point-in-time top-50 universe (panel.universe)",
    "etl:adj_open": "prompt and evaluation: trading calendar, alpha (panel.frame)",
    "company_basic_info": "prompt: company short names (format_trial.company_names)",
    "etl:adj_close": "validation: eval_selfcheck, tsic_placebo, known_answer momentum",
    "price_earning_ratio:本益比": "validation: known_answer value factor",
    "price:開盤價": "validation: eval_selfcheck ex-dividend check",
    "price:收盤價": "validation: eval_selfcheck ex-dividend check",
    "monthly_revenue:去年同月增減(%)": "legacy screening pipeline only (market._frames); not in the A0/A1/A2 prompt",
    "fundamental_features:ROE稅後": "legacy screening pipeline only (market._frames); not in the A0/A1/A2 prompt",
    "institutional_investors_trading_summary:外陸資買賣超股數(不含外資自營商)":
        "legacy screening pipeline only (market._frames); not in the A0/A1/A2 prompt",
}
LONG_TABLES = ("tw_news_cnyes", "company_basic_info")       # RangeIndex tables: diffed as row multisets
ADJUSTED = ("etl:adj_open", "etl:adj_close")                 # diffed as daily ratios
QUOTA_HEADROOM = 1.5
FEEDBACK_NULL = {
    "array": "results/feedback_null/pooled_random_cs_h10.npy",
    "built_from": "results/eval_layer/known_answer_20260913_172942/random_actual/decisions.parquet",
    "data_last_dates": {"adj_open": "2026-09-11", "adj_close": "2026-09-11", "market_value": "2026-09-11", "pe": "2026-09-11"},
    "source_cache": "the live finlab_cache_ace of 2026-09-13, before any frozen snapshot existed",
    "sd": 0.1440475566,
    "rebuilt_on_snapshot": False,
    "rule": "v9.3 §5.0: the null is frozen and not recomputed when the data are refreshed",
}
_ACTIVE = None


def file_name(dataset):
    return dataset.replace(":", "#") + ".feather"


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _now():
    return _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=8)))


# ------------------------------------------------------------------ reading

def activate(snapshot_id=None):
    """Point finlab at the frozen snapshot, local reads only. Idempotent within a process."""
    global _ACTIVE
    sid = snapshot_id or os.environ.get("TWSTOCK_FINLAB_SNAPSHOT") or SNAPSHOT_ID
    if sid is None:
        raise RuntimeError("no frozen snapshot: run `python -m eval.twstock.data_snapshot --refetch` and set SNAPSHOT_ID")
    if _ACTIVE is not None:
        if _ACTIVE["snapshot_id"] != sid:
            raise RuntimeError(f"snapshot {_ACTIVE['snapshot_id']} already active in this process; refusing {sid}")
        return _ACTIVE
    root = SNAPSHOT_ROOT / sid
    man_path = root / "MANIFEST.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    if man["snapshot_id"] != sid:
        raise RuntimeError(f"{man_path} names {man['snapshot_id']}, expected {sid}")
    problems = verify_files(root, man)
    if problems:
        raise RuntimeError(f"snapshot {sid} does not match its manifest: {problems[:5]}")
    view = Path(tempfile.mkdtemp(prefix=f"twstock_view_{sid}_"))
    for info in man["files"].values():
        (view / info["file"]).symlink_to(root / info["file"])
    data.set_storage(FileStorage(str(view)))
    data.force_cloud_download = False
    data.use_local_data_only = True
    _ACTIVE = {"snapshot_id": sid, "path": str(root), "frozen_at": man["frozen_at"],
               "manifest_md5": md5(man_path), "datasets": sorted(man["files"]),
               "checked_on_activation": "md5 and read-only mode of every file", "view": str(view)}
    return _ACTIVE


def verify_files(root, man):
    problems = []
    for ds, info in man["files"].items():
        p = Path(root) / info["file"]
        if not p.is_file():
            problems.append((ds, "missing"))
            continue
        if p.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            problems.append((ds, "writable"))
        if md5(p) != info["md5"]:
            problems.append((ds, "md5"))
    if set(man["files"]) != set(DATASETS):
        problems.append(("datasets", sorted(set(DATASETS) ^ set(man["files"]))))
    return problems


def provenance():
    """What goes into a manifest: the active snapshot, or an explicit live-data marker."""
    if _ACTIVE is None:
        return {"snapshot_id": None, "mode": "live FinLab cache (not a frozen snapshot)"
                if os.environ.get("TWSTOCK_FINLAB_LIVE") == "1" else "no FinLab data read in this process"}
    return {k: v for k, v in _ACTIVE.items() if k != "view"}


# ------------------------------------------------------------------ refetch and freeze

def _long_diff(old_path, new_path):
    old, new = pd.read_feather(old_path), pd.read_feather(new_path)
    cols = [c for c in new.columns if c in old.columns]
    ho = Counter(pd.util.hash_pandas_object(old[cols].astype(str), index=False).tolist())
    hn = Counter(pd.util.hash_pandas_object(new[cols].astype(str), index=False).tolist())
    return {"compared_as": "row multiset over shared columns", "rows_old": int(len(old)), "rows_new": int(len(new)),
            "columns_only_old": sorted(set(old.columns) - set(new.columns)),
            "columns_only_new": sorted(set(new.columns) - set(old.columns)),
            "rows_only_in_old": int(sum((ho - hn).values())), "rows_only_in_new": int(sum((hn - ho).values()))}


def _wide_diff(dataset, old_path, new_path):
    from . import finlab_revision_scan as scan
    old, new = scan.load(Path(old_path)), scan.load(Path(new_path))
    res = scan.compare(old, new, "ratio" if dataset in ADJUSTED else "value")
    res.update({"old_last_index": scan.date_str(old.index.max()), "new_last_index": scan.date_str(new.index.max()),
                "new_index_after_old_last": int((new.index > old.index.max()).sum())})
    return res


def _fetch(dataset):
    buf = io.StringIO()
    t0 = _now()
    with contextlib.redirect_stdout(buf):
        df = data.get(dataset)
    out = buf.getvalue()
    usage = re.findall(r"Daily usage: ([\d.]+) / ([\d.]+) MB", out)
    return df, {"fetched_at": t0.isoformat(timespec="seconds"), "stdout": out[-2000:],
                "daily_usage_mb": [float(usage[-1][0]), float(usage[-1][1])] if usage else None}


def refetch_and_freeze():
    from . import finlab_revision_scan as scan
    from . import market, records
    os.environ["TWSTOCK_FINLAB_LIVE"] = "1"
    market.login()
    old_cache = Path(market.CACHE_DIR)
    sid = f"finlab_{_now():%Y%m%d_%H%M%S}"
    staging = SNAPSHOT_ROOT / f"_staging_{sid}"
    staging.mkdir(parents=True)
    report_dir = records.REPO_ROOT / "results" / "finlab_snapshot" / sid
    report_dir.mkdir(parents=True, exist_ok=True)
    data.set_storage(FileStorage(str(staging)))
    data.use_local_data_only = False
    data.force_cloud_download = True

    expected_mb = sum((old_cache / file_name(d)).stat().st_size for d in DATASETS if (old_cache / file_name(d)).exists()) / 1e6
    fetched, order = {}, ["company_basic_info"] + [d for d in DATASETS if d != "company_basic_info"]
    report = {"snapshot_id": sid, "started_at": _now().isoformat(timespec="seconds"), "git": records.git_state(),
              "expected_download_mb_from_old_cache": round(expected_mb, 1), "old_cache": str(old_cache), "fetch": {}}
    for i, ds in enumerate(order):
        df, info = _fetch(ds)
        report["fetch"][ds] = info
        fetched[ds] = df
        if i == 0:
            usage = info["daily_usage_mb"]
            report["quota_after_first"] = usage
            if usage is None or usage[1] - usage[0] < QUOTA_HEADROOM * expected_mb:
                report["stopped"] = (f"quota headroom unknown or below {QUOTA_HEADROOM} x {expected_mb:.0f} MB; "
                                     "reported before downloading the rest")
                records.to_json(report, report_dir / "refetch_report.json")
                raise SystemExit(report["stopped"])
    data.force_cloud_download = False

    diff = {}
    for ds in DATASETS:
        new_path, old_path = staging / file_name(ds), old_cache / file_name(ds)
        if not new_path.exists():
            raise RuntimeError(f"{ds}: finlab did not write {new_path}")
        if not old_path.exists():
            diff[ds] = {"old": "absent from the old cache"}
            continue
        try:
            d = _long_diff(old_path, new_path) if ds in LONG_TABLES else _wide_diff(ds, old_path, new_path)
        except Exception as e:
            d = {"error": f"{type(e).__name__}: {e}"}
        d["old_file_mtime"] = _dt.datetime.fromtimestamp(old_path.stat().st_mtime).isoformat(timespec="seconds")
        diff[ds] = d
    report["diff_against_old_cache"] = diff

    final = SNAPSHOT_ROOT / sid
    final.mkdir()
    files = {}
    for ds in DATASETS:
        src, dst = staging / file_name(ds), final / file_name(ds)
        shutil.copy2(src, dst)
        df = fetched[ds]
        files[ds] = {"file": dst.name, "bytes": dst.stat().st_size, "md5": md5(dst), "used_by": DATASETS[ds],
                     "rows": int(df.shape[0]), "columns": int(df.shape[1]),
                     "first_index": str(df.index.min()) if len(df) else None, "last_index": str(df.index.max()) if len(df) else None,
                     "finlab_hash": data.hash(df) if hasattr(data, "hash") else None,
                     "fetched_at": report["fetch"][ds]["fetched_at"]}
    man = {"snapshot_id": sid, "frozen_at": _now().isoformat(timespec="seconds"),
           "source": "FinLab data.get with force_cloud_download=True, finlab " + getattr(__import__("finlab"), "__version__", "?")
                     + ", every dataset downloaded in one process into a fresh directory",
           "fetch_window": [min(f["fetched_at"] for f in files.values()), max(f["fetched_at"] for f in files.values())],
           "git": report["git"], "files": files, "feedback_null": FEEDBACK_NULL,
           "reading": "eval.twstock.data_snapshot.activate (symlink view, data.use_local_data_only)"}
    (final / "MANIFEST.json").write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
    for p in final.iterdir():
        p.chmod(0o444)
    final.chmod(0o555)
    shutil.rmtree(staging)
    report.update({"frozen": str(final), "manifest": man, "bytes_total": sum(f["bytes"] for f in files.values())})
    records.to_json(report, report_dir / "refetch_report.json")
    print(json.dumps({"snapshot_id": sid, "path": str(final), "bytes_total": report["bytes_total"],
                      "quota_after_first": report["quota_after_first"],
                      "diff": {k: {kk: v.get(kk) for kk in ("rows_only_in_old", "rows_only_in_new", "old_last_index",
                                                            "new_last_index", "new_index_after_old_last", "error", "old")}
                               | ({"revised": v["revised"]["cells"], "material": v["materially_revised"]["cells"],
                                   "value_to_nan": v["value_to_nan"]["cells"], "nan_to_value": v["nan_to_value"]["cells"]}
                                  if "revised" in v else {})
                               for k, v in diff.items()}}, ensure_ascii=False, indent=1, default=str))
    print("report:", report_dir / "refetch_report.json")
    return sid


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--refetch", action="store_true")
    g.add_argument("--verify", action="store_true")
    ap.add_argument("--snapshot", default=None)
    args = ap.parse_args()
    if args.refetch:
        refetch_and_freeze()
        return
    sid = args.snapshot or SNAPSHOT_ID
    man = json.loads((SNAPSHOT_ROOT / sid / "MANIFEST.json").read_text(encoding="utf-8"))
    problems = verify_files(SNAPSHOT_ROOT / sid, man)
    print(f"{sid}: {'OK' if not problems else problems}")
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
