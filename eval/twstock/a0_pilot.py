"""A0 gate pilot runner (arm_protocol.PILOT_A0, judged by eval/twstock/a0_pilot_gate.md v2.1). LIVE LLM CALLS.

    nohup setsid .venv/bin/python -m eval.twstock.a0_pilot --live > results/a0_pilot/<run>.log 2>&1 &

Refuses to start unless: the gate file on disk, its last commit and GATE_COMMIT/GATE_SHA256 all agree and it has no
local edits; the frozen snapshot is the expected one and every file matches its md5; the request model is
cline-pass/deepseek-v4-flash; ACE_MAX_RETRIES=1; no STOP.json from an earlier stop (--after-stop overrides, for the
user only). Every manifest says run_kind="pilot", usable_as_result=false and names its pilot_window.

Runs, in order (each finishes before the next starts):
  abln   news_ablation, headlines removed          post_85 dates with index % 5 == 0 (17)
  ablr   news_ablation, repeated unchanged          same 17 dates
  post85 post_85                                    2026-04-27..2026-08-26 (85)
  long2505 long_2025_05                             2025-05-01..2026-08-26 (324)
One A0 generation per date through arm_protocol.run_decision (2 retries, void rules unchanged).

Resume: every decision date is written to <run>/dates/<date>.json the moment it finishes (scores, every attempt with
its raw response, tokens, latency, provider fields, timestamps). A restart skips the dates on disk. A date interrupted
mid-way is redone from its first attempt.

Stop conditions, fixed before running (user, 2026-09-15); on any of them the runner writes STOP.json and MORNING.md
and exits without retrying:
  S1 10 consecutive failed decision dates (a failed date = void after its retries), counted in run order
  S2 cumulative failed dates over 20% of completed dates, checked from the 10th completed date on
     (before that S1 covers the start; a single early failure would otherwise be 100%)
  S3 requests (provider-log lines, retries included) reach 1.5 x the high estimate 487 = 730
  S4 a request whose body names a model other than EXPECTED_MODEL (request side; the response's model field is
     only recorded, never a stop condition)
Cost is recorded per call (usage.cost) and is not a stop condition.

When a run completes, its records are written (records.write_run). When all four are done, a0_gate.evaluate applies
the gate and MORNING.md is written at the repo root; it is also rewritten after every run and on any stop.
"""

import argparse
import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("ACE_MAX_RETRIES", "1")

import numpy as np
import pandas as pd

import utils

from . import a0_gate, alpha, data_snapshot, market, news, panel, records
from . import arm_protocol as ap
from . import feedback as fb
from . import replay as rp
from . import roles
from .decision_input import build_decision_input, without_news
from .format_trial import company_names, render_context

EXPECTED_MODEL = "cline-pass/deepseek-v4-flash"
EXPECTED_SNAPSHOT = "finlab_20260915_200558"
GATE_PATH = "eval/twstock/a0_pilot_gate.md"
GATE_COMMIT = "4c408f386edabaaf16cab3bcba1a8595b68fad05"      # gate v2.1, committed 2026-09-15
GATE_SHA256 = "d70a06283f5ee5a13ad534dbda5883aa35ca84b08d62e77870d3cd27bee81884"
ESTIMATE_FILE = records.REPO_ROOT / "results" / "budget" / "budget_20260915_205257.json"
ESTIMATE_HIGH_CALLS = 487
CALL_CAP = int(1.5 * ESTIMATE_HIGH_CALLS)       # 730
STOP_CONSECUTIVE = 10
STOP_FAIL_RATE = 0.20
FAIL_RATE_FROM_DATES = 10
MORNING = records.REPO_ROOT / "MORNING.md"
TZ = _dt.timezone(_dt.timedelta(hours=8))


def now():
    return _dt.datetime.now(TZ).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Run:
    name: str            # directory name
    label: str           # call-id label: no "-" or "_"
    pilot_window: str
    strip_news: bool


RUNS = (Run("news_ablation_nonews", "abln", "news_ablation", True),
        Run("news_ablation_repeat", "ablr", "news_ablation", False),
        Run("post_85", "post85", "post_85", False),
        Run("long_2025_05", "long2505", "long_2025_05", False))


def run_dates(run):
    if run.pilot_window == "news_ablation":
        return panel.decision_dates(*rp.PILOT_WINDOWS["post_85"])[::ap.PILOT_NEWS_ABLATION_EVERY]
    return panel.decision_dates(*rp.PILOT_WINDOWS[run.pilot_window])


class NoNewsContexts(roles.Contexts):
    """The Generator context with every headline removed (decision_input.without_news)."""

    def get(self, day):
        day = pd.Timestamp(day)
        if day not in self._cache:
            inp = without_news(build_decision_input(self.index, day, self.cfg))
            context, _, _ = render_context(inp, self.names)
            self._cache[day] = roles.DateInput(context, tuple(u["stock_id"] for u in inp.universe), inp.entry_date)
        return self._cache[day]


class StopRun(Exception):
    pass


# ------------------------------------------------------------------ preflight

def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def check_gate():
    root = records.REPO_ROOT
    disk = sha256_bytes((root / GATE_PATH).read_bytes())
    git = lambda *a: subprocess.run(["git", *a], cwd=root, capture_output=True, check=True).stdout
    last = git("log", "-1", "--format=%H", "--", GATE_PATH).decode().strip()
    committed = sha256_bytes(git("show", f"{GATE_COMMIT}:{GATE_PATH}"))
    dirty = git("status", "--porcelain", "--", GATE_PATH).decode().strip()
    problems = [p for p, bad in (("disk sha256 != GATE_SHA256", disk != GATE_SHA256),
                                 ("committed sha256 != GATE_SHA256", committed != GATE_SHA256),
                                 ("gate file changed after GATE_COMMIT", last != GATE_COMMIT),
                                 ("gate file has local edits", bool(dirty)),
                                 ("a0_gate reads a different file", a0_gate.gate_sha256() != GATE_SHA256)) if bad]
    if problems:
        raise SystemExit(f"gate check failed: {problems}")
    return {"gate_path": GATE_PATH, "gate_commit": GATE_COMMIT, "gate_sha256": GATE_SHA256}


def preflight(out, after_stop):
    if roles.MODEL != EXPECTED_MODEL or ap.SAMPLING["sent"]["model"] != EXPECTED_MODEL:
        raise SystemExit(f"request model is {roles.MODEL!r}, expected {EXPECTED_MODEL!r}: not starting")
    if os.environ.get("ACE_MAX_RETRIES") != "1":
        raise SystemExit("ACE_MAX_RETRIES must be 1")
    if (out / "STOP.json").exists() and not after_stop:
        raise SystemExit(f"{out / 'STOP.json'} exists: a stop condition fired earlier; only the user restarts (--after-stop)")
    gate = check_gate()
    market.login()
    snap = data_snapshot.provenance()
    if snap.get("snapshot_id") != EXPECTED_SNAPSHOT:
        raise SystemExit(f"data snapshot is {snap.get('snapshot_id')!r}, expected {EXPECTED_SNAPSHOT!r}")
    return gate, snap


# ------------------------------------------------------------------ request hook

class Wire:
    """Wraps utils._log_provider_record: request-side model check, first raw response, first request params."""

    def __init__(self, out):
        self.out = out
        self.mismatch = None
        self._orig = utils._log_provider_record

    def install(self):
        utils._log_provider_record = self.hook

    def hook(self, request, status, payload, body):
        try:
            sent = json.loads(request.content or b"{}")
        except Exception:
            sent = {}
        if sent.get("model") != EXPECTED_MODEL:
            self.mismatch = sent.get("model")
        first = self.out / "first_response_raw.json"
        if not first.exists():
            params = {k: v for k, v in sent.items() if k != "messages"}
            first.write_text(json.dumps({
                "captured_at": now(), "http_status": status, "request_url": str(request.url),
                "request_params_without_messages": params,
                "response_headers": None, "response_body_raw": body.decode("utf-8", "replace")},
                ensure_ascii=False, indent=2), encoding="utf-8")
        self._orig(request, status, payload, body)


def provider_lines(out):
    p = out / "provider.jsonl"
    return sum(1 for _ in open(p, encoding="utf-8")) if p.exists() else 0


# ------------------------------------------------------------------ state

def date_file(out, run, day):
    return out / run.name / "dates" / f"{pd.Timestamp(day).date()}.json"


def load_done(out):
    done = []
    for run in RUNS:
        d = out / run.name / "dates"
        if d.exists():
            for f in d.glob("*.json"):
                done.append(json.loads(f.read_text(encoding="utf-8")))
    return sorted(done, key=lambda r: r["finished_at"])


def counters(done):
    consecutive, failed = 0, 0
    for r in done:
        if r["voided"]:
            consecutive += 1
            failed += 1
        else:
            consecutive = 0
    return {"completed_dates": len(done), "failed_dates": failed, "consecutive_failed": consecutive}


def check_stop(out, wire, call_cap, done):
    c = counters(done)
    calls = provider_lines(out)
    if wire.mismatch is not None:
        return "S4", f"a request named model {wire.mismatch!r}, expected {EXPECTED_MODEL!r}", c, calls
    if c["consecutive_failed"] >= STOP_CONSECUTIVE:
        return "S1", f"{c['consecutive_failed']} consecutive failed decision dates", c, calls
    if c["completed_dates"] >= FAIL_RATE_FROM_DATES and c["failed_dates"] / c["completed_dates"] > STOP_FAIL_RATE:
        return "S2", f"failed dates {c['failed_dates']}/{c['completed_dates']} over {STOP_FAIL_RATE:.0%}", c, calls
    if calls >= call_cap:
        return "S3", f"{calls} requests reached the cap {call_cap}", c, calls
    return None, None, c, calls


def write_json_atomic(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


# ------------------------------------------------------------------ finishing a run

def finish_run(out, run, gate, snap):
    dates = run_dates(run)
    recs = [json.loads(date_file(out, run, d).read_text(encoding="utf-8")) for d in dates]
    U = panel.universes(dates)
    pnl = alpha.build_alpha_panel(dates, panel.frame("adj_open"), U)
    frames = [ap.score_rows(r["decision_date"], r["outcome"]) for r in recs]
    frames = [f for f in frames if len(f)]
    scores = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["decision_date", "stock_id", "score"])
    scores["decision_date"] = pd.to_datetime(scores["decision_date"])
    extra = {**rp.manifest_fields("pilot", pilot_window=run.pilot_window), "pilot_run": run.name,
             "headlines_removed": run.strip_news, **gate, "request_model": EXPECTED_MODEL,
             "stop_conditions": {"S1": STOP_CONSECUTIVE, "S2": STOP_FAIL_RATE, "S3": CALL_CAP},
             "provider_log": str(out / "provider.jsonl")}
    records.write_run(out / run.name / "records", pnl, scores, "A0", extra=extra,
                      llm_meta=[r["meta_row"] for r in recs])


def run_complete(out, run):
    return (out / run.name / "records" / "manifest.json").exists()


# ------------------------------------------------------------------ MORNING.md

def _fmt(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def morning(out, status, stop=None, verdict=None, crash=None):
    done = load_done(out)
    est = json.loads(ESTIMATE_FILE.read_text(encoding="utf-8"))["a0_pilot"] if ESTIMATE_FILE.exists() else None
    lines = [f"# A0 pilot 早安報告", "", f"產生時間：{now()}　狀態：**{status}**　輸出目錄：`{out}`", ""]
    lines.append("以下全部由執行程式自動產生。Claude 沒有在跑完後逐項親自檢視；第 7 節是自動檢查，不是人工判讀。")
    lines += ["", "## 1. 跑完了沒", "", "| run | 完成決策日 | 總數 | 剩餘 | records 已寫 |", "|---|---|---|---|---|"]
    for run in RUNS:
        n = len(run_dates(run))
        k = len(list((out / run.name / "dates").glob("*.json"))) if (out / run.name / "dates").exists() else 0
        lines.append(f"| {run.name} | {k} | {n} | {n - k} | {'是' if run_complete(out, run) else '否'} |")

    c = counters(done)
    kinds, retried, attempts_total, calls_error = {}, 0, 0, 0
    cost, latency, forms, echoed, final_prov, resolved = 0.0, 0.0, {}, {}, {}, {}
    long_calls = []
    for r in done:
        o = r["outcome"]
        if o["voided"]:
            kinds[o["void_reason"]] = kinds.get(o["void_reason"], 0) + 1
        retried += o["n_retries"] > 0
        for a in o["attempts"]:
            attempts_total += 1
            calls_error += a.get("failure") == "call_error"
            prov = a.get("provider") or {}
            cost += ((prov.get("usage") or {}).get("cost") or 0.0)
            latency += a.get("latency_s") or 0.0
            if a.get("latency_s") and a["latency_s"] > 300:
                long_calls.append((r["decision_date"], a["latency_s"]))
            forms[a.get("final_answer_form") or "none"] = forms.get(a.get("final_answer_form") or "none", 0) + 1
            for dct, key in ((echoed, "echoed_model"), (final_prov, "finalProvider"), (resolved, "resolvedProvider")):
                v = str(prov.get(key))
                dct[v] = dct.get(v, 0) + 1
    calls = provider_lines(out)
    lines += ["", "## 2. 失敗與停止條件", "",
              f"- 完成決策日 {c['completed_dates']}，其中失敗（重試用盡後作廢）{c['failed_dates']} 日；作廢種類：{kinds or '無'}",
              f"- 需要重試的決策日 {retried}；嘗試次數 {attempts_total}；其中呼叫錯誤（HTTP 或逾時）{calls_error} 次",
              f"- provider log 請求數 {calls}（上限 {CALL_CAP}）",
              f"- 停止條件：{'**觸發 ' + stop['code'] + '：' + stop['reason'] + '**' if stop else '未觸發'}"]
    if crash:
        lines += ["", "**程式異常終止**（不屬於四個停止條件）：", "", "```", crash[-3000:], "```"]

    wall = None
    if done:
        wall = (_dt.datetime.fromisoformat(done[-1]["finished_at"]) - _dt.datetime.fromisoformat(done[0]["started_at"])).total_seconds() / 3600
    lines += ["", "## 3. 成本與時間（對照預估 `budget_20260915_205257.json`）", "",
              f"- 實際：請求 {calls} 次、參考成本 ${cost:.2f}（usage.cost 加總，ClinePass 配額計量，非現金）、"
              f"逐次延遲加總 {latency / 3600:.2f} h、牆鐘時間 {_fmt(wall, 2)} h"]
    if est:
        t = {sc: est[sc]["total"] for sc in ("low", "mid", "high")}
        lines.append(f"- 預估（四個 run 合計）：呼叫 {t['low']['calls']}／{t['mid']['calls']}／{t['high']['calls']}，"
                     f"{t['low']['hours']}／{t['mid']['hours']}／{t['high']['hours']} h，"
                     f"${t['low']['cost_usd_reference']}／${t['mid']['cost_usd_reference']}／${t['high']['cost_usd_reference']}（low／mid／high）")

    lines += ["", "## 4. 第一層 C1–C5", ""]
    if verdict is None:
        lines.append("尚未判定（四個 run 未全部完成，或判定程式未執行）。")
    else:
        c1 = verdict.get("C1_news_dependence_post", {})
        c1v = "不可判" if c1.get("unjudgeable") else ("不通過" if c1.get("fail") else "通過")
        lines += [f"- **C1 新聞依賴性（post_85 的 17 日）**：ρ_無新聞 {_fmt(c1.get('median_rho_nonews'))}、"
                  f"ρ_重複 {_fmt(c1.get('median_rho_repeat'))}、差距 {_fmt(c1.get('gap'))} → **{c1v}**"
                  f"（門檻：ρ_重複 < 0.50 不可判；ρ_無新聞 ≥ 0.90 或差距 < 0.15 不通過）"]
        for w, res in verdict["windows"].items():
            if "missing" in res:
                lines.append(f"- {w}：未完成")
                continue
            c2, c3, c4, c5 = res["C2_zero_news"], res["C3_dispersion"], res["C4_persistence"], res["C5_momentum"]
            yn = lambda f: "不通過" if f else "通過"
            lines += [f"- **{w}**",
                      f"  - C2 零新聞對照：KS {_fmt(c2['ks'])}，安慰劑 95 分位 {_fmt(c2['placebo_q95'])}，零新聞樣本 {c2['n_zero_news_stock_days']} → "
                      f"**{'不可判' if not c2['evaluable'] else yn(c2['fail'])}**",
                      f"  - C3 離散度：每日 sd 中位數 {_fmt(c3['median_sd'])}，sd < 0.05 的日期比例 {_fmt(c3['share_dates_below_floor'])} → **{yn(c3['fail'])}**",
                      f"  - C4 時間自相關：中位數 {_fmt(c4['median_rho_previous_date'])} → **{yn(c4['fail'])}**",
                      f"  - C5 動能共線：vs 5 日 {_fmt(c5['momentum_5d']['median_rho'])}、vs 20 日 {_fmt(c5['momentum_20d']['median_rho'])} → **{yn(c5['fail'])}**"]
    lines += ["", "## 5. 第二層：平均 csIC（h=10）", ""]
    if verdict is not None:
        for w, res in verdict["windows"].items():
            if "missing" in res:
                lines.append(f"- {w}：未完成")
                continue
            l2 = res["layer2_csic_h10"]
            lines.append(f"- **{w}**：平均 csIC {_fmt(l2['mean_csic'], 4)}（可計算 {l2['n_computable']}／{l2['n_dates']} 日，作廢 {l2['voided_dates']} 日）→ "
                         f"**{'無效（不判定）' if l2['invalid'] else ('通過' if l2['pass'] else '不通過')}**")
            bh = res.get("by_horizon_reported_only", {})
            if bh:
                lines.append("  - 僅供參考（不參與判定）：" + "；".join(
                    f"h={h} csIC {_fmt(v['mean_csic'], 4)}／tsIC {_fmt(v['mean_tsic'], 4)}" for h, v in bh.items())
                             + "（tsIC 的 LLM 安慰劑基準線未定義，不可與 0 或 −0.189 比）")
        if "long_2025_05" in verdict["windows"]:
            lines.append("- 長窗解讀規則：可能含污染，通過是模糊的、不通過是強證據；不與 post_85 合併。")
    lines += ["", "## 6. 整體判定", ""]
    if verdict is None:
        lines.append("未判定。")
    else:
        for w, res in verdict["windows"].items():
            lines.append(f"- {w}：**{res.get('verdict')}**")
        lines.append("- 判準檔規定兩窗分開判定、分開報告，未定義合併；退路由使用者決定。")

    lines += ["", "## 7. 自動檢查到、需要你看的東西", ""]
    notes = []
    if len(echoed) != 1:
        notes.append(f"回應中的模型識別欄位出現 {len(echoed)} 種相異值：{echoed}")
    notes.append(f"回應模型欄位（echoed_model）相異值：{echoed}")
    notes.append(f"finalProvider 分布：{final_prov}；resolvedProvider 分布：{resolved}")
    notes.append(f"final_answer 形式分布（所有嘗試）：{forms}")
    if long_calls:
        notes.append(f"延遲超過 300 秒的呼叫 {len(long_calls)} 次：{long_calls[:10]}")
    first = out / "first_response_raw.json"
    if first.exists():
        fr = json.loads(first.read_text(encoding="utf-8"))
        sent = fr.get("request_params_without_messages", {})
        want = ap.SAMPLING["sent"]
        diff = {k: (sent.get(k), v) for k, v in want.items() if sent.get(k) != v}
        notes.append(f"第一次請求的取樣參數：{sent}；與 arm_protocol.SAMPLING 不一致處：{diff or '無'}"
                     f"（完整回應原樣在 `{first}`）")
    if verdict is not None and verdict.get("gate_sha256") != GATE_SHA256:
        notes.append("判定時讀到的判準檔 sha256 與 GATE_SHA256 不同")
    if c["failed_dates"]:
        notes.append(f"有作廢的決策日：{kinds}")
    if crash:
        notes.append("程式異常終止，見第 2 節")
    lines += [f"- {n}" for n in notes]
    lines += ["", "（本檔由 `eval/twstock/a0_pilot.py` 產生；判定細節見輸出目錄的 `gate_verdict.json`。）", ""]
    MORNING.write_text("\n".join(lines), encoding="utf-8")


# ------------------------------------------------------------------ main loop

def run_pilot(out, runs=RUNS, call_cap=CALL_CAP, after_stop=False, stop_after_dates=None):
    """Returns "complete", "stopped" or "interrupted" (stop_after_dates, used by the offline test only)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ["ACE_PROVIDER_LOG"] = str(out / "provider.jsonl")
    gate, snap = preflight(out, after_stop)
    wire = Wire(out)
    wire.install()
    cfg = news.load_config()
    index, _ = news.build(cfg)
    names = company_names()
    null = fb.load_null()
    client = roles.make_client()
    session_dates = 0
    done = load_done(out)
    write_json_atomic(out / "session_start.json", {"at": now(), "pid": os.getpid(), **gate, "snapshot": snap,
                                                   "call_cap": call_cap, "runs": [r.name for r in runs]})
    try:
        for run in runs:
            dates = run_dates(run)
            ctx = (NoNewsContexts if run.strip_news else roles.Contexts)(index, cfg, names)
            live = roles.LiveRoles(client, ctx, names, null, dates, ap.A1_WINDOW_FORMAT, out / run.name / "llm_logs",
                                   roles.AcePlaybook(), run.label)
            for day in dates:
                if date_file(out, run, day).exists():
                    continue
                code, reason, c, calls = check_stop(out, wire, call_cap, done)
                if code:
                    raise StopRun(json.dumps({"code": code, "reason": reason, "counters": c, "requests": calls}))
                if stop_after_dates is not None and session_dates >= stop_after_dates:
                    morning(out, "中斷（測試用）")
                    return "interrupted"
                members = list(ctx.get(day).universe)
                started = now()

                def call(n, day=day):
                    t = now()
                    res = live.generate("A0", day, (), n)
                    res.update({"requested_at": t, "answered_at": now()})
                    return res
                outcome = ap.run_decision(call, members)
                rec = {"run": run.name, "decision_date": str(pd.Timestamp(day).date()), "started_at": started,
                       "finished_at": now(), "universe": members, "voided": outcome["voided"], "outcome": outcome,
                       "meta_row": ap.meta_row(day, outcome)}
                write_json_atomic(date_file(out, run, day), rec)
                done.append(json.loads(json.dumps(rec, ensure_ascii=False, default=str)))
                session_dates += 1
                code, reason, c, calls = check_stop(out, wire, call_cap, done)
                if code:
                    raise StopRun(json.dumps({"code": code, "reason": reason, "counters": c, "requests": calls}))
            if not run_complete(out, run):
                finish_run(out, run, gate, snap)
            morning(out, f"執行中（{run.name} 完成）")
    except StopRun as e:
        stop = json.loads(str(e))
        write_json_atomic(out / "STOP.json", {"at": now(), **stop})
        verdict = _evaluate_available(out)
        morning(out, "已停止（觸發停止條件）", stop=stop, verdict=verdict)
        return "stopped"
    except Exception:
        tb = traceback.format_exc()
        write_json_atomic(out / "STOP.json", {"at": now(), "code": "crash", "reason": tb[-2000:]})
        morning(out, "異常終止", crash=tb, verdict=None)
        raise
    finally:
        utils._log_provider_record = wire._orig
    verdict = _evaluate_available(out)
    morning(out, "全部完成，已判定", verdict=verdict)
    return "complete"


def _evaluate_available(out):
    dirs = {r.name: (out / r.name / "records") if run_complete(out, r) else None for r in RUNS}
    if dirs["post_85"] is None and dirs["long_2025_05"] is None:
        return None
    try:
        v = a0_gate.evaluate(dirs["post_85"], dirs["long_2025_05"], dirs["news_ablation_nonews"],
                             dirs["news_ablation_repeat"], purpose="pilot_gate")
    except Exception:
        v = {"error": traceback.format_exc()[-2000:], "windows": {}, "C1_news_dependence_post": {}}
    records.to_json(v, out / "gate_verdict.json")
    return v if "error" not in v else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--live", action="store_true", help="make real LLM calls (required)")
    p.add_argument("--out", default=str(records.REPO_ROOT / "results" / "a0_pilot" / "pilot_20260915"))
    p.add_argument("--after-stop", action="store_true", help="user only: continue after a stop condition fired")
    args = p.parse_args()
    if not args.live:
        raise SystemExit("refusing to run without --live (the offline test calls run_pilot directly)")
    status = run_pilot(args.out, after_stop=args.after_stop)
    print("pilot", status, flush=True)
    sys.exit(0 if status == "complete" else 2)


if __name__ == "__main__":
    main()
