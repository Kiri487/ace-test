"""Offline validation of driver part 2, and prompt tokens per arm. No LLM call, no network.

    .venv/bin/python -m eval.twstock.replay_offline [--conditions post,pre] [--formats with_reasoning]

Production code runs unchanged: replay.run_replay, roles.LiveRoles / AcePlaybook, the ACE Generator,
Reflector, Curator and BulletpointAnalyzer with their templates, llm.timed_llm_call, the OpenAI SDK,
utils.initialize_clients and utils._ClineUnwrapTransport with its provider log. Only the wire is
replaced: the transport's parent handle_request is answered by FakeLLM instead of the network, in the
ClinePass response envelope captured on 2026-09-14. Every prompt is therefore assembled, sent, logged,
unwrapped and parsed on the production path, and the checks read the prompts as they were sent.

FakeLLM answers from the request body. It also reads the call label timed_llm_call sets (arm, date,
role, attempt), but only to inject the planned faults and to stamp its own output with a trace token
- 〔TRACE arm date〕 in a generation, 〔SRC arm date〕 in a reflection, 〔FROM arm date〕 in a bullet - taken
from the prompt it was sent, so the checks can follow every text into the prompts it reaches later.
Sizes follow the budget's mid scenario (results/budget/budget_20260914_120638.json) so measured minus
estimated prompt tokens isolates prompt assembly, not model behaviour:
  Generator  scores for the universe listed in its own context, seeded by (label, arm, date); the
             reasoning text of one of the five real non-reasoning trial answers
             (results/format_trial/trial_20260914_110451); bullet_ids = the first quarter of the
             bullets in its playbook (mid bullets_used_share 0.25).
  Reflector  ~1,268 tokens of reasoning (mid), 〔SRC arm date〕 from the trace in its prompt, every bullet
             listed as used tagged helpful.
  Curator    one ADD whose playbook line is ~180 tokens (mid growth per call): 〔FROM arm date〕 from the
             reflection in its prompt, then that date's own headlines from its question context.

Pass criteria, fixed before the first run. Every FAIL is kept, printed, written; exit status 1.
Planned faults use the decision index j inside the condition.

P0 No network: every request goes to http://offline.invalid/; provider-log lines == requests answered.
P1 Prompt memory isolation, on every Generator request as sent, per condition and A1 format:
 a A0: playbook slot == the ACE empty playbook, reflection == "(empty)", question == QUESTION, context ==
   that date's render_context character for character; no TRACE/SRC/FROM token.
 b A1: playbook, reflection, question as A0; context == render_context, plus "\\n\\n" + window when any
   decision has matured. The window lists exactly the non-void A1 decisions among maturity slots
   j-15..j-11 (arm_protocol.window_indices), oldest first, each with maturity date = decision + 11
   sessions, every member in universe order, score == A1's own recorded score (+.2f), alpha == the panel's
   alpha_h10 (+.2f%, NA when missing). with_reasoning: the TRACE tokens present are exactly
   〔TRACE A1 d〕 for the listed d; compact: none. No SRC/FROM token.
 c A2: reflection "(empty)", question QUESTION, context == render_context exactly; playbook slot == the
   playbook text after the last commit on or before t (empty playbook before any) and its sha256 ==
   the attempt's playbook_sha256; no TRACE/SRC token; every FROM token names A2.
 d Only A2 reaches the Reflector and Curator. A Reflector prompt's trace carries 〔TRACE A2 d〕 for the
   decision d of its call; a Curator prompt's reflection carries 〔SRC A2 d〕 for the same d and its
   question context == render_context of d.
P2 A2's playbook at t holds feedback only from t-11 or earlier:
 a every bullet line in every A2 Generator prompt at t carries exactly one 〔FROM A2 d〕 with
   position(d) <= position(t) - 11 on the trading calendar, and its bullet was added on a day <= t;
 b bullet lines == bullets added on days <= t minus bullets removed by dedup on days <= t;
 c negative control: replay._assert_no_future_feedback raises for an entry from t-10 sessions and for
   one that matured after t.
P3 Feedback text, on every Reflector prompt:
 a ground_truth slot: header dates == the decision date, the panel entry_date and mature_date_h10, and
   mature_date_h10 == the harvest day; one line per member in universe order; score == A2's recorded
   score (+.2f); alpha == panel alpha_h10 (+.2f%, NA when missing).
 b csIC == scipy.stats.spearmanr over members with both values, within 5e-5 (print rounding); pair
   count equal.
 c quantile == scipy.stats.percentileofscore(null, csIC, kind="mean"), both rounded to 12 decimals,
   within 0.05 percentage points (print rounding); sd printed as 0.144; the null array loaded by
   feedback.load_null matches the known-answer summary in n, mean, sd, share positive and q05..q95
   exactly.
 d hand cases through feedback.render on 50 members: scores ordered like alpha -> "+1.0000", "100.0%";
   reversed -> "-1.0000", "0.0%"; 19 alphas present -> "無法計算", quantile "無". null_quantile on
   [-0.1, 0, 0, 0.2]: 0 -> 0.5, 0.2 -> 0.875, 0.3 -> 1.0, -0.2 -> 0.0.
 e neither slot holds a verdict word (正確 錯誤 答對 答錯 correct incorrect wrong); every date in them
   is on or before the harvest day.
P4 Void and failure paths:
 a A2 generation incomplete x3 at j=7: void (incomplete, 3 attempts); skipped_void at maturity; no
   Reflector or Curator request for it; A2 playbook slot identical at j=17 and j=18.
 b A2 generation incomplete once at j=10: 2 attempts, not void.
 c A1 generation duplicate key x3 at j=4: void; A1 windows at j=15..19 omit it with
   window_skipped_void == 1, nothing older pulled in.
 d A1 generation out-of-universe key once at j=9: 2 attempts, not void.
 e A0 HTTP 500 x3 at j=30: void (call_error); 3 provider-log lines, http_status 500, provider_metadata null.
 f Reflector not JSON x3 for j=15: reflector failed (3 attempts), curator not_run, no Curator request;
   A2 playbook slot identical at j=25 and j=26.
 g Reflector not JSON once for j=20: ok after 2 attempts; Curator ran; a bullet 〔FROM A2 dates[20]〕 follows.
 h Curator missing "reasoning" x3 for j=25: curator failed (3 attempts, curator_schema); no bullet from
   dates[25]; slot at j=36 vs j=35: same bullet ids and contents, helpful count +1 exactly on the
   bullets its Reflector tagged.
 i Curator not JSON once for j=28: ok after 2 attempts.
P5 provider_metadata reaches disk:
 a one provider-log line per request, in order, with its call_id and role; for answered requests
   generation_id, finalProvider, resolvedProvider, modelAttemptCount and usage == what the wire returned;
 b every attempt that got an answer, in attempts_json and aux_attempts_json read back by
   records.read_run, carries the same generation_id, finalProvider, resolvedProvider,
   modelAttemptCount and usage.cost as its provider-log line;
 c every request body: model cline-pass/deepseek-v4-flash, reasoning {"enabled": false},
   response_format json_object, temperature 0, max_tokens 65536;
 d every answered call has one ACE llm_logs file whose provider.generation_id matches.
 e (schema 4, added 2026-09-15 with these criteria fixed before its first run) llm_calls.parquet per arm:
   rows == requests the wire received for that arm == manifest llm_calls; an answered row's cost_usd ==
   its provider-log usage.cost exactly, an unanswered row has none; per date, generator rows sum to meta
   cost_usd and reflector + curator rows to meta aux_cost_usd (both null when no row has a cost), within
   1e-12; all rows of the label sum to the provider log's usage.cost within 1e-9.
P6 Structure:
 a Generator requests per (arm, date) == recorded attempts, only on decision dates;
 b no regeneration: no Generator request for an A2 decision after its first Reflector request;
 c the Reflector reached exactly the non-void A2 decisions maturing on or before the last decision date;
 d records.write_run / read_run succeed for the three arms, one meta row per date; playbook_bullets rows
   == bullets added;
 e the analyzer ran once per successful Curator commit, always merge=False, with no request during it.
P7 Guards: LiveRoles refuses a1_format None, "other" and "compact" (decided 2026-09-15: with_reasoning);
   ReasoningOffClient refuses max_retries != 0.
P8 a The A1 candidates are the ones the budget measured: the five trial decisions rendered by
     roles.a1_decision_block give per-decision tokens and 5-decision window totals equal to a1_window in
     results/budget/budget_20260914_120638.json.
   b A0 Generator prompt tokens over each condition's dates: n, mean, min, max == that file's
     a0_prompt_tokens.

Not pass/fail, written to tokens.json: prompt tokens (local DeepSeek-V4-Flash tokenizer + 26, the
JSON-mode overhead the trial measured 5/5) per condition, arm and role over the first, middle and last
third of the decision dates, next to the budget formula for the same dates.
"""

import os

os.environ["ACE_MAX_RETRIES"] = "1"
os.environ["CLINE_BASE_URL"] = "http://offline.invalid/api/v1"
os.environ["CLINE_API_KEY"] = "offline-dummy"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import copy
import datetime as _dt
import glob
import hashlib
import json
import math
import re
import string
import sys
import traceback
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore, spearmanr

import utils
from ace.prompts.curator import CURATOR_PROMPT
from ace.prompts.generator import GENERATOR_PROMPT
from ace.prompts.reflector import REFLECTOR_PROMPT
from playbook_utils import parse_playbook_line

from . import alpha, market, news, panel, records
from . import arm_protocol as ap
from . import feedback as fb
from . import replay as rp
from . import roles
from .format_trial import QUESTION, company_names, load_tokenizer

RESULTS = records.REPO_ROOT / "results"
BUDGET = RESULTS / "budget" / "budget_20260914_120638.json"
TRIAL = RESULTS / "format_trial" / "trial_20260914_110451"
JSON_MODE_OVERHEAD = 26
HOSTS = ("fireworks", "deepinfra")
TOKEN_RE = re.compile(r"〔(TRACE|SRC|FROM) (A[012]) (\d{4}-\d{2}-\d{2})〕")
VERDICT_WORDS = ("正確", "錯誤", "答對", "答錯", "correct", "incorrect", "wrong")
FAULTS = {  # (arm, j, role) -> {attempt: kind}; "all" covers every attempt
    ("A2", 7, "gen"): {"all": "incomplete"},
    ("A2", 10, "gen"): {1: "incomplete"},
    ("A1", 4, "gen"): {"all": "duplicate_key"},
    ("A1", 9, "gen"): {1: "out_of_universe"},
    ("A0", 30, "gen"): {"all": "http500"},
    ("A2", 15, "reflect"): {"all": "not_json"},
    ("A2", 20, "reflect"): {1: "not_json"},
    ("A2", 25, "curate"): {"all": "no_reasoning"},
    ("A2", 28, "curate"): {1: "not_json"},
}
ROLE_OF = {"generator": "gen", "reflector": "reflect", "curator": "curate"}
CHECKS = []


def check(label, name, fn):
    try:
        ok, detail = fn()
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"
    CHECKS.append({"label": label, "check": name, "pass": bool(ok), "detail": None if ok else str(detail)[:3000]})
    print(f"  {'PASS' if ok else 'FAIL'}  [{label}] {name}" + ("" if ok else f"  {str(detail)[:600]}"), flush=True)


# ---------------------------------------------------------------- templates as sent

def template_regex(template, names):
    parts, i = [], 0
    for literal, field, _, _ in string.Formatter().parse(template):
        parts.append(re.escape(literal))
        if field is not None:
            parts.append(f"(?P<{names[i]}>.*?)")
            i += 1
    return re.compile("^" + "".join(parts) + "$", re.S)


GEN_RE = template_regex(GENERATOR_PROMPT, ["playbook", "reflection", "question", "context"])
REF_RE = template_regex(REFLECTOR_PROMPT, ["question", "trace", "predicted", "ground_truth", "env", "bullets"])
CUR_RE = template_regex(CURATOR_PROMPT, ["token_budget", "current_step", "total_samples", "playbook_stats",
                                         "recent_reflection", "current_playbook", "question_context"])


def split(regex, prompt):
    m = regex.match(prompt)
    if not m:
        raise ValueError("prompt does not match its template")
    return m.groupdict()


def bullet_lines(text):
    return [ln for ln in text.split("\n") if parse_playbook_line(ln)]


def alpha_text(a):
    return "NA" if a is None or pd.isna(a) else f"{a * 100:+.2f}%"


def stable_int(*parts):
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------- the wire

class FakeLLM:
    def __init__(self, ntok, envelope, trial_reasonings, sizes):
        self.ntok, self.env, self.trial, self.sizes = ntok, envelope, trial_reasonings, sizes
        self.faults = {}          # (cond, arm, date, role) -> {attempt: kind}
        self.log = []
        self._pad = {}

    def pad(self, n_tokens):
        if n_tokens not in self._pad:
            out, k = [], 0
            while self.ntok(" ".join(out)) < n_tokens:
                k += 1
                out.append(f"Offline filler sentence {k} standing in for model reasoning of realistic length.")
            self._pad[n_tokens] = " ".join(out)
        return self._pad[n_tokens]

    def __call__(self, request):
        body = json.loads(request.content)
        prompt = body["messages"][0]["content"]
        label_ctx = getattr(utils._provider_state, "context", None) or {}
        call_id, role = label_ctx.get("call_id"), label_ctx.get("role")
        label, arm, ymd, crole, att = call_id.split("-")
        att = int(att[1:])
        date = pd.Timestamp(ymd)
        cond = label.split("_")[0]
        plan = self.faults.get((cond, arm, date, crole), {})
        kind = plan.get(att, plan.get("all"))
        entry = {"i": len(self.log), "url": str(request.url), "call_id": call_id, "role": role, "label": label,
                 "arm": arm, "date": date, "attempt": att, "kind": kind,
                 "body": {k: v for k, v in body.items() if k != "messages"},
                 "prompt": prompt, "prompt_tokens_local": self.ntok(prompt) + JSON_MODE_OVERHEAD}
        self.log.append(entry)
        if kind == "http500":
            entry["status"] = 500
            return httpx.Response(500, headers={"content-type": "application/json"},
                                  content=json.dumps({"error": "offline fake 500", "success": False}).encode(),
                                  request=request)
        dash = str(date.date())
        if role == "generator":
            content = self.generation(prompt, label, arm, dash, kind)
        elif role == "reflector":
            content = self.reflection(prompt, kind)
        else:
            content = self.curation(prompt, kind)
        return self.answer(request, entry, content)

    def generation(self, prompt, label, arm, dash, kind):
        s = split(GEN_RE, prompt)
        universe = re.findall(r"^\s*\d+\. (\S+)", s["context"].split("\n新聞標題")[0], re.M)
        rng = np.random.default_rng(stable_int(label, arm, dash))
        scores = {u: round(float(x), 2) for u, x in zip(universe, rng.uniform(-1, 1, len(universe)))}
        ids = [parse_playbook_line(ln)["id"] for ln in bullet_lines(s["playbook"])]
        used = ids[:math.ceil(0.25 * len(ids))]
        reasoning = f"〔TRACE {arm} {dash}〕" + self.trial[stable_int(dash) % len(self.trial)]
        if kind == "incomplete":
            scores.pop(universe[-1])
        if kind == "out_of_universe":
            scores["9999"] = 0.1
        fa = json.dumps(scores)
        if kind == "duplicate_key":
            fa = fa[:-1] + f', "{universe[0]}": 0.5}}'
        return ('{"reasoning": ' + json.dumps(reasoning, ensure_ascii=False) + ', "bullet_ids": '
                + json.dumps(used) + ', "final_answer": ' + fa + "}")

    def reflection(self, prompt, kind):
        if kind == "not_json":
            return "offline fake: this is not json"
        s = split(REF_RE, prompt)
        _, arm, dash = next(t for t in TOKEN_RE.findall(s["trace"]) if t[0] == "TRACE")
        src = f"〔SRC {arm} {dash}〕"
        ids = re.findall(r"^\[([^\]]+)\] helpful=", s["bullets"], re.M)
        return json.dumps({"reasoning": src + " " + s["env"].split("\n")[0] + " " + self.pad(int(self.sizes["reflector"])),
                           "error_identification": "offline", "root_cause_analysis": "offline",
                           "correct_approach": "offline", "key_insight": src,
                           "bullet_tags": [{"id": i, "tag": "helpful"} for i in ids]}, ensure_ascii=False)

    def curation(self, prompt, kind):
        if kind == "not_json":
            return "offline fake: not json {"
        s = split(CUR_RE, prompt)
        _, arm, dash = next(t for t in TOKEN_RE.findall(s["recent_reflection"]) if t[0] == "SRC")
        titles = [re.sub(r" \([^()]*\)$", "", ln[len(dash) + 3:]) for ln in s["question_context"].split("\n")
                  if ln.startswith(f"[{dash}] ")]
        rng = np.random.default_rng(stable_int("cur", arm, dash))
        rng.shuffle(titles)
        content = f"〔FROM {arm} {dash}〕"
        for t in titles:
            if self.ntok(f"[sai-00000] helpful=0 harmful=0 :: {content}") >= self.sizes["growth"]:
                break
            content += " " + t
        obj = {"reasoning": self.pad(int(self.sizes["curator"])),
               "operations": [{"type": "ADD", "section": "strategies_and_insights", "content": content}]}
        if kind == "no_reasoning":
            obj.pop("reasoning")
        return json.dumps(obj, ensure_ascii=False)

    def answer(self, request, entry, content):
        body = copy.deepcopy(self.env)
        d = body["data"]
        i = entry["i"]
        d["id"] = d["generationId"] = f"offline-gen-{i}"
        ch = d["choices"][0]
        msg = ch["message"]
        msg["content"] = content
        msg.pop("reasoning", None)
        msg.pop("reasoning_details", None)
        ch["finish_reason"] = "stop"
        routing = msg["provider_metadata"]["gateway"]["routing"]
        routing["finalProvider"] = HOSTS[i % 2]
        routing["resolvedProvider"] = HOSTS[(i + 1) % 2]
        routing["modelAttemptCount"] = 1 + (i % 3 == 0)
        p, c = entry["prompt_tokens_local"], self.ntok(content) + 1
        d["usage"] = {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c,
                      "cost": round(p * 0.44e-6 + c * 1.32e-6, 10),
                      "prompt_tokens_details": {"cached_tokens": 0}, "completion_tokens_details": {"reasoning_tokens": 0}}
        entry.update({"status": 200, "content": content, "generation_id": d["id"], "usage": d["usage"],
                      "routing": {k: routing.get(k) for k in ("finalProvider", "resolvedProvider", "modelAttemptCount")}})
        return httpx.Response(200, headers={"content-type": "application/json"},
                              content=json.dumps(body, ensure_ascii=False).encode("utf-8"), request=request)


class _Wire(httpx.HTTPTransport):
    responder = None

    def handle_request(self, request):
        return type(self).responder(request)


class OfflineClineTransport(utils._ClineUnwrapTransport, _Wire):
    """The real unwrap-and-log transport; its parent request goes to FakeLLM instead of the network."""


class AnalyzerSpy:
    def __init__(self, inner, fake):
        self.inner, self.fake, self.calls = inner, fake, []

    def analyze(self, playbook, threshold, merge):
        before = len(self.fake.log)
        out = self.inner.analyze(playbook=playbook, threshold=threshold, merge=merge)
        self.calls.append({"threshold": threshold, "merge": merge, "requests_during": len(self.fake.log) - before})
        return out


# ---------------------------------------------------------------- setup

class Env:
    pass


def setup(args, out):
    env = Env()
    market.login()
    env.names = company_names()
    cfg = news.load_config()
    index, _ = news.build(cfg)
    env.contexts = roles.Contexts(index, cfg, env.names)
    tok = load_tokenizer()
    env.ntok = tok
    env.null = fb.load_null()
    env.budget = json.loads(BUDGET.read_text(encoding="utf-8"))
    trading = panel.trading_days()
    adj_open = panel.frame("adj_open")
    env.cond = {}
    for cond in args.conditions:
        dates = panel.decision_dates(*rp.CONDITIONS[cond])
        U = panel.universes(dates)
        pnl = alpha.build_alpha_panel(dates, adj_open, U)
        outcomes = {d: dict(zip(g["stock_id"], g["alpha_h10"])) for d, g in pnl.groupby("decision_date", sort=True)}
        cal = rp.TradingCalendar(trading[trading >= dates[0]])
        env.cond[cond] = {"dates": dates, "U": U, "pnl": pnl, "outcomes": outcomes, "cal": cal,
                          "pnl_by": pnl.set_index(["decision_date", "stock_id"])}
    env.trial_calls = [json.loads(x) for x in (TRIAL / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    env.trial_calls = [c for c in env.trial_calls if c["ok_call"]]
    trial_reasonings = [json.loads(c["response"])["reasoning"] for c in env.trial_calls]
    envelope = json.load(open(glob.glob(str(RESULTS / "format_trial/identity_20260914/identity_out/*_identity.json"))[0],
                              encoding="utf-8"))["body"]
    mid = env.budget["scenarios"]["mid"]
    env.fake = FakeLLM(tok, envelope, trial_reasonings,
                       {"reflector": mid["reflector_completion"], "curator": mid["curator_completion"], "growth": mid["growth"]})
    for cond, c in env.cond.items():
        for (arm, j, role), plan in FAULTS.items():
            env.fake.faults[(cond, arm, c["dates"][j], role)] = plan
    _Wire.responder = env.fake
    utils._ClineUnwrapTransport = OfflineClineTransport
    env.client = roles.make_client()
    env.analyzer = roles.make_analyzer()
    return env


# ---------------------------------------------------------------- one replay

def run_label(env, cond, fmt, out):
    label = f"{cond}_{fmt}"
    c = env.cond[cond]
    spy = AnalyzerSpy(env.analyzer, env.fake)
    mem = {"A0": rp.A0Memory(), "A1": rp.A1Window(), "A2": roles.AcePlaybook(analyzer=spy)}
    ldir = out / label
    live = roles.LiveRoles(env.client, env.contexts, env.names, env.null, c["dates"], fmt, ldir / "llm_logs",
                           mem["A2"], label)
    print(f"\n== replay {label}: {len(c['dates'])} decision dates", flush=True)
    res = rp.run_replay(c["cal"], c["dates"], c["U"], live.generate, live.reflect, live.curate,
                        rp.OutcomeStore(c["outcomes"], c["cal"]), rp.ARM_ORDER_SEED, memories=mem,
                        curator_check=live.curator_check)
    written = {}
    for arm in rp.ARMS:
        frames = [ap.score_rows(d, res.outcomes[(arm, d)]) for d in c["dates"]]
        scores = pd.concat([f for f in frames if len(f)], ignore_index=True)
        written[arm] = records.write_run(
            ldir / "records" / arm, c["pnl"], scores, arm=f"offline:{arm}", llm_meta=res.meta_rows[arm],
            extra={**rp.manifest_fields(), "fake_llm": True, "condition": cond,
                   "a1_window_format": fmt if arm == "A1" else None,
                   "provider_log": os.environ["ACE_PROVIDER_LOG"]})
    mem["A2"].bullet_table().to_parquet(ldir / "records" / "A2" / "playbook_bullets.parquet", index=False)
    with open(ldir / "records" / "A2" / "playbook_commits.jsonl", "w", encoding="utf-8") as f:
        for rec in mem["A2"].commits:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"label": label, "cond": cond, "fmt": fmt, "res": res, "mem": mem, "spy": spy, "written": written,
            "live": live, "dir": ldir}


# ---------------------------------------------------------------- checks

def verify(env, run):
    label, cond, fmt, res, mem = run["label"], run["cond"], run["fmt"], run["res"], run["mem"]
    c = env.cond[cond]
    dates, cal, pnl_by = c["dates"], c["cal"], c["pnl_by"]
    idx = {d: j for j, d in enumerate(dates)}
    reqs = [e for e in env.fake.log if e["label"] == label]
    gen = [e for e in reqs if e["role"] == "generator"]
    ref = [e for e in reqs if e["role"] == "reflector"]
    cur = [e for e in reqs if e["role"] == "curator"]
    out_of = lambda arm, d: res.outcomes[(arm, d)]
    a1_void = [out_of("A1", d)["voided"] for d in dates]
    commits = sorted(mem["A2"].commits, key=lambda r: r["matured_on"])
    rows = mem["A2"].bullet_rows
    harvest = {(h["arm"], h["decision_date"]): h for h in res.harvests}
    first_gen = {}
    for e in gen:
        first_gen.setdefault((e["arm"], e["date"]), e)
    a2_slot = {d: split(GEN_RE, first_gen[("A2", d)]["prompt"])["playbook"] for d in dates}

    def base(d):
        return env.contexts.get(d).context

    def members(d):
        return list(c["outcomes"][d].keys())

    def al(d, sid):
        return pnl_by.loc[(d, sid), "alpha_h10"]

    check(label, "P0 every request to http://offline.invalid/",
          lambda: (all(e["url"].startswith("http://offline.invalid/") for e in reqs) and len(reqs) > 0, len(reqs)))

    def p1a():
        bad = []
        for e in gen:
            if e["arm"] != "A0":
                continue
            s = split(GEN_RE, e["prompt"])
            if not (s["playbook"] == roles.EMPTY_PLAYBOOK and s["reflection"] == "(empty)" and s["question"] == QUESTION
                    and s["context"] == base(e["date"]) and not TOKEN_RE.search(e["prompt"])):
                bad.append(e["call_id"])
        return not bad, bad[:5]
    check(label, "P1a A0 prompt: empty playbook, (empty) reflection, QUESTION, exact context, no memory token", p1a)

    def p1b():
        bad = []
        for e in gen:
            if e["arm"] != "A1":
                continue
            d, j = e["date"], idx[e["date"]]
            s = split(GEN_RE, e["prompt"])
            if not (s["playbook"] == roles.EMPTY_PLAYBOOK and s["reflection"] == "(empty)" and s["question"] == QUESTION
                    and s["context"].startswith(base(d))):
                bad.append((e["call_id"], "slots"))
                continue
            rest = s["context"][len(base(d)):]
            kept, _ = ap.window_indices(j, a1_void)
            expected = [dates[i] for i in kept]
            toks = TOKEN_RE.findall(e["prompt"])
            want_toks = sorted(("TRACE", "A1", str(x.date())) for x in expected) if fmt == "with_reasoning" else []
            if sorted(toks) != want_toks:
                bad.append((e["call_id"], "tokens", toks[:3]))
                continue
            if not expected:
                if rest != "":
                    bad.append((e["call_id"], "window without maturity"))
                continue
            if not rest.startswith("\n\n" + roles.A1_WINDOW_HEADER):
                bad.append((e["call_id"], "header"))
                continue
            lines = rest.split("\n")
            heads = [(k, re.match(r"^決策日 (\d{4}-\d{2}-\d{2})（h=10 於 (\d{4}-\d{2}-\d{2}) 到期）$", ln))
                     for k, ln in enumerate(lines)]
            heads = [(k, m) for k, m in heads if m]
            if [pd.Timestamp(m.group(1)) for _, m in heads] != expected:
                bad.append((e["call_id"], "dates", [m.group(1) for _, m in heads]))
                continue
            for (k, m), x in zip(heads, expected):
                if pd.Timestamp(m.group(2)) != cal.shift(x, rp.DELAY):
                    bad.append((e["call_id"], "mature", m.group(2)))
                mem_ids = members(x)
                body_lines = lines[k + 2:k + 2 + len(mem_ids)]
                sc = out_of("A1", x)["scores"]
                for sid, ln in zip(mem_ids, body_lines):
                    parts = ln.split(" ")
                    if parts[0] != sid or parts[-2] != f"{sc[sid]:+.2f}" or parts[-1] != alpha_text(al(x, sid)):
                        bad.append((e["call_id"], "line", ln))
                        break
        return not bad, bad[:5]
    check(label, "P1b A1 prompt: only A1's own matured, non-void decisions, own scores, panel alpha", p1b)

    def text_at(t):
        done = [r for r in commits if pd.Timestamp(r["matured_on"]) <= t]
        return done[-1]["text_after"] if done else roles.EMPTY_PLAYBOOK

    def p1c():
        bad = []
        for e in gen:
            if e["arm"] != "A2":
                continue
            d = e["date"]
            s = split(GEN_RE, e["prompt"])
            toks = TOKEN_RE.findall(e["prompt"])
            att = out_of("A2", d)["attempts"][e["attempt"] - 1]
            if not (s["reflection"] == "(empty)" and s["question"] == QUESTION and s["context"] == base(d)
                    and s["playbook"] == text_at(d) and roles.sha256(s["playbook"]) == att.get("playbook_sha256")
                    and all(t[0] == "FROM" and t[1] == "A2" for t in toks)):
                bad.append(e["call_id"])
        return not bad, bad[:5]
    check(label, "P1c A2 prompt: its own playbook as committed by t, exact context, no other memory", p1c)

    def p1d():
        bad = [e["call_id"] for e in ref + cur if e["arm"] != "A2"]
        for e in ref:
            if e["kind"] is None or e["attempt"] >= 1:
                s = split(REF_RE, e["prompt"])
                t = [x for x in TOKEN_RE.findall(s["trace"]) if x[0] == "TRACE"]
                if t != [("TRACE", "A2", str(e["date"].date()))]:
                    bad.append((e["call_id"], t))
        for e in cur:
            s = split(CUR_RE, e["prompt"])
            t = {x for x in TOKEN_RE.findall(s["recent_reflection"]) if x[0] == "SRC"}
            if t != {("SRC", "A2", str(e["date"].date()))} or s["question_context"] != base(e["date"]):
                bad.append((e["call_id"], t))
        return not bad and len(ref) > 0 and len(cur) > 0, bad[:5]
    check(label, "P1d only A2 reaches Reflector/Curator, each on its own matured decision", p1d)

    def p2a():
        bad = []
        added = {r["bullet_id"]: pd.Timestamp(r["created_decision_date"]) for r in rows}
        for d in dates:
            for ln in bullet_lines(a2_slot[d]):
                toks = [t for t in TOKEN_RE.findall(ln)]
                bid = parse_playbook_line(ln)["id"]
                if len(toks) != 1 or toks[0][:2] != ("FROM", "A2") \
                        or cal.position(pd.Timestamp(toks[0][2])) > cal.position(d) - rp.DELAY \
                        or added.get(bid) is None or added[bid] > d:
                    bad.append((str(d.date()), ln[:80]))
        return not bad, bad[:5]
    check(label, "P2a every A2 bullet at t came from a decision <= t-11 sessions, added by t", p2a)

    def p2b():
        bad = []
        for d in dates:
            n_exp = sum(1 for r in rows if pd.Timestamp(r["created_decision_date"]) <= d
                        and not (r["deleted_decision_date"] and pd.Timestamp(r["deleted_decision_date"]) <= d))
            if len(bullet_lines(a2_slot[d])) != n_exp:
                bad.append((str(d.date()), len(bullet_lines(a2_slot[d])), n_exp))
        return not bad, bad[:5]
    check(label, "P2b A2 bullet count at t == added minus deduped by t", p2b)

    def p2c():
        t = dates[20]
        raised = []
        for snap in ((rp.PlaybookEntry("x", dates[10], dates[20], "b"),), (rp.PlaybookEntry("x", dates[5], dates[21], "b"),)):
            try:
                rp._assert_no_future_feedback("A2", snap, t, cal, rp.DELAY)
                raised.append(False)
            except AssertionError:
                raised.append(True)
        return all(raised), raised
    check(label, "P2c negative control: future feedback in a snapshot raises", p2c)

    def p3ab():
        bad, n = [], 0
        for e in ref:
            d = e["date"]
            s = split(REF_RE, e["prompt"])
            gt = s["ground_truth"].split("\n")
            m = re.match(r"^決策日 (\S+) 的評分與實現 α（h=10：(\S+) 開盤進場，(\S+) 開盤結算，(\S+) 收盤後揭露；", gt[0])
            first = c["pnl"][c["pnl"]["decision_date"] == d].iloc[0]
            hday = harvest[("A2", d)]["harvested_on"]
            if not m or pd.Timestamp(m.group(1)) != d or pd.Timestamp(m.group(2)) != first["entry_date"] \
                    or pd.Timestamp(m.group(3)) != first["mature_date_h10"] or pd.Timestamp(m.group(4)) != hday \
                    or first["mature_date_h10"] != hday:
                bad.append((e["call_id"], "header", gt[0]))
                continue
            mem_ids = members(d)
            sc = out_of("A2", d)["scores"]
            if len(gt) != 2 + len(mem_ids):
                bad.append((e["call_id"], "line count"))
                continue
            for sid, ln in zip(mem_ids, gt[2:]):
                parts = ln.split(" ")
                if parts[0] != sid or parts[-2] != f"{sc[sid]:+.2f}" or parts[-1] != alpha_text(al(d, sid)):
                    bad.append((e["call_id"], ln))
                    break
            x = np.array([sc[sid] for sid in mem_ids], dtype=float)
            a = np.array([al(d, sid) for sid in mem_ids], dtype=float)
            ok = ~np.isnan(x) & ~np.isnan(a)
            rho = float(spearmanr(x[ok], a[ok]).statistic)
            env_lines = s["env"].split("\n")
            m1 = re.match(r"^該決策日的 csIC（評分與實現 α 的 Spearman 相關，有效配對 (\d+) 檔）：([+-]\d\.\d{4})$", env_lines[0])
            m2 = re.search(r"分位數：(\d+\.\d)%（.*標準差 0\.144）$", env_lines[1])
            q = percentileofscore(np.round(env.null, 12), round(rho, 12), kind="mean")
            if not m1 or int(m1.group(1)) != int(ok.sum()) or abs(float(m1.group(2)) - rho) > 5e-5 + 1e-12:
                bad.append((e["call_id"], "csic", env_lines[0], rho))
            elif not m2 or abs(float(m2.group(1)) - q) > 0.05 + 1e-9:
                bad.append((e["call_id"], "quantile", env_lines[1], q))
            n += 1
        return not bad and n == len(ref) and n > 0, bad[:5]
    check(label, "P3a/b/c feedback: per-stock score and alpha, csIC and quantile recomputed independently", p3ab)

    def p3e():
        bad = []
        for e in ref:
            s = split(REF_RE, e["prompt"])
            txt = s["ground_truth"] + "\n" + s["env"]
            hday = harvest[("A2", e["date"])]["harvested_on"]
            if any(w in txt.lower() for w in VERDICT_WORDS):
                bad.append((e["call_id"], "verdict"))
            if any(pd.Timestamp(x) > hday for x in re.findall(r"\d{4}-\d{2}-\d{2}", txt)):
                bad.append((e["call_id"], "date after harvest"))
        return not bad, bad[:5]
    check(label, "P3e feedback holds no verdict word and no date after the harvest day", p3e)

    # P4
    meta = {arm: {r["decision_date"]: r for r in res.meta_rows[arm]} for arm in rp.ARMS}
    D = lambda j: dates[j]

    def reqs_for(role, arm, d):
        return [e for e in reqs if e["role"] == role and e["arm"] == arm and e["date"] == d]

    check(label, "P4a A2 void at j=7: skipped at maturity, no learning call, playbook unchanged that day", lambda: (
        out_of("A2", D(7))["voided"] and out_of("A2", D(7))["void_reason"] == "incomplete"
        and out_of("A2", D(7))["n_attempts"] == 3 and harvest[("A2", D(7))]["status"] == "skipped_void"
        and not reqs_for("reflector", "A2", D(7)) and not reqs_for("curator", "A2", D(7))
        and a2_slot[D(17)] == a2_slot[D(18)], harvest[("A2", D(7))]))
    check(label, "P4b A2 generation retry at j=10: 2 attempts, not void", lambda: (
        out_of("A2", D(10))["n_attempts"] == 2 and not out_of("A2", D(10))["voided"], out_of("A2", D(10))["n_attempts"]))

    def p4c():
        ok = out_of("A1", D(4))["voided"] and out_of("A1", D(4))["void_reason"] == "duplicate_key"
        for j in range(15, 20):
            r = meta["A1"][D(j)]
            listed = [pd.Timestamp(x) for x in json.loads(r["window_decision_dates"])]
            lo = j - rp.DELAY - rp.WINDOW_K + 1
            ok &= r["window_skipped_void"] == 1 and D(4) not in listed and all(dates.get_loc(x) >= lo for x in listed)
        return ok, [meta["A1"][D(j)]["window_decision_dates"] for j in range(15, 20)]
    check(label, "P4c A1 void at j=4: windows at j=15..19 leave the slot empty, nothing older pulled in", p4c)
    check(label, "P4d A1 out-of-universe once at j=9: 2 attempts, not void", lambda: (
        out_of("A1", D(9))["n_attempts"] == 2 and not out_of("A1", D(9))["voided"], out_of("A1", D(9))["n_attempts"]))

    plog = [json.loads(x) for x in Path(os.environ["ACE_PROVIDER_LOG"]).read_text(encoding="utf-8").splitlines()]
    plog = [x for x in plog if (x.get("call_id") or "").startswith(label + "-")]

    def p4e():
        lines = [x for x in plog if x["call_id"].startswith(f"{label}-A0-{D(30):%Y%m%d}-gen-")]
        o = out_of("A0", D(30))
        return (o["voided"] and o["void_reason"] == "call_error" and o["n_attempts"] == 3 and len(lines) == 3
                and all(x["http_status"] == 500 and x["provider_metadata"] is None for x in lines)), len(lines)
    check(label, "P4e A0 HTTP 500 x3 at j=30: void call_error, three 500 lines in the provider log", p4e)
    check(label, "P4f Reflector fails for j=15: no Curator, playbook unchanged that day", lambda: (
        harvest[("A2", D(15))]["reflector_status"] == "failed" and harvest[("A2", D(15))]["curator_status"] == "not_run"
        and len(reqs_for("reflector", "A2", D(15))) == 3 and not reqs_for("curator", "A2", D(15))
        and a2_slot[D(25)] == a2_slot[D(26)], harvest[("A2", D(15))]))
    check(label, "P4g Reflector retry for j=20: ok after 2, Curator ran, its bullet follows", lambda: (
        len(reqs_for("reflector", "A2", D(20))) == 2 and harvest[("A2", D(20))]["reflector_status"] == "ok"
        and harvest[("A2", D(20))]["curator_status"] == "ok"
        and any(r["source_matured_decision_date"] == str(D(20).date()) for r in rows), harvest[("A2", D(20))]))

    def p4h():
        h = harvest[("A2", D(25))]
        before = {parse_playbook_line(x)["id"]: parse_playbook_line(x) for x in bullet_lines(a2_slot[D(35)])}
        after = {parse_playbook_line(x)["id"]: parse_playbook_line(x) for x in bullet_lines(a2_slot[D(36)])}
        tagged = set(out_of("A2", D(25))["attempts"][-1].get("bullet_ids") or [])
        same = list(before) == list(after) and all(before[i]["content"] == after[i]["content"] for i in before)
        counts = all(after[i]["helpful"] - before[i]["helpful"] == (1 if i in tagged else 0) for i in before)
        return (h["curator_status"] == "failed" and len(reqs_for("curator", "A2", D(25))) == 3
                and not any(r["source_matured_decision_date"] == str(D(25).date()) for r in rows)
                and same and counts and len(tagged) > 0), (h, len(tagged))
    check(label, "P4h Curator fails for j=25: no bullet, only the Reflector's helpful counts moved", p4h)
    check(label, "P4i Curator retry for j=28: ok after 2", lambda: (
        len(reqs_for("curator", "A2", D(28))) == 2 and harvest[("A2", D(28))]["curator_status"] == "ok", harvest[("A2", D(28))]))

    # P5
    def p5a():
        if len(plog) != len(reqs):
            return False, (len(plog), len(reqs))
        bad = []
        for x, e in zip(plog, reqs):
            if x["call_id"] != e["call_id"] or x["role"] != e["role"]:
                bad.append((x["call_id"], e["call_id"]))
            elif e.get("status") == 200 and not (x["generation_id"] == e["generation_id"] and x["usage"] == e["usage"]
                                                   and all(x[k] == v for k, v in e["routing"].items())):
                bad.append((x["call_id"], "fields"))
        return not bad, bad[:5]
    check(label, "P5a one provider-log line per request, fields equal to the wire", p5a)

    def p5b():
        by_gen = {x["generation_id"]: x for x in plog if x.get("generation_id")}
        bad, n = [], 0
        for arm in rp.ARMS:
            _, m, _ = records.read_run(run["written"][arm])
            for col in ("attempts_json", "aux_attempts_json"):
                if col not in m:
                    continue
                for js in m[col].dropna():
                    for a in json.loads(js):
                        if not a.get("ok_call"):
                            continue
                        p = a.get("provider") or {}
                        x = by_gen.get(p.get("generation_id"))
                        n += 1
                        if x is None or any(p.get(k) != x[k] for k in ("finalProvider", "resolvedProvider", "modelAttemptCount")) \
                                or (p.get("usage") or {}).get("cost") != x["usage"]["cost"]:
                            bad.append((arm, a.get("call_id")))
        answered = sum(1 for e in reqs if e.get("status") == 200)
        return not bad and n == answered, (bad[:5], n, answered)
    check(label, "P5b provider fields and cost in attempts_json / aux_attempts_json read back from disk", p5b)

    def p5e():
        by_gen = {x["generation_id"]: x for x in plog if x.get("generation_id")}
        bad, n_rows, total_rows = [], 0, 0.0
        for arm in rp.ARMS:
            calls = pd.read_parquet(run["written"][arm] / "llm_calls.parquet")
            _, m, man = records.read_run(run["written"][arm])
            n_rows += len(calls)
            want = sum(1 for e in reqs if e["arm"] == arm)
            if len(calls) != want or man.get("llm_calls") != want:
                bad.append((arm, "rows", len(calls), want, man.get("llm_calls")))
            for r in calls.itertuples(index=False):
                x = by_gen.get(r.generation_id) if pd.notna(r.generation_id) else None
                if bool(r.ok_call):
                    if x is None or x["usage"]["cost"] != r.cost_usd:
                        bad.append((arm, r.call_id, "cost"))
                elif pd.notna(r.cost_usd):
                    bad.append((arm, r.call_id, "cost on a call with no answer"))
            total_rows += float(calls["cost_usd"].sum())
            md = m.set_index("decision_date")
            for col, rs in (("cost_usd", ("generator",)), ("aux_cost_usd", ("reflector", "curator"))):
                sums = calls[calls["role"].isin(rs)].groupby("decision_date")["cost_usd"].sum(min_count=1)
                for d, v in md[col].items():
                    s = sums.get(d, np.nan)
                    if pd.isna(v) != pd.isna(s) or (pd.notna(v) and abs(float(v) - float(s)) > 1e-12):
                        bad.append((arm, col, str(d.date()), v, s))
        total_log = sum(((x.get("usage") or {}).get("cost") or 0.0) for x in plog)
        if abs(total_log - total_rows) > 1e-9:
            bad.append(("total", total_log, total_rows))
        return not bad and n_rows == len(reqs), (bad[:5], n_rows, len(reqs))
    check(label, "P5e llm_calls.parquet: one row per request, cost == provider log, per-date sums == meta", p5e)
    check(label, "P5c request bodies: model, reasoning off, JSON mode, temperature 0, max_tokens 65536", lambda: (
        all(e["body"].get("model") == roles.MODEL and e["body"].get("reasoning") == {"enabled": False}
            and e["body"].get("response_format") == {"type": "json_object"} and e["body"].get("temperature") == 0.0
            and e["body"].get("max_tokens") == roles.MAX_TOKENS for e in reqs), reqs[0]["body"]))

    def p5d():
        bad = []
        for e in reqs:
            if e.get("status") != 200:
                continue
            files = glob.glob(str(run["dir"] / "llm_logs" / f"{e['role']}_{e['call_id']}_*.json"))
            if len(files) != 1 or (json.load(open(files[0], encoding="utf-8")).get("provider") or {}).get("generation_id") != e["generation_id"]:
                bad.append((e["call_id"], len(files)))
        return not bad, bad[:5]
    check(label, "P5d ACE llm_logs carry the provider record of each answered call", p5d)

    # P6
    def p6a():
        from collections import Counter
        cnt = Counter((e["arm"], e["date"]) for e in gen)
        return (all(cnt[(arm, d)] == out_of(arm, d)["n_attempts"] for arm in rp.ARMS for d in dates)
                and set(e["date"] for e in gen) <= set(dates)), None
    check(label, "P6a Generator requests per arm and date == recorded attempts, decision dates only", p6a)

    def p6b():
        bad = []
        for d in dates:
            r = [e["i"] for e in reqs if e["role"] == "reflector" and e["date"] == d]
            g = [e["i"] for e in gen if e["arm"] == "A2" and e["date"] == d]
            if r and g and max(g) > min(r):
                bad.append(str(d.date()))
        return not bad, bad
    check(label, "P6b no regeneration after reflection", p6b)

    def p6c():
        want = {d for d in dates if not out_of("A2", d)["voided"] and cal.shift(d, rp.DELAY) is not None
                and cal.shift(d, rp.DELAY) <= dates[-1]}
        got = {e["date"] for e in ref}
        n_void = sum(1 for j in range(len(dates) - rp.DELAY) if out_of("A2", dates[j])["voided"])
        return got == want and len(want) == len(dates) - rp.DELAY - n_void, (len(got), len(want), n_void)
    check(label, "P6c Reflector reached exactly the non-void A2 decisions maturing inside the window (n-11-voids)", p6c)

    def p6d():
        ok = True
        for arm in rp.ARMS:
            _, m, _ = records.read_run(run["written"][arm])
            ok &= len(m) == len(dates)
        bt = pd.read_parquet(run["written"]["A2"] / "playbook_bullets.parquet")
        return ok and len(bt) == sum(len(r["added"]) for r in commits), len(bt)
    check(label, "P6d records written and read back for three arms; bullet table rows == bullets added", p6d)
    check(label, "P6e analyzer once per successful Curator commit, merge=False, no request during it", lambda: (
        len(run["spy"].calls) == sum(1 for r in commits if r["curator_status"] == "ok")
        and all(x["merge"] is False and x["requests_during"] == 0 for x in run["spy"].calls),
        (len(run["spy"].calls), sum(len(r["removed_by_dedup"]) for r in commits))))

    def p8b():
        a0 = [e["prompt_tokens_local"] for e in gen if e["arm"] == "A0" and e["attempt"] == 1]
        b = env.budget["a0_prompt_tokens"][cond]
        got = {"n": len(a0), "mean": float(np.mean(a0)), "min": float(min(a0)), "max": float(max(a0))}
        return all(got[k] == b[k] for k in got), (got, {k: b[k] for k in got})
    check(label, "P8b A0 prompt tokens equal the budget's a0_prompt_tokens", p8b)

    return token_rows(env, run, reqs)


# ---------------------------------------------------------------- tokens

def token_rows(env, run, reqs):
    c = env.cond[run["cond"]]
    dates = c["dates"]
    idx = {d: j for j, d in enumerate(dates)}
    B = env.budget
    mid, high, T, W = B["scenarios"]["mid"], B["scenarios"]["high"], B["templates"], B["a1_window"]
    a0 = {idx[e["date"]]: e["prompt_tokens_local"] for e in reqs
          if e["role"] == "generator" and e["arm"] == "A0" and e["attempt"] == 1}
    rows = []
    for e in reqs:
        if e["attempt"] != 1:
            continue
        j = idx[e["date"]]
        row = {"label": run["label"], "cond": run["cond"], "fmt": run["fmt"], "arm": e["arm"], "role": e["role"],
               "decision_date": str(e["date"].date()), "tokens": e["prompt_tokens_local"]}
        if e["role"] == "generator":
            row["index"] = j
            if e["arm"] == "A0":
                row["estimate_mid"] = a0[j]
            elif e["arm"] == "A1":
                k = min(rp.WINDOW_K, max(0, j - rp.DELAY + 1))
                row["estimate_mid"] = a0[j] + (k * mid["a1_per_decision"] + W["header"] if k else 0)
                row["estimate_high"] = a0[j] + (k * high["a1_per_decision"] + W["header"] if k else 0)
                row["window_tokens"] = e["prompt_tokens_local"] - a0[j]
            else:
                row["estimate_mid"] = a0[j] + mid["growth"] * max(0, j - rp.DELAY + 1)
        else:
            h = j + rp.DELAY
            m = h - rp.DELAY + 1
            row["index"] = h
            if e["role"] == "reflector":
                row["estimate_mid"] = (T["reflector_template"] + T["question"] + mid["gen_completion"] + mid["final_answer"]
                                       + W["ground_truth_block"]["median"] + 30 + mid["bullets_used_share"] * mid["growth"] * (m - 1))
            else:
                ctx = env.ntok(env.contexts.get(e["date"]).context)
                row["estimate_mid"] = (T["curator_template"] + 80 + mid["reflector_completion"] + T["empty_playbook"]
                                       + mid["growth"] * (m - 1) + ctx)
        rows.append(row)
    thirds = np.array_split(np.arange(len(dates)), 3)
    phase = {int(i): p for p, arr in zip(("cold_start", "middle", "end"), thirds) for i in arr}
    for r in rows:
        r["phase"] = phase[r["index"]]
    return rows


def summarise_tokens(rows):
    df = pd.DataFrame(rows)
    out = {}
    for (label, arm, role), g in df.groupby(["label", "arm", "role"], sort=True):
        per = {}
        for p in ("cold_start", "middle", "end"):
            h = g[g["phase"] == p]
            if not len(h):
                continue
            d = {"n": int(len(h)), "first_date": h["decision_date"].min(), "last_date": h["decision_date"].max(),
                 "measured_mean": round(float(h["tokens"].mean()), 1), "measured_min": int(h["tokens"].min()),
                 "measured_max": int(h["tokens"].max()), "estimate_mid_mean": round(float(h["estimate_mid"].mean()), 1),
                 "measured_minus_mid": round(float((h["tokens"] - h["estimate_mid"]).mean()), 1)}
            if "estimate_high" in h and h["estimate_high"].notna().any():
                d["estimate_high_mean"] = round(float(h["estimate_high"].mean()), 1)
                d["measured_minus_high"] = round(float((h["tokens"] - h["estimate_high"]).mean()), 1)
            if "window_tokens" in h and h["window_tokens"].notna().any():
                d["window_tokens_mean"] = round(float(h["window_tokens"].mean()), 1)
                d["window_tokens_max"] = int(h["window_tokens"].max())
            per[p] = d
        out.setdefault(label, {})[f"{arm}/{role}"] = per
    return out


# ---------------------------------------------------------------- A1 candidates

def a1_candidates(env, out):
    c = env.cond["post"]
    dates, cal = c["dates"], c["cal"]
    slots = []
    for call in env.trial_calls:
        T = pd.Timestamp(call["decision_date"])
        obj = json.loads(call["response"])
        a = ap.analyse(call["response"], list(c["outcomes"][T]))
        slots.append(rp.Slot(T, False, a["scores"], c["outcomes"][T], reasoning=str(obj.get("reasoning", "")),
                             matured_on=cal.shift(T, rp.DELAY)))
    slots.sort(key=lambda s: s.decision_date)
    W = env.budget["a1_window"]

    def p8a():
        got = {}
        for fmt, key in (("compact", "per_decision_compact"), ("with_reasoning", "per_decision_with_reasoning")):
            toks = [env.ntok(roles.a1_decision_block(s, env.names, fmt)) for s in slots]
            win = env.ntok(roles.a1_window_block(tuple(slots), env.names, fmt))
            got[fmt] = {"min": min(toks), "median": float(np.median(toks)), "max": max(toks), "window5": win}
        want = {"compact": {k: W["per_decision_compact"][k] for k in ("min", "median", "max")} | {"window5": W["window5_compact"]},
                "with_reasoning": {k: W["per_decision_with_reasoning"][k] for k in ("min", "median", "max")} | {"window5": W["window5_with_reasoning"]}}
        return all(got[f][k] == want[f][k] for f in got for k in got[f]), (got, want)
    check("candidates", "P8a A1 candidate renderings reproduce the budget's 775 / 2,455 measurements", p8a)

    target = dates[15]
    if [s.decision_date for s in slots] != [dates[i] for i in range(5)] or \
            [s.matured_on for s in slots] != [dates[i] for i in range(11, 16)]:
        raise RuntimeError("trial decisions are not dates[0..4] maturing on dates[11..15]")
    sample_dir = out / "a1_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    info = {}
    for fmt in (roles.A1_WINDOW_FORMAT,):
        label = f"sample_{fmt}"
        live = roles.LiveRoles(env.client, env.contexts, env.names, env.null, dates, fmt, sample_dir / "llm_logs",
                               roles.AcePlaybook(), label)
        live.generate("A1", target, tuple(slots), 1)
        e = [x for x in env.fake.log if x["label"] == label][-1]
        s = split(GEN_RE, e["prompt"])
        window = s["context"][len(env.contexts.get(target).context):].lstrip("\n")
        (sample_dir / f"prompt_A1_{fmt}_{target.date()}.txt").write_text(e["prompt"], encoding="utf-8")
        (sample_dir / f"window_A1_{fmt}_{target.date()}.txt").write_text(window, encoding="utf-8")
        (sample_dir / f"decision_block_A1_{fmt}_{slots[0].decision_date.date()}.txt").write_text(
            roles.a1_decision_block(slots[0], env.names, fmt), encoding="utf-8")
        info[fmt] = {"decision_date": str(target.date()), "window_decisions": [str(s.decision_date.date()) for s in slots],
                     "prompt_tokens_with_json_overhead": e["prompt_tokens_local"],
                     "window_tokens": env.ntok(window),
                     "a0_prompt_tokens_same_date": env.ntok(GENERATOR_PROMPT.format(
                         roles.EMPTY_PLAYBOOK, "(empty)", QUESTION, env.contexts.get(target).context)) + JSON_MODE_OVERHEAD,
                     "files": sorted(str(p) for p in sample_dir.glob(f"*_{fmt}_*.txt"))}
    records.to_json(info, sample_dir / "samples.json")
    return info


# ---------------------------------------------------------------- main

def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--conditions", default="post,pre")
    ap_.add_argument("--formats", default=roles.A1_WINDOW_FORMAT)
    args = ap_.parse_args()
    args.conditions = args.conditions.split(",")
    args.formats = args.formats.split(",")
    out = RESULTS / "replay_offline" / f"run_{_dt.datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    os.environ["ACE_PROVIDER_LOG"] = str(out / "provider.jsonl")

    def p3c():
        cmp = fb.compare(fb._stats(np.load(fb.NULL_PATH)), fb.saved_summary())
        return all(v["equal"] for v in cmp.values()), cmp
    check("feedback", "P3c null array matches the known-answer summary exactly", p3c)

    env = setup(args, out)

    def p3d():
        U = [str(1000 + i) for i in range(50)]
        names = {}
        a = {u: 0.01 * i for i, u in enumerate(U)}
        up = fb.render("2026-05-04", "2026-05-05", "2026-05-19", U, names, {u: 0.02 * i - 0.5 for i, u in enumerate(U)}, a, env.null)
        down = fb.render("2026-05-04", "2026-05-05", "2026-05-19", U, names, {u: 0.5 - 0.02 * i for i, u in enumerate(U)}, a, env.null)
        sparse = fb.render("2026-05-04", "2026-05-05", "2026-05-19", U, names, {u: 0.1 * (i % 3) for i, u in enumerate(U)},
                           {u: (a[u] if i < 19 else float("nan")) for i, u in enumerate(U)}, env.null)
        nq = np.array([-0.1, 0.0, 0.0, 0.2])
        qs = [fb.null_quantile(x, nq) for x in (0.0, 0.2, 0.3, -0.2)]
        ok = ("：+1.0000" in up["environment_feedback"] and "分位數：100.0%" in up["environment_feedback"]
              and "：-1.0000" in down["environment_feedback"] and "分位數：0.0%" in down["environment_feedback"]
              and "無法計算" in sparse["environment_feedback"] and "分位數：無" in sparse["environment_feedback"]
              and sparse["n_pairs"] == 19 and qs == [0.5, 0.875, 1.0, 0.0])
        return ok, (up["environment_feedback"], down["environment_feedback"], sparse["environment_feedback"], qs)
    check("feedback", "P3d hand cases: +1 / -1 / too few pairs, and the mid-rank quantile", p3d)

    def p7():
        refused = []
        for bad in (None, "other", "compact"):
            try:
                roles.LiveRoles(env.client, env.contexts, env.names, env.null, env.cond[args.conditions[0]]["dates"],
                                bad, out / "guard", roles.AcePlaybook(), "guard")
                refused.append(False)
            except ValueError:
                refused.append(True)
        try:
            roles.ReasoningOffClient(utils.initialize_clients("clinepass")[0])
            refused.append(False)
        except ValueError:
            refused.append(True)
        return all(refused) and len(env.fake.log) == 0, refused
    check("guards", "P7 no A1 format / a client with SDK retries is refused, before any request", p7)

    samples = a1_candidates(env, out) if "post" in env.cond else None

    rows = []
    for cond in args.conditions:
        for fmt in args.formats:
            run = run_label(env, cond, fmt, out)
            rows += verify(env, run)
            for e in env.fake.log:
                if e["label"] == run["label"]:
                    e["prompt"] = None
                    e["content"] = None
    pd.DataFrame(rows).to_csv(out / "prompt_tokens_per_request.csv", index=False)
    tokens = {"source": "local DeepSeek-V4-Flash tokenizer + 26 JSON-mode tokens, first attempt of each call",
              "estimate": f"budget formula, mid scenario (A1 also high) from {BUDGET}",
              "fake_sizes_follow_mid": {"generator_completion": "real trial reasoning", "reflector_completion": "mid",
                                        "curator_growth": "mid", "bullets_used_share": 0.25},
              "phases": "decision-date index split in thirds; Reflector/Curator by harvest index",
              "a1_samples": samples, "summary": summarise_tokens(rows)}
    records.to_json(tokens, out / "tokens.json")
    records.to_json({"created_at": _dt.datetime.now().isoformat(timespec="seconds"), "git": records.git_state(),
                     "conditions": args.conditions, "formats": args.formats, "requests": len(env.fake.log),
                     "checks": CHECKS, "all_pass": all(x["pass"] for x in CHECKS)}, out / "checks.json")
    failed = [x for x in CHECKS if not x["pass"]]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} PASS" + ("" if not failed else "; FAILED:\n  " +
          "\n  ".join(f"[{x['label']}] {x['check']}" for x in failed)))
    print("written to", out)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
