"""Format-only trial of the A0 Generator prompt, non-reasoning mode. No IC, no evaluation.

    .venv/bin/python -m eval.twstock.format_trial [--dry-run]

A0 = the unmodified ace/prompts/generator.py template, the ACE empty playbook,
reflection "(empty)"; every task instruction lives in the question and context
slots. The call goes through ace.core.generator.Generator -> llm.timed_llm_call,
the same path an ACE run uses, with a thin recording wrapper around the client so
finish_reason, reasoning tokens, the raw response and the request parameters are kept.

Reasoning is requested off with the gateway field reasoning={"enabled": false}. That is
the shape Cline's own client sends to api.cline.bot (sdk/packages/llms routing rule
provider.cline.reasoning) for models whose catalog entry can turn reasoning off; the
models.dev entry for cline-pass/deepseek-v4-flash lists effort "none". Whether it takes
effect is checked on every call: any reasoning tokens or reasoning text stop the trial,
so the trial never continues in reasoning mode.

Measured per call: JSON parse, how many of the 50 stocks come back and which are
missing, whether every score is a number in [-1, 1] and what the out-of-range
forms are, latency, API prompt/completion tokens against the local DeepSeek
tokenizer, and the serving host (provider_metadata, logged by utils.py).

No retries at any layer (ACE_MAX_RETRIES=1, OpenAI SDK max_retries=0), so every
failure is recorded as it happened and MAX_REQUESTS is an exact HTTP count.

max_tokens stays at 65536: in reasoning mode (2026-09-13) the model spent 25,166
reasoning tokens on one prompt and smaller limits ended as HTTP 500 "empty response
content". A large limit costs nothing when the answer is short.
"""

import argparse
import datetime as _dt
import itertools
import json
import os
from pathlib import Path

os.environ.setdefault("ACE_MAX_RETRIES", "1")     # attempt < 1 never holds: no ACE-level retry

import httpx
import numpy as np
import pandas as pd
from finlab import data

from ace.ace import ACE
from ace.core.generator import Generator
from utils import extract_answer, initialize_clients, provider_log_path

from . import news, panel, records
from .arm_protocol import analyse
from .decision_input import build_decision_input

MODEL = "cline-pass/deepseek-v4-flash"
MAX_TOKENS = 65536
READ_TIMEOUT_S = 1800
REASONING = {"enabled": False}
RUNS_PER_DATE = 1
N_DATES = 5
MAX_REQUESTS = 5             # 5 dates x 1 run, no retries

QUESTION = """任務：對股票池中的每一檔股票給出方向性評分。

評分的意義：你預期該股從下一個交易日開盤起、未來數週的報酬，相對於股票池 50 檔等權平均報酬的高低。
- 評分是介於 -1 與 1 之間（含端點）的數字
- 正值表示預期優於股票池平均，負值表示預期劣於平均，0 表示沒有看法
- 絕對值越大表示看法越強

輸出規定：
- final_answer 必須是一個 JSON 物件，鍵是股票代號（字串），值是評分（數字）
- 必須涵蓋 Context 中股票池的全部 50 檔，不可遺漏，也不可加入股票池以外的代號
- 股票池中沒有相關新聞的股票也必須給分
- 範例格式：{"2330": 0.2, "2317": -0.1, "2454": 0.0}"""


def company_names():
    df = pd.DataFrame(data.get("company_basic_info"))
    return dict(zip(df["stock_id"].astype(str), df["公司簡稱"].astype(str)))


def render_context(inp, names):
    """Headlines listed once, each followed by the universe members it tags."""
    rank = {u["stock_id"]: u["universe_rank"] for u in inp.universe}
    tags, heads = {}, {}
    for sid, hs in inp.news_by_stock.items():
        for h in hs:
            tags.setdefault(h.aid, []).append(sid)
            heads[h.aid] = h
    order = sorted(heads, key=lambda a: (heads[a].date.normalize(), a))
    w = inp.news_window
    first_day = (w["calendar_days_after"] + pd.Timedelta(days=1)).date()
    lines = [
        f"決策日：{inp.decision_date.date()}（收盤後做決策；下一個交易日 {inp.entry_date.date()} 開盤進場）",
        "",
        "股票池：決策日市值前 50 大，依市值由大到小排列（代號 簡稱）",
    ]
    lines += [f"{u['universe_rank']:2d}. {u['stock_id']} {names.get(u['stock_id'], '')}".rstrip()
              for u in inp.universe]
    lines += [
        "",
        f"新聞標題：鉅亨網，{first_day} 至 {inp.decision_date.date()}（{w['first_session'].date()} 起共 5 個交易日，"
        f"含其間的非交易日），共 {len(order)} 則，依日期排序。",
        "每則格式為「[日期] 標題 (提及的股票池代號)」；只列出提及股票池內股票的新聞。",
    ]
    news_lines = [f"[{heads[a].date.date()}] {heads[a].title} "
                  f"({', '.join(sorted(tags[a], key=rank.get))})" for a in order]
    titles_only = [heads[a].title for a in order]
    return "\n".join(lines + news_lines), "\n".join(news_lines), "\n".join(titles_only)


class RecordingClient:
    """Pass-through OpenAI client: adds the reasoning-off field, keeps the last request and response."""

    def __init__(self, client):
        self._client = client
        self.chat = self
        self.completions = self
        self.requests = 0
        self.last = None
        self.last_params = None

    def create(self, **kwargs):
        if self.requests >= MAX_REQUESTS:
            raise RuntimeError(f"request budget of {MAX_REQUESTS} reached")
        self.requests += 1
        kwargs["extra_body"] = {**(kwargs.get("extra_body") or {}), "reasoning": REASONING}
        self.last_params = {k: v for k, v in kwargs.items() if k != "messages"}
        self.last = None
        self.last = self._client.chat.completions.create(**kwargs)
        return self.last


def provider_line(call_id):
    """Last provider-log record written for call_id (the log is shared by every call)."""
    path = Path(provider_log_path())
    hit = None
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("call_id") == call_id:
                hit = rec
    return hit


def stability(runs, universe):
    """Pairwise differences between runs of one date, on stocks scored in both."""
    good = [r for r in runs if r["analysis"]["scores"]]
    pairs = []
    for a, b in itertools.combinations(good, 2):
        sa, sb = a["analysis"]["scores"], b["analysis"]["scores"]
        common = [s for s in universe if s in sa and s in sb]
        if len(common) < 3:
            continue
        x = np.array([sa[s] for s in common])
        y = np.array([sb[s] for s in common])
        rho = pd.Series(x).corr(pd.Series(y), method="spearman")
        pairs.append({
            "runs": [a["rep"], b["rep"]], "common": len(common),
            "identical_response_text": a["response"] == b["response"],
            "identical_scores_share": float(np.mean(x == y)),
            "mean_abs_diff": float(np.mean(np.abs(x - y))),
            "max_abs_diff": float(np.max(np.abs(x - y))),
            "sign_agreement": float(np.mean(np.sign(x) == np.sign(y))),
            "spearman": None if pd.isna(rho) else float(rho),
        })
    return {"valid_runs": len(good), "pairs": pairs}


def load_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V4-Flash")
    return lambda s: len(tok(s, add_special_tokens=False)["input_ids"])


def analyse_selftest():
    """The parser is checked on known strings before any call is spent."""
    U = [str(1000 + i) for i in range(50)]
    full = {s: 0.1 for s in U}
    cases = {
        "object": (json.dumps({"reasoning": "r", "bullet_ids": [], "final_answer": full}),
                   lambda a: a["final_answer_form"] == "object" and a["value_forms"] == {"in_range": 50}),
        "json_string": (json.dumps({"final_answer": json.dumps(full)}),
                        lambda a: a["final_answer_form"] == "json_string" and a["n_valid_ids"] == 50),
        "forms": (json.dumps({"final_answer": {**{s: 0 for s in U[:46]}, U[46]: "0.3", U[47]: "30%",
                                               U[48]: 1.5, "9999": 0.1}}),
                  lambda a: a["missing"] == [U[49]] and a["extra_keys"] == ["9999"]
                  and a["value_forms"] == {"in_range": 46, "numeric_string": 1, "percent_string": 1,
                                           "number_out_of_range": 1}),
        "duplicate": ('{"final_answer": {"1000": 0.1, "1000": 0.2}}', lambda a: a["duplicate_keys"] == ["1000"]),
        "broken": ('{"final_answer": {"1000": 0.1', lambda a: not a["json_ok"]),
    }
    for name, (raw, ok) in cases.items():
        a = analyse(raw, U)
        if not ok(a):
            raise AssertionError(f"analyse self-test failed on {name}: {a}")
    print("analyse self-test: 5/5 cases pass")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(records.REPO_ROOT / "results" / "format_trial"))
    ap.add_argument("--dry-run", action="store_true", help="render prompts and count tokens; no LLM call")
    args = ap.parse_args()
    analyse_selftest()

    cfg = news.load_config()
    index, info = news.build(cfg)
    names = company_names()
    dates = list(panel.decision_dates("2026-04-27", "2026-08-26"))[:N_DATES]
    ds_tokens = load_tokenizer()
    import tiktoken
    cl100k = tiktoken.get_encoding("cl100k_base")

    run_dir = Path(args.out) / f"trial_{_dt.datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    playbook = ACE._initialize_empty_playbook(None)
    base_client = None if args.dry_run else initialize_clients("clinepass")[0].with_options(
        max_retries=0, timeout=httpx.Timeout(READ_TIMEOUT_S, connect=10.0))
    client = RecordingClient(base_client)
    gen = Generator(client, "clinepass", MODEL, max_tokens=MAX_TOKENS)

    from ace.prompts.generator import GENERATOR_PROMPT
    calls, per_date = [], {}
    stop = None
    for T in dates:
        inp = build_decision_input(index, T, cfg)
        universe = [u["stock_id"] for u in inp.universe]
        context, news_block, titles_only = render_context(inp, names)
        prompt = GENERATOR_PROMPT.format(playbook, "(empty)", QUESTION, context)
        (run_dir / f"prompt_{T.date()}.txt").write_text(prompt, encoding="utf-8")
        tok = {
            "prompt_deepseek_local": ds_tokens(prompt),
            "prompt_cl100k_local": len(cl100k.encode(prompt)),
            "news_block_format3_deepseek": ds_tokens(news_block),
            "titles_only_deepseek": ds_tokens(titles_only),
            "n_headlines": news_block.count("\n") + 1 if news_block else 0,
        }
        print(f"\n== {T.date()}: {tok}")
        if args.dry_run:
            continue
        runs = []
        for rep in range(1, RUNS_PER_DATE + 1):
            if client.requests >= MAX_REQUESTS:
                stop = "request budget reached"
                break
            call_id = f"test_fmt_{T.date()}_r{rep}"
            before = client.requests
            rec = {"decision_date": str(T.date()), "rep": rep, "call_id": call_id, "tokens_local": tok}
            t0 = _dt.datetime.now()
            try:
                response, bullet_ids, info_ = gen.generate(
                    question=QUESTION, playbook=playbook, context=context, reflection="(empty)",
                    use_json_mode=True, call_id=call_id, log_dir=str(run_dir / "llm_logs"))
                raw = client.last.model_dump() if client.last is not None else {}
                (run_dir / f"raw_{T.date()}_r{rep}.json").write_text(
                    json.dumps(raw, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
                ch = (raw.get("choices") or [{}])[0]
                msg = ch.get("message") or {}
                usage = raw.get("usage") or {}
                reasoning_text = msg.get("reasoning") or ""
                if reasoning_text:
                    (run_dir / f"reasoning_{T.date()}_r{rep}.txt").write_text(reasoning_text, encoding="utf-8")
                prov = info_.get("provider") or {}
                rec.update({
                    "ok_call": True,
                    "request_params": client.last_params,
                    "response": response,
                    "call_time_s": info_.get("call_time"),
                    "total_time_s": info_.get("total_time"),
                    "api_prompt_tokens": usage.get("prompt_tokens"),
                    "api_completion_tokens": usage.get("completion_tokens"),
                    "api_reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                    "api_cached_prompt_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                    "completion_content_deepseek_local": ds_tokens(response),
                    "completion_reasoning_deepseek_local": ds_tokens(reasoning_text) if reasoning_text else 0,
                    "finish_reason": ch.get("finish_reason"),
                    "reasoning_chars": len(reasoning_text),
                    "model_echoed": raw.get("model"),
                    "generation_id": prov.get("generation_id"),
                    "final_provider": prov.get("finalProvider"),
                    "resolved_provider": prov.get("resolvedProvider"),
                    "model_attempt_count": prov.get("modelAttemptCount"),
                    "bullet_ids": bullet_ids,
                    "ace_extract_answer_head": str(extract_answer(response))[:200],
                })
                rec["analysis"] = analyse(response, universe)
            except Exception as e:
                prov = provider_line(call_id) or {}
                rec.update({"ok_call": False, "error": f"{type(e).__name__}: {str(e)[:300]}",
                            "request_params": client.last_params, "response": None,
                            "http_status": prov.get("http_status"), "error_body": prov.get("error_body"),
                            "generation_id": prov.get("generation_id"),
                            "final_provider": prov.get("finalProvider"),
                            "resolved_provider": prov.get("resolvedProvider"),
                            "model_attempt_count": prov.get("modelAttemptCount"),
                            "analysis": analyse("", universe)})
                (run_dir / f"raw_{T.date()}_r{rep}_failure.json").write_text(
                    json.dumps({"exception": repr(e), "provider_record": prov}, ensure_ascii=False,
                               indent=1, default=str), encoding="utf-8")
            rec["http_requests"] = client.requests - before
            rec["wall_s"] = (_dt.datetime.now() - t0).total_seconds()
            a = rec["analysis"]
            print(f"   rep {rep}: call_ok={rec['ok_call']} finish={rec.get('finish_reason')} "
                  f"json={a['json_ok']} form={a['final_answer_form']} returned={a['n_returned']} "
                  f"valid={a['n_valid_ids']} forms={a['value_forms']} missing={len(a['missing'])} "
                  f"api_prompt={rec.get('api_prompt_tokens')} local_prompt={tok['prompt_deepseek_local']} "
                  f"completion={rec.get('api_completion_tokens')} local_completion="
                  f"{rec.get('completion_content_deepseek_local')} reasoning={rec.get('api_reasoning_tokens')} "
                  f"t={rec.get('call_time_s')} host={rec.get('final_provider')} err={rec.get('error')}")
            runs.append(rec)
            calls.append(rec)
            with open(run_dir / "calls.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            if rec["ok_call"] and ((rec.get("api_reasoning_tokens") or 0) > 0 or rec.get("reasoning_chars")):
                stop = (f"reasoning was not disabled: reasoning_tokens={rec.get('api_reasoning_tokens')}, "
                        f"reasoning_chars={rec.get('reasoning_chars')}")
                break
        per_date[str(T.date())] = {"tokens_local": tok, "stability": stability(runs, universe)}
        if stop:
            print("STOPPED:", stop)
            break

    ok = [c for c in calls if c["ok_call"]]
    summary = {
        "git": records.git_state(), "model": MODEL, "max_tokens": MAX_TOKENS, "json_mode": True,
        "reasoning_request": REASONING, "temperature": 0.0, "runs_per_date": RUNS_PER_DATE,
        "dates": [str(d.date()) for d in dates], "dry_run": args.dry_run,
        "http_requests": client.requests, "stopped": stop, "question": QUESTION,
        "news_config_hash": info["config_hash"],
        "calls": len(calls), "calls_ok": len(ok),
        "json_ok": sum(c["analysis"]["json_ok"] for c in calls),
        "all_50_in_range": sum(c["analysis"]["n_valid_ids"] == 50 and
                               c["analysis"]["value_forms"].get("in_range", 0) == 50 and
                               not c["analysis"]["extra_keys"] for c in calls),
        "finish_reasons": pd.Series([c.get("finish_reason") for c in calls], dtype=object).value_counts(dropna=False).to_dict(),
        "final_answer_forms": pd.Series([c["analysis"]["final_answer_form"] for c in calls], dtype=object).value_counts(dropna=False).to_dict(),
        "final_providers": [c.get("final_provider") for c in calls],
        "per_date": per_date,
        "latency_s": [c.get("call_time_s") for c in calls],
        "tokens_api_vs_local": [
            {"date": c["decision_date"], "api_prompt": c.get("api_prompt_tokens"),
             "local_prompt": c["tokens_local"]["prompt_deepseek_local"],
             "api_completion": c.get("api_completion_tokens"),
             "api_reasoning": c.get("api_reasoning_tokens"),
             "local_completion_content": c.get("completion_content_deepseek_local")} for c in calls],
    }
    records.to_json(summary, run_dir / "summary.json")
    print(json.dumps({k: summary[k] for k in ("http_requests", "stopped", "calls", "calls_ok", "json_ok",
                                              "all_50_in_range", "finish_reasons", "final_answer_forms",
                                              "final_providers")},
                     ensure_ascii=False, default=str, indent=2))
    print("written to", run_dir)


if __name__ == "__main__":
    main()
