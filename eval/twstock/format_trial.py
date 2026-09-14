"""Format-only trial of the A0 Generator prompt. No IC, no evaluation.

    .venv/bin/python -m eval.twstock.format_trial

A0 = the unmodified ace/prompts/generator.py template, the ACE empty playbook,
reflection "(empty)"; every task instruction lives in the question and context
slots. The call goes through ace.core.generator.Generator -> llm.timed_llm_call,
the same path an ACE run uses, with a thin recording wrapper around the client so
finish_reason, reasoning tokens and the raw usage block are kept.

Measured per call: JSON parse, how many of the 50 stocks come back and which are
missing, whether every score is a number in [-1, 1] and what the out-of-range
forms are, run-to-run differences within a date, API prompt tokens against the
local DeepSeek tokenizer, and latency.

No retries at any layer (ACE_MAX_RETRIES=1, OpenAI SDK max_retries=0), so every
failure is recorded as it happened and MAX_REQUESTS is an exact HTTP count.

max_tokens is 65536: on 2026-09-13 the model spent 25,166 reasoning tokens on the
2026-04-27 prompt, so 4096 and 16384 both ended with an empty body that the
gateway reports as HTTP 500 "empty response content".
"""

import argparse
import datetime as _dt
import itertools
import json
import os
import re
from pathlib import Path

os.environ.setdefault("ACE_MAX_RETRIES", "1")     # attempt < 1 never holds: no ACE-level retry

import httpx
import numpy as np
import pandas as pd
from finlab import data

from ace.ace import ACE
from ace.core.generator import Generator
from utils import extract_answer, initialize_clients

from . import news, panel, records
from .decision_input import build_decision_input

MODEL = "cline-pass/deepseek-v4-flash"
MAX_TOKENS = 65536           # reasoning alone used 25,166 on 2026-04-27; see docstring
READ_TIMEOUT_S = 1800        # 65536 tokens at the measured ~109 tok/s is ~600 s
RUNS_PER_DATE = 3
N_DATES = 5
MAX_REQUESTS = 14            # 15 runs, one reused from the identical 65536 probe

QUESTION = """任務：對股票池中的每一檔股票給出方向性評分。

評分的意義：你預期該股從下一個交易日開盤起、未來 10 個交易日的報酬，相對於股票池 50 檔等權平均報酬的高低。
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
    """Pass-through OpenAI client that keeps the last raw response."""

    def __init__(self, client):
        self._client = client
        self.chat = self
        self.completions = self
        self.requests = 0
        self.last = None

    def create(self, **kwargs):
        if self.requests >= MAX_REQUESTS:
            raise RuntimeError(f"request budget of {MAX_REQUESTS} reached")
        self.requests += 1
        self.last = None
        self.last = self._client.chat.completions.create(**kwargs)
        return self.last


NUM_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)$")


def classify_value(v):
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, (int, float)):
        return "in_range" if -1 <= v <= 1 else "number_out_of_range"
    if v is None:
        return "null"
    if isinstance(v, str):
        s = v.strip()
        if NUM_RE.match(s):
            return "numeric_string"
        if s.endswith("%") and NUM_RE.match(s[:-1].strip()):
            return "percent_string"
        return "text"
    return "nested_" + type(v).__name__


def analyse(raw, universe):
    out = {"json_ok": False, "top_keys": None, "final_answer_form": None, "n_returned": 0,
           "n_valid_ids": 0, "missing": [], "extra_keys": [], "duplicate_keys": [],
           "value_forms": {}, "out_of_range_examples": [], "scores": {}}
    dups = []

    def hook(pairs):
        d = {}
        for k, v in pairs:
            if k in d:
                dups.append(k)
            d[k] = v
        return d

    try:
        obj = json.loads(raw, object_pairs_hook=hook)
    except Exception as e:
        out["json_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        return out
    if not isinstance(obj, dict):
        out["json_error"] = f"top level is {type(obj).__name__}"
        return out
    out["json_ok"] = True
    out["top_keys"] = sorted(obj)
    fa = obj.get("final_answer")
    if isinstance(fa, dict):
        out["final_answer_form"] = "object"
        scores = fa
    elif isinstance(fa, str):
        try:
            inner = json.loads(fa, object_pairs_hook=hook)
            out["final_answer_form"] = "json_string" if isinstance(inner, dict) else "string_other"
            scores = inner if isinstance(inner, dict) else None
        except Exception:
            out["final_answer_form"] = "text"
            scores = None
    elif fa is None:
        out["final_answer_form"] = "missing"
        scores = None
    else:
        out["final_answer_form"] = type(fa).__name__
        scores = None
    out["duplicate_keys"] = dups
    if scores is None:
        return out

    members = set(universe)
    keys = {str(k).strip(): v for k, v in scores.items()}
    out["n_returned"] = len(keys)
    valid = {k: v for k, v in keys.items() if k in members}
    out["n_valid_ids"] = len(valid)
    out["missing"] = [s for s in universe if s not in valid]
    out["extra_keys"] = [k for k in keys if k not in members]
    forms = {}
    for k, v in valid.items():
        f = classify_value(v)
        forms[f] = forms.get(f, 0) + 1
        if f != "in_range" and len(out["out_of_range_examples"]) < 10:
            out["out_of_range_examples"].append({k: v})
        if f == "in_range":
            out["scores"][k] = float(v)
    out["value_forms"] = forms
    return out


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
    ap.add_argument("--reuse-probe", default=None,
                    help="JSON of an earlier identical request (same prompt, model, params) used as "
                         "--reuse-date rep 1 instead of re-sending it")
    ap.add_argument("--reuse-date", default="2026-04-27")
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
    reuse = json.loads(Path(args.reuse_probe).read_text(encoding="utf-8")) if args.reuse_probe else None
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
            print(prompt[:2200])
            print("   ...")
            print(prompt[-900:])
            continue
        runs = []
        for rep in range(1, RUNS_PER_DATE + 1):
            if reuse and rep == 1 and str(T.date()) == args.reuse_date:
                old_prompt = Path(args.reuse_probe).parent / f"prompt_{T.date()}.txt"
                if not old_prompt.exists() or old_prompt.read_text(encoding="utf-8") != prompt:
                    raise SystemExit(f"reuse refused: the prompt rendered now differs from {old_prompt}")
                u = reuse.get("usage") or {}
                rec = {"decision_date": str(T.date()), "rep": rep, "tokens_local": tok,
                       "source": f"reused probe {args.reuse_probe}", "ok_call": bool(reuse.get("ok")),
                       "response": reuse.get("content"), "call_time_s": reuse.get("elapsed_s"),
                       "total_time_s": reuse.get("elapsed_s"),
                       "api_prompt_tokens": u.get("prompt_tokens"),
                       "api_completion_tokens": u.get("completion_tokens"),
                       "api_reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                       "finish_reason": reuse.get("finish_reason"), "reasoning_chars": reuse.get("reasoning_chars"),
                       "reasoning_mentions_oververbosity":
                           "oververbosity" in (reuse.get("reasoning_tail") or "").lower(),
                       "model_echoed": reuse.get("echoed_model"), "http_requests": 0}
                rec["analysis"] = analyse(rec["response"] or "", universe)
                print(f"   rep {rep}: reused probe, json={rec['analysis']['json_ok']} "
                      f"valid={rec['analysis']['n_valid_ids']} forms={rec['analysis']['value_forms']}")
                runs.append(rec)
                calls.append(rec)
                with open(run_dir / "calls.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                continue
            if client.requests >= MAX_REQUESTS:
                stop = "request budget reached"
                break
            before = client.requests
            rec = {"decision_date": str(T.date()), "rep": rep, "tokens_local": tok}
            t0 = _dt.datetime.now()
            try:
                response, bullet_ids, info_ = gen.generate(
                    question=QUESTION, playbook=playbook, context=context, reflection="(empty)",
                    use_json_mode=True, call_id=f"test_fmt_{T.date()}_r{rep}", log_dir=str(run_dir / "llm_logs"))
                raw = client.last.model_dump() if client.last is not None else {}
                ch = (raw.get("choices") or [{}])[0]
                usage = raw.get("usage") or {}
                reasoning_text = (ch.get("message") or {}).get("reasoning") or ""
                (run_dir / f"reasoning_{T.date()}_r{rep}.txt").write_text(reasoning_text, encoding="utf-8")
                rec["reasoning_mentions_oververbosity"] = "oververbosity" in reasoning_text.lower()
                rec.update({
                    "ok_call": True,
                    "response": response,
                    "call_time_s": info_.get("call_time"),
                    "total_time_s": info_.get("total_time"),
                    "api_prompt_tokens": info_.get("prompt_num_tokens"),
                    "api_completion_tokens": info_.get("response_num_tokens"),
                    "api_reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                    "finish_reason": ch.get("finish_reason"),
                    "reasoning_chars": len(((ch.get("message") or {}).get("reasoning")) or ""),
                    "model_echoed": raw.get("model"),
                    "bullet_ids": bullet_ids,
                    "ace_extract_answer_head": str(extract_answer(response))[:200],
                })
                rec["analysis"] = analyse(response, universe)
            except Exception as e:
                rec.update({"ok_call": False, "error": f"{type(e).__name__}: {str(e)[:300]}",
                            "response": None, "analysis": analyse("", universe)})
            rec["http_requests"] = client.requests - before
            rec["wall_s"] = (_dt.datetime.now() - t0).total_seconds()
            a = rec["analysis"]
            print(f"   rep {rep}: call_ok={rec['ok_call']} finish={rec.get('finish_reason')} "
                  f"json={a['json_ok']} form={a['final_answer_form']} returned={a['n_returned']} "
                  f"valid={a['n_valid_ids']} forms={a['value_forms']} missing={len(a['missing'])} "
                  f"api_prompt={rec.get('api_prompt_tokens')} completion={rec.get('api_completion_tokens')} "
                  f"reasoning={rec.get('api_reasoning_tokens')} t={rec.get('call_time_s')}")
            runs.append(rec)
            calls.append(rec)
            with open(run_dir / "calls.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        per_date[str(T.date())] = {"tokens_local": tok, "stability": stability(runs, universe)}
        if stop:
            break

    ok = [c for c in calls if c["ok_call"]]
    summary = {
        "git": records.git_state(), "model": MODEL, "max_tokens": MAX_TOKENS, "json_mode": True,
        "temperature": 0.0, "runs_per_date": RUNS_PER_DATE, "dates": [str(d.date()) for d in dates],
        "http_requests": client.requests, "stopped": stop, "question": QUESTION,
        "news_config_hash": info["config_hash"],
        "calls": len(calls), "calls_ok": len(ok),
        "json_ok": sum(c["analysis"]["json_ok"] for c in calls),
        "all_50_in_range": sum(c["analysis"]["n_valid_ids"] == 50 and
                               c["analysis"]["value_forms"].get("in_range", 0) == 50 and
                               not c["analysis"]["extra_keys"] for c in calls),
        "finish_reasons": pd.Series([c.get("finish_reason") for c in calls]).value_counts(dropna=False).to_dict(),
        "final_answer_forms": pd.Series([c["analysis"]["final_answer_form"] for c in calls]).value_counts(dropna=False).to_dict(),
        "per_date": per_date,
        "latency_s": [c.get("call_time_s") for c in calls],
        "api_vs_local_prompt_tokens": [
            {"date": c["decision_date"], "rep": c["rep"], "api": c.get("api_prompt_tokens"),
             "deepseek_local": c["tokens_local"]["prompt_deepseek_local"],
             "cl100k_local": c["tokens_local"]["prompt_cl100k_local"]} for c in calls],
    }
    records.to_json(summary, run_dir / "summary.json")
    print(json.dumps({k: summary[k] for k in ("http_requests", "stopped", "calls", "calls_ok", "json_ok",
                                              "all_50_in_range", "finish_reasons", "final_answer_forms")},
                     ensure_ascii=False, default=str, indent=2))
    print("written to", run_dir)


if __name__ == "__main__":
    main()
