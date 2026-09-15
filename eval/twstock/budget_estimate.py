"""Inputs and arithmetic for the v9.3 §5.2.1 compute budget, per arm. No LLM call.

    .venv/bin/python -m eval.twstock.budget_estimate [--trial DIR]

READ CAVEATS FIRST. The whole table assumes A2 makes about 3 calls per decision date,
which rests on v9 §10.1 (二) max_num_rounds = 1 (decided 2026-09-14). Were multi-round
refinement restored (the ACE loop's worst case 14 calls per decision point) the whole
table would be invalid. Latency p90 is an interpolation over 5 calls and is not a tail estimate.

Measured here: decision-date counts from the trading calendar; the A0 prompt of every
decision date in both windows, rendered by the trial code and counted with the local
DeepSeek-V4-Flash tokenizer (+26 JSON-mode tokens, 5/5 exact in the trial); an A1 window
rendered from the trial's real scores, realized alpha and reasoning; ACE template sizes;
Reflector and Curator output sizes and playbook growth from the earlier ACE runs on disk.

Assumed, each with a low / mid / high value printed next to the result: the A2 playbook
growth per curator call, how much longer Reflector/Curator output gets in non-reasoning
mode, the share of the playbook passed as "bullets used", and a retry allowance.
The A1 window is no longer a scenario variable: its format is decided (2026-09-15,
arm_protocol.A1_WINDOW_FORMAT = "with_reasoning", one call per date), so every scenario uses
the measured per-decision median with reasoning (2,455 tokens). The compact and two-call
variants are gone from the table.

Arm protocol assumed (arm-interleaving-protocol): A0 one generation per date; A1 one
generation per date whose context carries the last 5 matured decisions with their reasoning;
A2 one generation per date plus, from the 12th date on, one Reflector and one Curator
call for the decision that matured - no regeneration, no multi-round refinement.
Latency = completion tokens x seconds per completion token measured in the trial;
prefill time is not modelled (the trial prompts were all ~12k, so it cannot be separated).
So the A1 window changes prompt tokens and cost here, not hours.
"""

import argparse
import datetime as _dt
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ace.ace import ACE
from ace.prompts.curator import CURATOR_PROMPT
from ace.prompts.generator import GENERATOR_PROMPT
from ace.prompts.reflector import REFLECTOR_PROMPT

from . import alpha, market, news, panel, records
from .arm_protocol import A1_CALLS_PER_DECISION, A1_WINDOW_FORMAT, analyse
from .decision_input import build_decision_input
from .format_trial import QUESTION, company_names, load_tokenizer, render_context
from .replay import CONDITIONS          # the only place the two phase-one windows are written

SEEDS = 3
DELAY = 11                    # h=10 feedback of date i is usable at date i+11 (v9 §5.1)
WINDOW_K = 5
JSON_MODE_OVERHEAD = 26
PRICE_IN, PRICE_OUT = 0.44e-6, 1.32e-6      # the reference price the usage "cost" field meters
RESULTS = records.REPO_ROOT / "results"
DEEPSEEK_RUNS = ["ace_run_20260903_113410_finer_offline", "ace_run_20260903_120152_finer_offline",
                 "ace_run_20260903_121716_finer_offline", "ace_run_20260903_160524_twstock_offline"]
QWEN_RUN = "ace_run_20260902_002452_finer_offline"
REASONING_TRIAL = "trial_20260913_233145"   # same 04-27 prompt, reasoning mode

CAVEATS = {
    "a2_calls_per_decision": (
        "THE WHOLE TABLE assumes A2 makes about 3 calls per decision date (1 generation, then 1 Reflector + "
        "1 Curator per maturity, no regeneration). That rests on v9 §10.1 (二) max_num_rounds = 1, decided "
        "2026-09-14. Were multi-round refinement restored (the ACE loop's worst case 14 calls per decision "
        "point) the whole table would be invalid - A2, the totals and the v8 bridge alike - not merely A2 a little higher."),
    "p90_and_high_scenario": (
        "Latency p90 is an interpolation over 5 trial calls, not a tail estimate. The high scenario takes the "
        "generation length as the maximum of the same 5 calls, and Reflector/Curator lengths as p90 of FiNER "
        "runs (another task). Every figure that cites p90 or the high scenario carries this limit."),
    "prefill_not_modelled": (
        "Time is completion tokens x seconds per completion token from ~12k-token trial prompts; the extra "
        "prefill of longer A1 and A2 prompts is not modelled - so A1's reasoning window (~12.5k tokens once "
        "full) raises prompt tokens and cost in this table but not its hours."),
}
HIGH_LIMIT = "p90/max over 5 trial calls; not a tail estimate (see caveats)"
TABLE_DEPENDS_ON = "A2 ~3 calls/decision = v9 §10.1 (二) max_num_rounds=1 (decided); invalid if refinement were restored"


def pct(a, q):
    return float(np.percentile(np.asarray(a, dtype=float), q)) if len(a) else float("nan")


def dist(a):
    a = list(a)
    return {"n": len(a), "min": float(min(a)) if a else None, "median": pct(a, 50), "mean": float(np.mean(a)) if a else None,
            "p90": pct(a, 90), "max": float(max(a)) if a else None}


def trial_inputs(trial_dir, ntok):
    calls = [json.loads(x) for x in (trial_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    ok = [c for c in calls if c["ok_call"]]
    t = np.array([c["call_time_s"] for c in ok])
    comp = np.array([c["api_completion_tokens"] for c in ok])
    prompt = np.array([c["api_prompt_tokens"] for c in ok])
    cost_api, cost_model = [], []
    for c in ok:
        raw = json.loads((trial_dir / f"raw_{c['decision_date']}_r{c['rep']}.json").read_text(encoding="utf-8"))
        cost_api.append(raw["usage"]["cost"])
        cost_model.append(c["api_prompt_tokens"] * PRICE_IN + c["api_completion_tokens"] * PRICE_OUT)
    fa_tokens, reasoning_tokens = [], []
    for c in ok:
        obj = json.loads(c["response"])
        fa = obj["final_answer"]
        fa_tokens.append(ntok(fa if isinstance(fa, str) else json.dumps(fa, ensure_ascii=False)))
        reasoning_tokens.append(ntok(str(obj.get("reasoning", ""))))
    return {
        "calls": ok,
        "latency_s": {**dist(t), "limit": "n=5; p90 is an interpolation, not a tail estimate"},
        "completion_tokens": dist(comp),
        "prompt_tokens": dist(prompt),
        "s_per_completion_token": {**dist(t / comp), "limit": "n=5; p90 is an interpolation, not a tail estimate"},
        "cost_api_vs_price_model": [(round(a, 7), round(b, 7)) for a, b in zip(cost_api, cost_model)],
        "final_answer_tokens": dist(fa_tokens),
        "reasoning_field_tokens": dist(reasoning_tokens),
    }


def reasoning_mode_content_ratio(trial_calls):
    """Same 04-27 prompt: non-reasoning completion vs reasoning-mode visible content."""
    path = RESULTS / "format_trial" / REASONING_TRIAL / "calls.jsonl"
    rm = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    visible = [c["api_completion_tokens"] - c["api_reasoning_tokens"] for c in rm
               if c.get("api_completion_tokens") and c.get("api_reasoning_tokens") is not None]
    nr = [c["api_completion_tokens"] for c in trial_calls if c["decision_date"] == "2026-04-27"]
    return {"reasoning_mode_visible": visible, "non_reasoning": nr,
            "ratio": float(np.mean(nr) / np.mean(visible))}


def a0_prompts(dates, index, cfg, names, ntok, playbook):
    rows = []
    for T in dates:
        inp = build_decision_input(index, T, cfg)
        context, _, _ = render_context(inp, names)
        prompt = GENERATOR_PROMPT.format(playbook, "(empty)", QUESTION, context)
        rows.append({"date": T, "prompt": ntok(prompt) + JSON_MODE_OVERHEAD, "context": ntok(context)})
    return pd.DataFrame(rows)


def a1_window(trial_calls, names, ntok):
    """A window of the 5 trial decisions with their realized h=10 alpha, in two formats."""
    dates = pd.DatetimeIndex([pd.Timestamp(c["decision_date"]) for c in trial_calls])
    U = panel.universes(dates)
    pnl = alpha.build_alpha_panel(dates, panel.frame("adj_open"), U, horizons=(10,))
    compact, with_reasoning, gt_only = [], [], []
    for c in trial_calls:
        T = pd.Timestamp(c["decision_date"])
        a = analyse(c["response"], U[T])
        obj = json.loads(c["response"])
        g = pnl[pnl["decision_date"] == T].set_index("stock_id")
        head = f"決策日 {T.date()}（h=10 於 {pd.Timestamp(g['mature_date_h10'].iloc[0]).date()} 到期）"
        lines, gt = [head, "代號 簡稱 當時評分 實際α(10日)"], [head, "代號 實際α(10日)"]
        for sid in U[T]:
            av = g.loc[sid, "alpha_h10"]
            av_s = f"{av * 100:+.2f}%" if pd.notna(av) else "NA"
            lines.append(f"{sid} {names.get(sid, '')} {a['scores'].get(sid, float('nan')):+.2f} {av_s}")
            gt.append(f"{sid} {av_s}")
        block = "\n".join(lines)
        compact.append(block)
        with_reasoning.append(block + "\n當時的推理：" + str(obj.get("reasoning", "")))
        gt_only.append("\n".join(gt))
    header = "最近 5 個已到期的決策與其實際報酬：\n"
    return {
        "per_decision_compact": dist([ntok(b) for b in compact]),
        "per_decision_with_reasoning": dist([ntok(b) for b in with_reasoning]),
        "window5_compact": ntok(header + "\n\n".join(compact)),
        "window5_with_reasoning": ntok(header + "\n\n".join(with_reasoning)),
        "ground_truth_block": dist([ntok(b) for b in gt_only]),
        "header": ntok(header),
    }


def templates(ntok, playbook):
    return {
        "empty_playbook": ntok(playbook),
        "question": ntok(QUESTION),
        "generator_without_context": ntok(GENERATOR_PROMPT.format(playbook, "(empty)", QUESTION, "")),
        "reflector_template": ntok(REFLECTOR_PROMPT.format("", "", "", "", "", "")),
        "curator_template": ntok(CURATOR_PROMPT.format(
            current_step="", total_samples="", token_budget="", playbook_stats="",
            recent_reflection="", current_playbook="", question_context="")),
    }


def role_sizes(run_names, ntok):
    out = {}
    for role in ("generator", "reflector", "curator"):
        resp, prm = [], []
        for d in run_names:
            for f in glob.glob(str(RESULTS / d / "detailed_llm_logs" / f"{role}_*.json")):
                j = json.loads(Path(f).read_text(encoding="utf-8"))
                if j.get("response"):
                    resp.append(ntok(j["response"]))
                if j.get("prompt") and role != "generator":
                    prm.append(ntok(j["prompt"]))
        out[role] = {"response_tokens": dist(resp), "prompt_tokens": dist(prm) if prm else None}
    return out


def playbook_growth(ntok, empty_tokens):
    rows = []
    for d in DEEPSEEK_RUNS + [QWEN_RUN]:
        run = RESULTS / d
        n_cur = len(glob.glob(str(run / "detailed_llm_logs" / "curator_*.json")))
        files = sorted(glob.glob(str(run / "intermediate_playbooks" / "*step_*_playbook.txt")),
                       key=lambda p: int(re.search(r"step_(\d+)", p).group(1)))
        points = [(int(re.search(r"step_(\d+)", p).group(1)), Path(p).read_text(encoding="utf-8")) for p in files]
        points.append((n_cur, (run / "final_playbook.txt").read_text(encoding="utf-8")))
        for step, text in points:
            tok = ntok(text)
            rows.append({"run": d, "backbone": "qwen3-14b" if d == QWEN_RUN else "deepseek-v4-flash",
                         "curator_calls": step, "tokens": tok,
                         "bullets": len(re.findall(r"^\s*\[", text, flags=re.M)),
                         "tokens_per_curator_call": (tok - empty_tokens) / step if step else None})
    return pd.DataFrame(rows).drop_duplicates(["run", "curator_calls"], keep="last")


def call(prompt, completion, s_per_tok):
    return np.array([prompt, completion, completion * s_per_tok, prompt * PRICE_IN + completion * PRICE_OUT, 1.0])


def arm_runs(p, S, T):
    """One run of each arm over one condition. Returns {arm: [prompt, completion, seconds, cost, calls]}."""
    a0, ctx = p["prompt"].to_numpy(), p["context"].to_numpy()
    n = len(a0)
    spt, gen_c = S["s_per_tok"], S["gen_completion"]
    out = {k: np.zeros(5) for k in ("A0", "A1", "A2")}
    for j in range(n):
        out["A0"] += call(a0[j], gen_c, spt)

        k = min(WINDOW_K, max(0, j - DELAY + 1))      # A1: one call, window with reasoning in its context
        window = k * S["a1_per_decision"] + (T["a1_header"] if k else 0)
        out["A1"] += call(a0[j] + window, gen_c, spt)

        m = max(0, j - DELAY + 1)          # matured decisions already folded into the playbook at j
        if j >= DELAY:
            pb = S["growth"] * (m - 1)      # playbook before this maturity is processed
            refl_prompt = (T["reflector_template"] + T["question"] + gen_c + S["final_answer"]
                           + T["ground_truth"] + 30 + S["bullets_used_share"] * pb)
            out["A2"] += call(refl_prompt, S["reflector_completion"], spt)
            cur_prompt = (T["curator_template"] + 80 + S["reflector_completion"]
                          + T["empty_playbook"] + pb + ctx[j - DELAY])
            out["A2"] += call(cur_prompt, S["curator_completion"], spt)
        out["A2"] += call(a0[j] + S["growth"] * m, gen_c, spt)
    return {k: v * S["retry_factor"] for k, v in out.items()}, {
        "a2_final_playbook_tokens": T["empty_playbook"] + S["growth"] * max(0, n - DELAY),
        "a2_playbook_at_thirds": [T["empty_playbook"] + S["growth"] * max(0, int(n * q) - DELAY + 1) for q in (1 / 3, 2 / 3, 1.0)],
    }


def fmt_row(label, v, seeds=1):
    p, c, s, cost, n = v * seeds
    return {"arm": label, "calls": int(round(n)), "prompt_M_tokens": round(p / 1e6, 2),
            "completion_M_tokens": round(c / 1e6, 2), "hours": round(s / 3600, 1), "cost_usd_reference": round(cost, 2)}


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--trial", default=str(RESULTS / "format_trial" / "trial_20260914_110451"))
    args = ap_.parse_args()
    market.login()                  # company_names() reads finlab before any panel call would log in
    ntok = load_tokenizer()
    playbook = ACE._initialize_empty_playbook(None)
    names = company_names()

    trial = trial_inputs(Path(args.trial), ntok)
    ratio = reasoning_mode_content_ratio(trial["calls"])
    tmpl = templates(ntok, playbook)
    win = a1_window(trial["calls"], names, ntok)
    roles_ds = role_sizes(DEEPSEEK_RUNS, ntok)
    growth = playbook_growth(ntok, tmpl["empty_playbook"])

    if A1_WINDOW_FORMAT != "with_reasoning" or A1_CALLS_PER_DECISION != 1:
        raise RuntimeError("arm_runs models A1 as one call with the reasoning window; arm_protocol now says otherwise")
    cfg = news.load_config()
    index, _ = news.build(cfg)
    windows = {k: panel.decision_dates(*v) for k, v in CONDITIONS.items()}
    prompts = {k: a0_prompts(d, index, cfg, names, ntok, playbook) for k, d in windows.items()}

    ds_growth = growth[(growth["backbone"] == "deepseek-v4-flash") & growth["tokens_per_curator_call"].notna()]
    g_all = growth[growth["tokens_per_curator_call"].notna()]["tokens_per_curator_call"]
    rc = roles_ds["reflector"]["response_tokens"]
    cc = roles_ds["curator"]["response_tokens"]
    gen = trial["completion_tokens"]
    spt = trial["s_per_completion_token"]
    a1_tok = win["per_decision_with_reasoning"]["median"]      # the decided format, in every scenario

    T = {**tmpl, "a1_header": win["header"], "ground_truth": win["ground_truth_block"]["median"]}
    scenarios = {
        "low": {"gen_completion": gen["min"], "s_per_tok": spt["mean"], "a1_per_decision": a1_tok,
                "growth": float(g_all.min()), "reflector_completion": rc["median"], "curator_completion": cc["median"],
                "bullets_used_share": 0.0, "final_answer": trial["final_answer_tokens"]["median"], "retry_factor": 1.00},
        "mid": {"gen_completion": gen["mean"], "s_per_tok": spt["mean"], "a1_per_decision": a1_tok,
                "growth": float(ds_growth["tokens_per_curator_call"].median()),
                "reflector_completion": rc["median"] * ratio["ratio"], "curator_completion": cc["median"] * ratio["ratio"],
                "bullets_used_share": 0.25, "final_answer": trial["final_answer_tokens"]["median"], "retry_factor": 1.02},
        "high": {"gen_completion": gen["max"], "s_per_tok": spt["mean"], "a1_per_decision": a1_tok,
                 "growth": float(g_all.max()),
                 "reflector_completion": rc["p90"] * ratio["ratio"], "curator_completion": cc["p90"] * ratio["ratio"],
                 "bullets_used_share": 0.5, "final_answer": trial["final_answer_tokens"]["median"], "retry_factor": 1.10},
    }

    results, playbooks = {}, {}
    for sc, S in scenarios.items():
        for cond, p in prompts.items():
            r, pbk = arm_runs(p, S, T)
            results[(sc, cond)] = r
            playbooks[(sc, cond)] = pbk

    table = []
    for sc in scenarios:
        limit = HIGH_LIMIT if sc == "high" else ""
        for cond in prompts:
            r = results[(sc, cond)]
            for arm in ("A0", "A1", "A2"):
                table.append({"scenario": sc, "condition": cond, **fmt_row(arm, r[arm], SEEDS),
                              "limit": limit, "depends_on": TABLE_DEPENDS_ON})
            table.append({"scenario": sc, "condition": cond, **fmt_row("total(A0+A1+A2)", r["A0"] + r["A1"] + r["A2"], SEEDS),
                          "limit": limit, "depends_on": TABLE_DEPENDS_ON})
        both = sum(results[(sc, c)]["A0"] + results[(sc, c)]["A1"] + results[(sc, c)]["A2"] for c in prompts)
        table.append({"scenario": sc, "condition": "post+pre", **fmt_row("total(A0+A1+A2)", both, SEEDS),
                      "limit": limit, "depends_on": TABLE_DEPENDS_ON})
    tbl = pd.DataFrame(table)

    # sensitivity: move one assumption from mid to high, total over both conditions
    def total_hours_cost(S):
        h = c = 0.0
        for p in prompts.values():
            r, _ = arm_runs(p, S, T)
            v = (r["A0"] + r["A1"] + r["A2"]) * SEEDS
            h += v[2] / 3600
            c += v[3]
        return h, c
    base_h, base_c = total_hours_cost(scenarios["mid"])
    sens = []
    for key in ("gen_completion", "growth", "reflector_completion", "curator_completion",
                "bullets_used_share", "retry_factor"):
        S = dict(scenarios["mid"])
        S[key] = scenarios["high"][key]
        h, c = total_hours_cost(S)
        sens.append({"assumption_to_high": key, "mid": scenarios["mid"][key], "high": scenarios["high"][key],
                     "hours_delta": round(h - base_h, 1), "cost_delta": round(c - base_c, 2),
                     "limit": HIGH_LIMIT})

    tail = {"per_call_latency_s": trial["latency_s"],
            "hours_if_every_call_at_p90_s_per_token": {
                sc: round(sum((results[(sc, c)]["A0"] + results[(sc, c)]["A1"] + results[(sc, c)]["A2"])[1]
                              for c in prompts) * SEEDS * trial["s_per_completion_token"]["p90"] / 3600, 1)
                for sc in scenarios},
            "limit": "trial n=5; p90 of 5 points is an interpolation and NOT a tail estimate; "
                     "these hours are a what-if, not a tail bound"}

    v8_reference = {"decision_points": {"post": 82, "pre": 245}, "seconds_per_call": 45,
                    "calls_per_decision": {"A0": 1, "A1": 2, "A2": "5-14 (~10) + initial test"},
                    "news_tokens_per_window": 5600, "hours": {"post": 45, "pre": 134, "total": 180}}
    v8_recomputed = {}
    for cond, n in (("post", 82), ("pre", 245)):
        v8_recomputed[cond] = round((n * (1 + 2 + 10) + n) * SEEDS * 45 / 3600, 1)

    out = {
        "CAVEATS": CAVEATS,
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "git": records.git_state(),
        "conditions": CONDITIONS,
        "data_last_dates": panel.data_last_dates(),
        "a1_protocol": {"window_format": A1_WINDOW_FORMAT, "calls_per_decision": A1_CALLS_PER_DECISION,
                        "window_k": WINDOW_K, "per_decision_tokens_used": a1_tok,
                        "window5_tokens_measured": win["window5_with_reasoning"]},
        "decision_dates": {k: {"count": int(len(d)), "first": str(d[0].date()), "last": str(d[-1].date())} for k, d in windows.items()},
        "a0_prompt_tokens": {k: dist(p["prompt"]) for k, p in prompts.items()},
        "context_tokens": {k: dist(p["context"]) for k, p in prompts.items()},
        "trial": {k: v for k, v in trial.items() if k != "calls"},
        "reasoning_mode_content_ratio": ratio,
        "templates": tmpl,
        "a1_window": win,
        "ace_role_sizes_deepseek_runs": roles_ds,
        "playbook_growth": growth.to_dict(orient="records"),
        "scenarios": {**scenarios, "high_limit": HIGH_LIMIT},
        "a2_playbook_tokens": {f"{sc}/{cond}": v for (sc, cond), v in playbooks.items()},
        "sensitivity_total_both_conditions": sens,
        "baseline_mid_total": {"hours": round(base_h, 1), "cost_usd_reference": round(base_c, 2),
                               "depends_on": TABLE_DEPENDS_ON},
        "tail": tail,
        "v8_reference": v8_reference,
        "v8_formula_recomputed_hours": v8_recomputed,
        "table_depends_on": TABLE_DEPENDS_ON,
        "table": table,
    }
    dest = RESULTS / "budget" / f"budget_{_dt.datetime.now():%Y%m%d_%H%M%S}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    records.to_json(out, dest)

    banner = "\n".join(f"!! {k}: {v}" for k, v in CAVEATS.items())
    print("== CAVEATS\n" + banner)
    pd.set_option("display.width", 250)
    for k in ("conditions", "data_last_dates", "a1_protocol", "decision_dates", "a0_prompt_tokens", "context_tokens", "trial", "reasoning_mode_content_ratio",
              "templates", "a1_window", "ace_role_sizes_deepseek_runs", "scenarios", "a2_playbook_tokens",
              "baseline_mid_total", "tail", "v8_formula_recomputed_hours"):
        print(f"\n== {k}\n{json.dumps(out[k], ensure_ascii=False, default=str, indent=1)}")
    print("\n== playbook growth\n", growth.to_string(index=False))
    print("\n== sensitivity (mid -> high, one assumption at a time; high carries: " + HIGH_LIMIT + ")\n",
          pd.DataFrame(sens).drop(columns=["limit"]).to_string(index=False))
    print("\n== budget table (x3 seeds) - depends on: " + TABLE_DEPENDS_ON + "\n",
          tbl.drop(columns=["depends_on"]).to_string(index=False))
    print("\n== CAVEATS (repeated)\n" + banner)
    print("\nwritten to", dest)


if __name__ == "__main__":
    main()
