"""Per-decision output handling shared by every arm (protocol fixed 2026-09-14).

Generation
1. final_answer may be a JSON object or a string holding one; both are accepted and
   the form is recorded as final_answer_form. This is parser tolerance, not an
   experiment parameter - but a form-jitter rate that differs by arm is a finding.
2. A decision point gets at most MAX_RETRIES = 2 retries after its first attempt, the
   same rule for every arm, and every attempt is recorded. An attempt is spent when
   it is not usable: the call raised, the output is not JSON, final_answer is in
   neither accepted form, a score key is duplicated or outside the universe, or not
   every universe member gets a number in [-1, 1].
3. When the retries are used up the decision point is void as a whole: no score is
   filled in and no single stock is dropped, so every scored date carries the full
   cross-section and csIC is always computed on the same base.
4. A duplicated score key or a key outside the universe voids the point even when all
   50 members are scored: the model did not answer on the given universe, the case
   is rare, and one standard is applied to every answer. It is its own void kind
   (duplicate_key, out_of_universe_key), kept apart from incomplete.

Learning calls (A2 Reflector and Curator)
5. Same retry rule as generation: at most 2 retries, every attempt recorded. The
   consequence differs - a failed generation leaves a date unscored, a failed
   reflection leaves the playbook one update short - so they are counted separately.
   A Reflector that still fails after its retries means no Curator call for that
   maturity (there is no reflection to curate).
6. A decision that was void has nothing to learn from when it matures: it is skipped,
   nothing replaces it, and the skip is counted. The same holds for A1's window: a void
   decision leaves its slot empty instead of pulling in an older one.

This must be the only retry layer. Run with ACE_MAX_RETRIES=1 and an OpenAI client
with max_retries=0; otherwise retries happen below the attempts counted here.

v9 §10.1, all five items settled (2026-09-14) - the M5a specification is complete
(二) Refinement off: MAX_NUM_ROUNDS = 1. At maturity the Reflector reflects once and the
    decision is never regenerated. After decoupling, refinement would mean regenerating a
    decision 11 trading days later with feedback that already contains realized returns;
    it would almost always come out "right", and the playbook would store after-the-fact
    rationalisation rather than transferable judgement. The budget's ~3 calls per decision
    date rests on this.
(一) Feedback is continuous, never a binary verdict (FEEDBACK_CONTENT): every stock's score
    at the time with its realized alpha, the date's csIC, and that csIC's quantile in the
    random-score distribution. No threshold, no three-way classes, no answer_is_correct-style
    label. With refinement off, answer_is_correct controls nothing (no refinement to trigger,
    no early stop), so the binary label is retired and (一) only decides what the Reflector reads.
(三) IC is computed on the single generation of each decision date; nothing is regenerated,
    so there is no choice of which generation to freeze.
(四) No initial test. A0 is its own arm, called interleaved by date, with the prompt
    "Generator prompt + empty playbook".
(五) v9 §4.2.1 (a)/(b) only arises on an offline warm-up path, which is not used.

A1 rolling window, decided by the user 2026-09-15 (settled; not to be reopened)
    Format A1_WINDOW_FORMAT = "with_reasoning": each matured decision in the window carries every
    member's score, its realized h=10 alpha and the reasoning the generation gave at the time -
    about 2,455 tokens per decision (mean 2,490) and 12,466 for a full window of 5
    (results/budget/budget_20260914_120638.json, v9.3 §5.2.1).
    Reason for the format: information parity with A2. A1 is the baseline A2 has to beat; giving it
    outcomes without the reasoning would weaken it before beating it and make an A2 win
    unconvincing. A2's playbook is itself a product of reasoning, so the two memories must sit at
    the same information level. The cost pushes the total toward the high scenario.

    Two calls per decision date, A1_CALLS_PER_DECISION = 2 (the user's second ruling of 2026-09-15;
    it replaces a one-call reading that had been taken without asking):
      1. reflection  roles.A1_REFLECTION_PROMPT over the window (the non-void decisions among the last
                     5 maturity slots, with reasoning) -> JSON {"reflection": text}
      2. generation  the Generator with that text in its reflection slot; the context is the date's
                     own input only, the raw window is not passed again
    Reason: A1 stands for CryptoTrade §2.4's Reflection Agent, a separate reflection call that reviews
    the matured decisions and their returns and writes a reflection the decision then uses. Folded
    into one Generator call, A1 would no longer be rolling-window reflection, which RQ2's wording and
    the §4.3 literature mapping name. And A2 spends two extra calls (Reflector, Curator) digesting its
    experience; making A1 digest the raw window inside its single decision call would weaken the
    baseline - the same reason as for the format.
    Before the first maturity (or when every slot is void) the window is empty: no reflection call,
    the Generator gets "(empty)", one call that date.
    The reflection call follows rule 5's retries. If it still fails (A1_REFLECTION_FAILURE_RULE) the
    date is still generated, with "(empty)" in the reflection slot, and the failure is counted apart
    (a1_reflection_status = failed) - the analogue of A2 generating on its last playbook when its
    Reflector fails. Recorded as this code's rule; the user has not ruled on it yet.
    Length: A1_REFLECTION_CAP_TOKENS = 1,300, stated in the prompt as about 700 English words or about
    1,500 Chinese characters (1.86 tokens per English word, measured on the five non-reasoning trial
    reasonings; 0.84 per Chinese character, measured on the 2026-05-19 headlines; local
    DeepSeek-V4-Flash tokenizer). Basis, set before any A1 call: parity with A2's Reflector output,
    whose mid budget estimate is 1,268 tokens (FiNER DeepSeek Reflector median 416 x the 3.05
    non-reasoning ratio), rounded up. It is an instruction, not max_tokens: a small max_tokens made
    every ClinePass call a 500 (2026-09-03) and a truncated JSON answer is unusable. Each reflection's
    completion tokens are recorded and an overrun is flagged (a1_reflection_over_cap), never cut.

A0 pilot, decided by the user 2026-09-15
    A0 run alone does not meet §5.2.0 (arms interleaved on the same date so the host mix cannot line
    up with the arm), so it can never be the thesis's A0. The first A0 run is a gate pilot only
    (PILOT_A0): one seed, the post-cutoff condition (85 dates), judged by the criteria file committed
    before it runs. The real A0 is rerun interleaved with A1 and A2; no pilot number enters a result
    table. Its manifest says run_kind = "pilot" and usable_as_result = false, and
    records.require_result_run refuses it.

Cost (v9.3 §6.4): every attempt keeps the usage.cost the gateway metered with its answer;
meta_row sums it per date (cost_usd for generation, aux_cost_usd for Reflector + Curator) and
records.write_run unpacks every attempt into llm_calls.parquet.
"""

import json
import re

import pandas as pd

MAX_RETRIES = 2
MAX_NUM_ROUNDS = 1      # §10.1 (二): single reflection at maturity, no regeneration
FEEDBACK_CONTENT = (    # §10.1 (一): what the Reflector reads; continuous, no verdict label
    "per stock: the score given on the decision date and its realized h=10 alpha",
    "the decision date's csIC",
    "that csIC's quantile in the pooled daily csIC of random scores (50 seeds x 399 dates, h=10 sd 0.144)",
)
DELAY = 11             # h=10 feedback of decision i is usable at decision date i+11 (v9 §5.1)
WINDOW_K = 5            # A1: the last 5 matured decisions
A1_WINDOW_FORMAT = "with_reasoning"     # decided 2026-09-15, see the docstring
A1_CALLS_PER_DECISION = 2               # reflection call, then generation (second ruling, 2026-09-15)
A1_WINDOW_FORMAT_REASON = (
    "information parity with A2: A1 is the baseline A2 must beat, and outcomes without the reasoning "
    "would weaken it before beating it; A2's playbook is itself a product of reasoning, so both "
    "memories sit at the same information level")
A1_CALLS_REASON = (
    "A1 stands for CryptoTrade §2.4's Reflection Agent, a separate reflection call; folded into the "
    "Generator it would stop being rolling-window reflection (RQ2, §4.3). A2 digests experience in two "
    "extra calls, so a single call for A1 would weaken the baseline")
A1_REFLECTION_ROLE = "a1_reflector"
A1_REFLECTION_CAP_TOKENS = 1300
A1_REFLECTION_CAP_WORDS = 700           # 1,300 / 1.86 tokens per English word
A1_REFLECTION_CAP_ZH_CHARS = 1500       # 1,300 / 0.84 tokens per Chinese character, rounded down
A1_REFLECTION_CAP_BASIS = (
    "parity with A2's Reflector output: budget mid estimate 1,268 tokens (FiNER DeepSeek Reflector median "
    "416 x 3.05 non-reasoning ratio), rounded up; words and characters from measured 1.86 tokens per "
    "English word and 0.84 per Chinese character; an instruction, not max_tokens; overruns flagged, not cut")
A1_REFLECTION_FAILURE_RULE = "generate_with_empty_reflection"   # the code's rule; not yet ruled on by the user
RUN_KINDS = ("offline_check", "pilot", "main")
PILOT_A0 = {"run_kind": "pilot", "arm": "A0", "condition": "post", "seeds": 1, "usable_as_result": False,
            "purpose": "gate: do headlines carry enough signal to make the A1/A2 comparison meaningful",
            "why_not_a_result": "A0 alone is not interleaved with A1/A2 on the same dates (v9.3 §5.2.0); the "
                                "real A0 is rerun interleaved"}
ACCEPTED_FORMS = ("object", "json_string")
# order in which a failure is named when several apply; every applicable kind is kept too
FAILURE_ORDER = ("call_error", "json", "final_answer_form", "duplicate_key", "out_of_universe_key", "incomplete")

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
    """Parse one generation. duplicate_keys holds duplicates that make the scores ambiguous:
    a repeated key inside final_answer (also after stripping whitespace) or a repeated
    final_answer; other repeated keys (e.g. two "reasoning") go to other_duplicate_keys."""
    out = {"json_ok": False, "top_keys": None, "final_answer_form": None, "n_returned": 0,
           "n_valid_ids": 0, "missing": [], "extra_keys": [], "duplicate_keys": [],
           "other_duplicate_keys": [], "value_forms": {}, "out_of_range_examples": [], "scores": {}}
    dup_pairs = []          # (id of the object holding the repeat, key)

    def hook(pairs):
        d = {}
        for k, v in pairs:
            if k in d:
                dup_pairs.append((id(d), k))
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

    scores_id = id(scores) if scores is not None else None
    for holder, k in dup_pairs:
        if holder == scores_id or (holder == id(obj) and k == "final_answer"):
            out["duplicate_keys"].append(k)
        else:
            out["other_duplicate_keys"].append(k)
    if scores is None:
        return out

    members = set(universe)
    keys = {}
    for k, v in scores.items():
        ks = str(k).strip()
        if ks in keys:
            out["duplicate_keys"].append(ks)
        keys[ks] = v
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


def failure_kinds(analysis, universe):
    """Every reason the generation is unusable, in FAILURE_ORDER; empty when usable."""
    if not analysis["json_ok"]:
        return ["json"]
    if analysis["final_answer_form"] not in ACCEPTED_FORMS:
        return ["final_answer_form"]
    kinds = []
    if analysis["duplicate_keys"]:
        kinds.append("duplicate_key")
    if analysis["extra_keys"]:
        kinds.append("out_of_universe_key")
    if analysis["missing"] or len(analysis["scores"]) != len(universe):
        kinds.append("incomplete")
    return kinds


def run_decision(call, universe, max_retries=MAX_RETRIES):
    """Attempt one decision point until usable or out of retries.

    call(attempt_no) returns a dict with the raw text under "response" plus any
    per-attempt facts to keep (tokens, latency, provider fields); it may raise.
    """
    attempts = []
    for attempt_no in range(1, max_retries + 2):
        rec = {"attempt": attempt_no}
        try:
            out = call(attempt_no)
            rec.update(out)
            rec["ok_call"] = True
            analysis = analyse(out.get("response") or "", universe)
            kinds = failure_kinds(analysis, universe)
        except Exception as e:
            rec["ok_call"] = False
            rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            analysis = analyse("", universe)
            kinds = ["call_error"]
        rec.update({
            "failure": kinds[0] if kinds else None,
            "failure_kinds": kinds,
            "json_ok": analysis["json_ok"],
            "json_error": analysis.get("json_error"),
            "final_answer_form": analysis["final_answer_form"],
            "n_valid_ids": analysis["n_valid_ids"],
            "n_scored_in_range": len(analysis["scores"]),
            "missing": analysis["missing"],
            "extra_keys": analysis["extra_keys"],
            "duplicate_keys": analysis["duplicate_keys"],
            "other_duplicate_keys": analysis["other_duplicate_keys"],
            "value_forms": analysis["value_forms"],
        })
        attempts.append(rec)
        if not kinds:
            return {"voided": False, "void_reason": None, "void_kinds": [], "scores": analysis["scores"],
                    "final_answer_form": analysis["final_answer_form"],
                    "n_attempts": attempt_no, "n_retries": attempt_no - 1, "attempts": attempts}
    return {"voided": True, "void_reason": attempts[-1]["failure"], "void_kinds": attempts[-1]["failure_kinds"],
            "scores": None, "final_answer_form": None, "n_attempts": len(attempts),
            "n_retries": len(attempts) - 1, "attempts": attempts}


def json_object_check(response):
    """Default usability check for a Reflector/Curator response in JSON mode."""
    if response is None or not str(response).strip():
        return "empty"
    try:
        obj = json.loads(response)
    except Exception:
        return "json"
    return None if isinstance(obj, dict) else "json"


def run_aux_call(call, role, check=json_object_check, max_retries=MAX_RETRIES):
    """One Reflector or Curator call under the generation retry rule, recorded on its own.

    call(attempt_no) returns a dict with "response"; check(response) returns None when
    usable, else the failure kind. A call that raises is a call_error.
    """
    attempts = []
    for attempt_no in range(1, max_retries + 2):
        rec = {"role": role, "attempt": attempt_no}
        try:
            out = call(attempt_no)
            rec.update(out)
            rec["ok_call"] = True
            kind = check(out.get("response"))
        except Exception as e:
            rec["ok_call"] = False
            rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            kind = "call_error"
        rec["failure"] = kind
        attempts.append(rec)
        if kind is None:
            return {"role": role, "status": "ok", "failure": None, "response": out.get("response"),
                    "n_attempts": attempt_no, "n_retries": attempt_no - 1, "attempts": attempts}
    return {"role": role, "status": "failed", "failure": attempts[-1]["failure"], "response": None,
            "n_attempts": len(attempts), "n_retries": len(attempts) - 1, "attempts": attempts}


def matured_index(j, delay=DELAY):
    """Index of the decision whose h=10 feedback becomes usable at decision date j, or None."""
    return j - delay if j >= delay else None


def window_indices(j, voided, k=WINDOW_K, delay=DELAY):
    """A1 window at date j: the last k maturity slots. Void decisions are dropped, not replaced.

    voided: sequence of booleans by decision index. Returns (kept, skipped_void).
    """
    last = j - delay
    if last < 0:
        return [], []
    slots = range(max(0, last - k + 1), last + 1)
    return [i for i in slots if not voided[i]], [i for i in slots if voided[i]]


def process_maturity(matured_date, matured_voided, reflect_call, curate_call,
                     reflector_check=json_object_check, curator_check=json_object_check):
    """A2 learning step for the decision maturing today.

    reflect_call(attempt_no) and curate_call(attempt_no, reflection) return dicts with
    "response". A void decision is skipped with no call; a Reflector that fails after its
    retries means the Curator is not run.
    """
    if matured_date is None:
        return {"maturity_status": "none", "matured_decision_date": None, "reflector": None, "curator": None}
    if matured_voided:
        return {"maturity_status": "skipped_void", "matured_decision_date": pd.Timestamp(matured_date),
                "reflector": None, "curator": None}
    r = run_aux_call(reflect_call, "reflector", reflector_check)
    c = None
    if r["status"] == "ok":
        c = run_aux_call(lambda n: curate_call(n, r["response"]), "curator", curator_check)
    return {"maturity_status": "processed", "matured_decision_date": pd.Timestamp(matured_date),
            "reflector": r, "curator": c}


def a1_reflection_check(response):
    """Usable A1 reflection: a JSON object whose "reflection" is a non-empty string."""
    kind = json_object_check(response)
    if kind is not None:
        return kind
    text = json.loads(response).get("reflection")
    return None if isinstance(text, str) and text.strip() else "a1_reflection_missing"


def process_a1_reflection(slots, reflect_call, check=a1_reflection_check, cap_tokens=A1_REFLECTION_CAP_TOKENS):
    """A1 step 1 on one decision date. No usable slot: no call. Otherwise one reflection under rule 5;
    if it fails, the Generator's reflection slot stays "(empty)" (A1_REFLECTION_FAILURE_RULE)."""
    if not slots:
        return {"status": "not_run", "reflection": None, "attempts": [], "n_attempts": 0, "n_retries": 0,
                "completion_tokens": None, "over_cap": None}
    r = run_aux_call(reflect_call, A1_REFLECTION_ROLE, check)
    ok = r["status"] == "ok"
    tokens = r["attempts"][-1].get("completion_tokens") if ok else None
    return {"status": r["status"], "reflection": json.loads(r["response"])["reflection"] if ok else None,
            "attempts": r["attempts"], "n_attempts": r["n_attempts"], "n_retries": r["n_retries"],
            "completion_tokens": tokens, "over_cap": None if tokens is None else bool(tokens > cap_tokens)}


def score_rows(decision_date, outcome):
    """Score rows for records.write_run: the whole universe, or nothing for a void date."""
    if outcome["voided"]:
        return pd.DataFrame(columns=["decision_date", "stock_id", "score"])
    return pd.DataFrame({"decision_date": pd.Timestamp(decision_date),
                         "stock_id": list(outcome["scores"]),
                         "score": list(outcome["scores"].values())})


def attempt_cost(attempt):
    """usage.cost the gateway metered for one attempt; None when the call got no answer."""
    return ((attempt.get("provider") or {}).get("usage") or {}).get("cost")


def total_cost(attempts):
    vals = [c for c in (attempt_cost(a) for a in attempts) if c is not None]
    return float(sum(vals)) if vals else None


def meta_row(decision_date, outcome, maturity=None, window=None, a1_reflection=None):
    """One llm_meta row for records.write_run.

    maturity: process_maturity() result (A2). window: (kept_dates, skipped_void_dates) (A1).
    a1_reflection: process_a1_reflection() result (A1).
    """
    atts = outcome["attempts"]

    def total(key):
        vals = [a.get(key) for a in atts if a.get(key) is not None]
        return sum(vals) if vals else None

    row = {
        "decision_date": pd.Timestamp(decision_date),
        "final_answer_form": outcome["final_answer_form"],
        "n_attempts": outcome["n_attempts"],
        "n_retries": outcome["n_retries"],
        "voided": outcome["voided"],
        "void_reason": outcome["void_reason"],
        "void_kinds": ",".join(outcome["void_kinds"]) if outcome["voided"] else None,
        "attempts_json": json.dumps(atts, ensure_ascii=False, default=str),
        "llm_raw_output": atts[-1].get("response"),
        "prompt_tokens": total("prompt_tokens"),
        "completion_tokens": total("completion_tokens"),
        "n_llm_calls": len(atts),
        "latency_s": total("latency_s"),
        "cost_usd": total_cost(atts),
    }
    if maturity is not None:
        r, c = maturity["reflector"], maturity["curator"]
        md = maturity["matured_decision_date"]
        aux = (r["attempts"] if r else []) + (c["attempts"] if c else [])
        row.update({
            "maturity_status": maturity["maturity_status"],
            "matured_decision_date": None if md is None else str(pd.Timestamp(md).date()),
            "reflector_status": r["status"] if r else "not_run",
            "reflector_attempts": r["n_attempts"] if r else 0,
            "reflector_retries": r["n_retries"] if r else 0,
            "curator_status": c["status"] if c else "not_run",
            "curator_attempts": c["n_attempts"] if c else 0,
            "curator_retries": c["n_retries"] if c else 0,
            "aux_cost_usd": total_cost(aux),
            "aux_attempts_json": json.dumps(aux, ensure_ascii=False, default=str),
        })
    if window is not None:
        kept, skipped = window
        row.update({
            "window_decision_dates": json.dumps([str(pd.Timestamp(d).date()) for d in kept]),
            "window_skipped_void": len(skipped),
        })
    if a1_reflection is not None:
        row.update({
            "a1_reflection_status": a1_reflection["status"],
            "a1_reflection_attempts": a1_reflection["n_attempts"],
            "a1_reflection_retries": a1_reflection["n_retries"],
            "a1_reflection_tokens": a1_reflection["completion_tokens"],
            "a1_reflection_over_cap": a1_reflection["over_cap"],
            "aux_cost_usd": total_cost(a1_reflection["attempts"]),
            "aux_attempts_json": json.dumps(a1_reflection["attempts"], ensure_ascii=False, default=str),
        })
    return row


def void_summary(meta):
    """Per arm: void count and rate by kind, retries spent, final_answer forms over every attempt."""
    rows = []
    for arm, g in meta.groupby("arm", sort=True):
        forms, kinds_any = {}, {}
        for js in g["attempts_json"].dropna():
            for a in json.loads(js):
                f = a.get("final_answer_form") or "none"
                forms[f] = forms.get(f, 0) + 1
        voided = g["voided"].fillna(False).astype(bool)
        if "void_kinds" in g:
            for ks in g.loc[voided, "void_kinds"].dropna():
                for k in ks.split(","):
                    kinds_any[k] = kinds_any.get(k, 0) + 1
        rows.append({
            "arm": arm,
            "decision_points": int(len(g)),
            "voided": int(voided.sum()),
            "void_rate": float(voided.mean()) if len(g) else float("nan"),
            "void_reasons": g.loc[voided, "void_reason"].value_counts().to_dict(),
            "void_kinds_any": kinds_any,
            "points_needing_retry": int((g["n_retries"].fillna(0) > 0).sum()),
            "retries_spent": int(g["n_retries"].fillna(0).sum()),
            "attempt_forms": forms,
        })
    return pd.DataFrame(rows)


def learning_summary(meta):
    """Per arm, kept apart from generation voids: maturities skipped because the decision was
    void, Reflector/Curator outcomes and retries, playbook updates made, and A1 window slots
    left empty by void decisions."""
    rows = []
    for arm, g in meta.groupby("arm", sort=True):
        row = {"arm": arm}
        if "maturity_status" in g and g["maturity_status"].notna().any():
            ms = g["maturity_status"]
            row.update({
                "maturities_due": int(ms.isin(["processed", "skipped_void"]).sum()),
                "maturities_skipped_void": int((ms == "skipped_void").sum()),
                "reflector_ok": int((g["reflector_status"] == "ok").sum()),
                "reflector_failed": int((g["reflector_status"] == "failed").sum()),
                "reflector_retries": int(g["reflector_retries"].fillna(0).sum()),
                "curator_ok": int((g["curator_status"] == "ok").sum()),
                "curator_failed": int((g["curator_status"] == "failed").sum()),
                "curator_not_run_after_reflector_failure": int(((ms == "processed") & (g["reflector_status"] == "failed")).sum()),
                "curator_retries": int(g["curator_retries"].fillna(0).sum()),
                "playbook_updates": int((g["curator_status"] == "ok").sum()),
            })
        if "window_skipped_void" in g and g["window_skipped_void"].notna().any():
            w = g["window_skipped_void"].fillna(0)
            row.update({"window_slots_skipped_void": int(w.sum()), "dates_with_short_window": int((w > 0).sum())})
        if "a1_reflection_status" in g and g["a1_reflection_status"].notna().any():
            st = g["a1_reflection_status"]
            row.update({"a1_reflection_ok": int((st == "ok").sum()), "a1_reflection_failed": int((st == "failed").sum()),
                        "a1_reflection_not_run": int((st == "not_run").sum()),
                        "a1_reflection_retries": int(g["a1_reflection_retries"].fillna(0).sum()),
                        "a1_reflection_over_cap": int(g["a1_reflection_over_cap"].fillna(False).astype(bool).sum())})
        rows.append(row)
    return pd.DataFrame(rows)
