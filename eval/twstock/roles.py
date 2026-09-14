"""The real Generator / Reflector / Curator behind the replay (v9.2 §5.2.0, §4.4, §10.1). M5a, part 2.

Nothing here calls an LLM on import or construction; calls happen only when replay.run_replay
invokes the callables, through the client passed in. ace/ is not modified: the ACE role classes
and templates are used as they are.

Generator, every arm (GENERATOR_PROMPT through ace.core.generator.Generator)
  question    the task and output format, format_trial.QUESTION (identical for the three arms)
  context     the decision date's news and numbers, format_trial.render_context
  reflection  "(empty)" for every arm: refinement is off (MAX_NUM_ROUNDS = 1), nothing is regenerated
  playbook    A0 and A1: the ACE empty playbook. A2: its own playbook text.
  A1 alone also gets its rolling window, appended to the context slot in the format the user
  chooses from A1_WINDOW_FORMATS. The format is not decided (v9.2 §5.2.1), so A1_WINDOW_FORMAT is
  None and LiveRoles refuses to run without an explicit choice.

Reflector, A2 only, once per non-void maturity (REFLECTOR_PROMPT through Reflector.reflect)
  question = QUESTION; reasoning_trace = the decision's full generation; predicted_answer =
  utils.extract_answer of it; bullets_used = the playbook bullets that generation cited, read from
  the playbook as it stands at maturity; the §10.1 (一) feedback from feedback.render:
  ground_truth = item (1), environment_feedback = items (2) and (3). No verdict label.

Curator, only after a usable Reflector (CURATOR_PROMPT through Curator.curate)
  current_playbook = the playbook with the Reflector's bullet tags counted (the order ACE uses);
  recent_reflection = the reflection; question_context = the matured decision date's context;
  current_step = matured decision index + 1; total_samples = n - 11; token_budget = ACE's default.
  A response counts as usable only when Curator's own schema check accepts it (curator_check).
  AcePlaybook.commit re-applies the accepted operations with playbook_utils.apply_curator_operations
  and requires the result to equal what Curator returned, then BulletpointAnalyzer.analyze runs with
  merge=False (v9.2 §5.2: analyzer on, bulletpoint_merge off; dedup keeps the first bullet, no LLM).

Client: one retry layer only (arm_protocol). make_client() requires ACE_MAX_RETRIES=1 and builds the
ClinePass client with max_retries=0; ReasoningOffClient sends reasoning={"enabled": false} on every
request. utils._ClineUnwrapTransport writes every HTTP response's provider_metadata to
utils.provider_log_path(); each attempt record here also keeps the provider fields and usage that
came back with it.
"""

import hashlib
import json
import math
import os
from dataclasses import dataclass

os.environ.setdefault("ACE_MAX_RETRIES", "1")

import httpx
import pandas as pd

from ace.ace import ACE
from ace.core.curator import Curator
from ace.core.generator import Generator
from ace.core.reflector import Reflector
from playbook_utils import (apply_curator_operations, extract_playbook_bullets, get_playbook_stats,
                            parse_playbook_line, update_bullet_counts)
from utils import extract_answer, initialize_clients

from . import arm_protocol as ap
from . import feedback as fb
from . import records
from .decision_input import build_decision_input
from .format_trial import QUESTION, render_context
from .replay import DELAY, PlaybookEntry

MODEL = "cline-pass/deepseek-v4-flash"
API_PROVIDER = "clinepass"
MAX_TOKENS = 65536
READ_TIMEOUT_S = 1800
REASONING = {"enabled": False}
USE_JSON_MODE = True
PLAYBOOK_TOKEN_BUDGET = 80000        # ACE's default playbook_token_budget (ace/ace.py)
USE_BULLETPOINT_ANALYZER = True      # v9.2 §5.2
BULLETPOINT_THRESHOLD = 0.90         # ACE's default
BULLETPOINT_MERGE = False            # v9.2 §5.2: must stay off
EMPTY_PLAYBOOK = ACE._initialize_empty_playbook(None)

A1_WINDOW_FORMAT = None              # NOT decided; the user picks one of A1_WINDOW_FORMATS
A1_WINDOW_FORMATS = ("compact", "with_reasoning")
A1_WINDOW_HEADER = "最近 5 個已到期的決策與其實際報酬：\n"
A1_SLOT = "context"                  # where the window goes; part of the undecided format

PROVIDER_FIELDS = ("generation_id", "http_status", "echoed_model", "finish_reason", "resolvedProvider",
                   "finalProvider", "modelAttemptCount", "totalProviderAttemptCount", "usage")

_SCHEMA = Curator(None, API_PROVIDER, MODEL)     # used only for its response schema check


def sha256(text):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


class ReasoningOffClient:
    """Pass-through OpenAI client: reasoning off on every request, optional hard request cap."""

    def __init__(self, client, max_requests=None):
        if getattr(client, "max_retries", None) != 0:
            raise ValueError("the OpenAI client needs max_retries=0: arm_protocol is the only retry layer")
        self._client = client
        self.chat = self
        self.completions = self
        self.max_requests = max_requests
        self.requests = 0

    def create(self, **kwargs):
        if self.max_requests is not None and self.requests >= self.max_requests:
            raise RuntimeError(f"request budget of {self.max_requests} reached")
        self.requests += 1
        kwargs["extra_body"] = {**(kwargs.get("extra_body") or {}), "reasoning": REASONING}
        return self._client.chat.completions.create(**kwargs)


def make_client(max_requests=None):
    if os.getenv("ACE_MAX_RETRIES") != "1":
        raise RuntimeError("set ACE_MAX_RETRIES=1: arm_protocol is the only retry layer")
    base = initialize_clients(API_PROVIDER)[0].with_options(
        max_retries=0, timeout=httpx.Timeout(READ_TIMEOUT_S, connect=10.0))
    return ReasoningOffClient(base, max_requests)


class _NoLLM:
    """Client for BulletpointAnalyzer: with merge=False it must never call; if it does, fail loudly."""

    def __init__(self):
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        raise RuntimeError("BulletpointAnalyzer tried an LLM call; bulletpoint_merge must be off")


def make_analyzer():
    from ace.core.bulletpoint_analyzer import BulletpointAnalyzer
    return BulletpointAnalyzer(_NoLLM(), MODEL, MAX_TOKENS)


@dataclass(frozen=True)
class DateInput:
    context: str
    universe: tuple
    entry_date: pd.Timestamp


class Contexts:
    """Rendered Generator context per decision date, built once."""

    def __init__(self, index, cfg, names):
        self.index, self.cfg, self.names = index, cfg, names
        self._cache = {}

    def get(self, day):
        day = pd.Timestamp(day)
        if day not in self._cache:
            inp = build_decision_input(self.index, day, self.cfg)
            context, _, _ = render_context(inp, self.names)
            self._cache[day] = DateInput(context, tuple(u["stock_id"] for u in inp.universe), inp.entry_date)
        return self._cache[day]


# ---------------------------------------------------------------- A1 window (format not decided)

def a1_decision_block(slot, names, fmt):
    """One matured decision as the budget estimate rendered it (budget_estimate.a1_window)."""
    if fmt not in A1_WINDOW_FORMATS:
        raise ValueError(f"A1 window format must be one of {A1_WINDOW_FORMATS}, got {fmt!r}")
    head = f"決策日 {pd.Timestamp(slot.decision_date).date()}（h=10 於 {pd.Timestamp(slot.matured_on).date()} 到期）"
    lines = [head, "代號 簡稱 當時評分 實際α(10日)"]
    for sid, av in slot.outcome.items():
        av_s = f"{av * 100:+.2f}%" if av is not None and pd.notna(av) else "NA"
        lines.append(f"{sid} {names.get(sid, '')} {slot.scores.get(sid, float('nan')):+.2f} {av_s}")
    block = "\n".join(lines)
    if fmt == "with_reasoning":
        block += "\n當時的推理：" + str(slot.reasoning or "")
    return block


def a1_window_block(slots, names, fmt):
    """The whole window, oldest first; empty string when no decision has matured yet."""
    if fmt not in A1_WINDOW_FORMATS:
        raise ValueError(f"A1 window format must be one of {A1_WINDOW_FORMATS}, got {fmt!r}")
    if not slots:
        return ""
    return A1_WINDOW_HEADER + "\n\n".join(a1_decision_block(s, names, fmt) for s in slots)


# ---------------------------------------------------------------- A2 playbook

class A2Snapshot(tuple):
    """Provenance entries of the bullets present (iterable, as replay's checks expect) plus the text."""

    def __new__(cls, entries, text, next_global_id):
        obj = super().__new__(cls, entries)
        obj.text = text
        obj.next_global_id = next_global_id
        obj.sha256 = sha256(text)
        return obj


def bullets_with_sections(text):
    section, out = "general", []
    for line in text.strip().split("\n"):
        s = line.strip()
        if s.startswith("##"):
            section = s[2:].strip().lower().replace(" ", "_").replace("&", "and")
            continue
        p = parse_playbook_line(line)
        if p:
            out.append({**p, "section": section})
    return out


def bullet_tags_of(reflection):
    """Reflector._extract_bullet_tags in JSON mode."""
    try:
        tags = json.loads(reflection).get("bullet_tags", [])
    except (TypeError, ValueError, AttributeError):
        return []
    return tags if isinstance(tags, list) else []


def counted(text, reflection):
    tags = bullet_tags_of(reflection)
    return update_bullet_counts(text, tags) if tags else text


class AcePlaybook:
    """A2's ACE text playbook, with the maturity every bullet came from."""
    arm = "A2"

    def __init__(self, analyzer=None, threshold=BULLETPOINT_THRESHOLD):
        if BULLETPOINT_MERGE:
            raise ValueError("bulletpoint_merge must be off (v9.2 §5.2)")
        self.text = EMPTY_PLAYBOOK
        self.next_global_id = 1
        self.analyzer = analyzer
        self.threshold = threshold
        self._provenance = {}
        self.bullet_rows = []        # records.PLAYBOOK_BULLET_FIELDS
        self.commits = []            # one per maturity handed to commit()

    def snapshot(self):
        ids = [b["id"] for b in bullets_with_sections(self.text)]
        return A2Snapshot(tuple(self._provenance[i] for i in ids), self.text, self.next_global_id)

    def commit(self, maturity, source_decision_date, matured_on):
        source, day = pd.Timestamp(source_decision_date), pd.Timestamp(matured_on)
        r, c = maturity["reflector"], maturity["curator"]
        before = self.text
        rec = {"matured_on": str(day.date()), "source_decision_date": str(source.date()),
               "maturity_status": maturity["maturity_status"],
               "reflector_status": r["status"] if r else "not_run", "curator_status": c["status"] if c else "not_run",
               "added": [], "removed_by_dedup": [], "analyzer": None}
        if r is not None and r["status"] == "ok":
            text = counted(self.text, r["response"])
            if c is not None and c["status"] == "ok":
                ops = _SCHEMA._extract_and_validate_operations(c["response"])["operations"]
                old = {b["id"] for b in bullets_with_sections(text)}
                text, next_id = apply_curator_operations(text, ops, self.next_global_id)
                if sha256(text) != c["attempts"][-1].get("curator_playbook_sha256"):
                    raise RuntimeError(f"re-applied operations for {source.date()} differ from the Curator's playbook")
                self.next_global_id = next_id
                for b in bullets_with_sections(text):
                    if b["id"] in old:
                        continue
                    self._provenance[b["id"]] = PlaybookEntry(b["content"], source, day, bullet_id=b["id"])
                    self.bullet_rows.append({
                        "arm": "A2", "bullet_id": b["id"], "section": b["section"], "content": b["content"],
                        "operation": "ADD", "created_decision_date": str(day.date()),
                        "source_matured_decision_date": str(source.date()), "helpful_count": None,
                        "harmful_count": None, "deleted_decision_date": None,
                        "specificity_class": "unclassified", "specificity_note": None,
                        "specificity_classified_by": None})
                    rec["added"].append(b["id"])
                if self.analyzer is not None:
                    ids_before = {b["id"] for b in bullets_with_sections(text)}
                    text = self.analyzer.analyze(playbook=text, threshold=self.threshold, merge=BULLETPOINT_MERGE)
                    removed = sorted(ids_before - {b["id"] for b in bullets_with_sections(text)})
                    rec["analyzer"] = {"threshold": self.threshold, "merge": BULLETPOINT_MERGE}
                    rec["removed_by_dedup"] = removed
                    for row in self.bullet_rows:
                        if row["bullet_id"] in removed:
                            row["deleted_decision_date"] = str(day.date())
            self.text = text
        rec.update({"sha256_before": sha256(before), "sha256_after": sha256(self.text),
                    "n_bullets_after": len(bullets_with_sections(self.text)), "text_after": self.text})
        self.commits.append(rec)

    def bullet_table(self):
        counts = {b["id"]: b for b in bullets_with_sections(self.text)}
        df = records.empty_bullet_table()
        rows = []
        for row in self.bullet_rows:
            b = counts.get(row["bullet_id"])
            rows.append({**row, "helpful_count": b["helpful"] if b else None, "harmful_count": b["harmful"] if b else None})
        if rows:
            df = pd.DataFrame(rows).astype({k: v for k, v in records.PLAYBOOK_BULLET_FIELDS.items()})
        return df


# ---------------------------------------------------------------- the three callables

def call_facts(info):
    prov = info.get("provider") or {}
    return {"prompt_tokens": info.get("prompt_num_tokens"), "completion_tokens": info.get("response_num_tokens"),
            "latency_s": info.get("call_time"), "provider": {k: prov.get(k) for k in PROVIDER_FIELDS}}


class LiveRoles:
    """generate / reflect / curate for replay.run_replay, on the ACE role classes."""

    def __init__(self, client, contexts, names, null_sorted, decision_dates, a1_format, log_dir,
                 memory_a2, label):
        if a1_format not in A1_WINDOW_FORMATS:
            raise ValueError(f"A1 window format not chosen (got {a1_format!r}); v9.2 leaves it to the user")
        self.contexts, self.names, self.null = contexts, names, null_sorted
        self.decision_dates = pd.DatetimeIndex(decision_dates)
        self.n_learning = len(self.decision_dates) - DELAY
        self.a1_format, self.log_dir, self.memory_a2, self.label = a1_format, str(log_dir), memory_a2, label
        self.generator = Generator(client, API_PROVIDER, MODEL, MAX_TOKENS)
        self.reflector = Reflector(client, API_PROVIDER, MODEL, MAX_TOKENS)
        self.curator = Curator(client, API_PROVIDER, MODEL, MAX_TOKENS)

    def call_id(self, arm, day, role, attempt_no):
        return f"{self.label}-{arm}-{pd.Timestamp(day):%Y%m%d}-{role}-a{attempt_no}"

    def generation_inputs(self, arm, day, snapshot):
        """(playbook, context) for one arm on one date; the question and reflection never vary."""
        d = self.contexts.get(day)
        if arm == "A0":
            return EMPTY_PLAYBOOK, d.context
        if arm == "A1":
            block = a1_window_block(tuple(snapshot), self.names, self.a1_format)
            return EMPTY_PLAYBOOK, d.context + ("\n\n" + block if block else "")
        if arm == "A2":
            return snapshot.text, d.context
        raise ValueError(arm)

    def generate(self, arm, day, snapshot, attempt_no):
        playbook, context = self.generation_inputs(arm, day, snapshot)
        call_id = self.call_id(arm, day, "gen", attempt_no)
        response, bullet_ids, info = self.generator.generate(
            question=QUESTION, playbook=playbook, context=context, reflection="(empty)",
            use_json_mode=USE_JSON_MODE, call_id=call_id, log_dir=self.log_dir)
        return {"response": response, "bullet_ids": list(bullet_ids), "call_id": call_id,
                "playbook_sha256": sha256(playbook), **call_facts(info)}

    def feedback(self, decision, realized):
        d = self.contexts.get(decision.decision_date)
        return fb.render(decision.decision_date, d.entry_date, decision.mature_date, d.universe, self.names,
                         dict(decision.scores), dict(realized), self.null)

    def reflect(self, decision, realized, attempt_no):
        f = self.feedback(decision, realized)
        playbook = self.memory_a2.snapshot().text
        call_id = self.call_id("A2", decision.decision_date, "reflect", attempt_no)
        response, tags, info = self.reflector.reflect(
            question=QUESTION, reasoning_trace=decision.response,
            predicted_answer=extract_answer(decision.response), ground_truth=f["ground_truth"],
            environment_feedback=f["environment_feedback"],
            bullets_used=extract_playbook_bullets(playbook, list(decision.bullet_ids)),
            use_ground_truth=True, use_json_mode=USE_JSON_MODE, call_id=call_id, log_dir=self.log_dir)
        q = f["quantile"]
        return {"response": response, "call_id": call_id, "n_bullet_tags": len(tags),
                "feedback": {"csic": None if math.isnan(f["csic"]) else f["csic"], "n_pairs": f["n_pairs"],
                             "quantile": None if math.isnan(q) else q},
                **call_facts(info)}

    def curate(self, snapshot, reflection, decision, attempt_no):
        playbook = counted(snapshot.text, reflection)
        call_id = self.call_id("A2", decision.decision_date, "curate", attempt_no)
        new_playbook, _, operations, info = self.curator.curate(
            current_playbook=playbook, recent_reflection=reflection,
            question_context=self.contexts.get(decision.decision_date).context,
            current_step=decision.decision_index + 1, total_samples=self.n_learning,
            token_budget=PLAYBOOK_TOKEN_BUDGET, playbook_stats=get_playbook_stats(playbook),
            use_ground_truth=True, use_json_mode=USE_JSON_MODE, call_id=call_id, log_dir=self.log_dir,
            next_global_id=snapshot.next_global_id)
        return {"response": info.get("response"), "call_id": call_id, "curator_operations": len(operations),
                "curator_playbook_sha256": sha256(new_playbook), **call_facts(info)}

    @staticmethod
    def curator_check(response):
        kind = ap.json_object_check(response)
        if kind is not None:
            return kind
        try:
            _SCHEMA._extract_and_validate_operations(response)
        except Exception:
            return "curator_schema"
        return None
