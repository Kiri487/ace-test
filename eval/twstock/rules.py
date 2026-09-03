"""The screening DSL: schema, validation, compilation, and random sampling.

The LLM's output is never executed. It is parsed into this restricted grammar
and interpreted, which buys three things at once: a rule that cannot crash the
harness, a bounded search space that limits overfitting, and - because the space
is bounded - the ability to *sample* from it, which is what makes the
random-rule control arm possible at all.

All operators are relative (percentile) or anchored at a natural zero, so a rule
carries no absolute thresholds and means the same thing in any window.
"""

import random

from .market import FIELD_SPECS, FIELD_NAMES

OPS = ("top_pct", "bottom_pct", "positive", "negative")
PCT_OPS = ("top_pct", "bottom_pct")
MIN_SELECTED = 3          # fewer than this and the rule is treated as invalid
MIN_N, MAX_N = 3, 25


class InvalidRule(ValueError):
    """The rule did not parse, or asked for something the DSL cannot express."""


def validate(rule):
    """Raise InvalidRule with a message the Reflector can learn from."""
    if not isinstance(rule, dict):
        raise InvalidRule("rule must be a JSON object")

    filters = rule.get("filters", [])
    if not isinstance(filters, list):
        raise InvalidRule("'filters' must be a list")
    for i, f in enumerate(filters):
        if not isinstance(f, dict):
            raise InvalidRule(f"filters[{i}] must be an object")
        field, op = f.get("field"), f.get("op")
        if field not in FIELD_SPECS:
            raise InvalidRule(
                f"filters[{i}] unknown field {field!r}; allowed: {list(FIELD_NAMES)}")
        if op not in OPS:
            raise InvalidRule(
                f"filters[{i}] unknown op {op!r}; allowed: {list(OPS)}")
        if op in PCT_OPS:
            v = f.get("value")
            if not isinstance(v, (int, float)) or not 1 <= v <= 99:
                raise InvalidRule(
                    f"filters[{i}] op {op} needs a 'value' percentage between 1 and 99")
        elif not FIELD_SPECS[field][2]:
            raise InvalidRule(
                f"filters[{i}] op {op!r} needs a field with a meaningful zero; "
                f"{field!r} has none")

    rank_by = rule.get("rank_by")
    if rank_by not in FIELD_SPECS:
        raise InvalidRule(
            f"unknown rank_by {rank_by!r}; allowed: {list(FIELD_NAMES)}")
    if rule.get("direction") not in ("high", "low"):
        raise InvalidRule("'direction' must be 'high' or 'low'")
    n = rule.get("n")
    if not isinstance(n, int) or not MIN_N <= n <= MAX_N:
        raise InvalidRule(f"'n' must be an integer between {MIN_N} and {MAX_N}")
    return True


def _apply_filter(series, op, value):
    """A boolean mask over `series`. NaN never passes: an unknown value is not
    evidence that the condition holds."""
    s = series.dropna()
    if op == "positive":
        keep = s[s > 0].index
    elif op == "negative":
        keep = s[s < 0].index
    else:
        k = max(1, int(round(len(s) * value / 100.0)))
        keep = (s.nlargest(k) if op == "top_pct" else s.nsmallest(k)).index
    return set(keep)


def select(rule, values):
    """Apply `rule` to {field: Series} and return the chosen stock ids.

    Raises InvalidRule when the rule is well-formed but selects too few names -
    a screen that leaves one stock is not a portfolio, and letting it through
    would reward degenerate rules.
    """
    validate(rule)

    candidates = None
    for f in rule.get("filters", []):
        passed = _apply_filter(values[f["field"]], f["op"], f.get("value"))
        candidates = passed if candidates is None else (candidates & passed)
    if candidates is None:
        candidates = set(values[rule["rank_by"]].dropna().index)

    ranked = values[rule["rank_by"]].reindex(sorted(candidates)).dropna()
    if len(ranked) < MIN_SELECTED:
        raise InvalidRule(
            f"only {len(ranked)} stocks passed the filters; need at least {MIN_SELECTED}")

    top = (ranked.nlargest(rule["n"]) if rule["direction"] == "high"
           else ranked.nsmallest(rule["n"]))
    return list(top.index)


def describe():
    """The DSL description handed to the LLM. Generated from the same specs the
    compiler uses, so the prompt cannot drift away from what is accepted."""
    lines = ["可用欄位:"]
    for name, (_key, desc, has_zero) in FIELD_SPECS.items():
        zero = ",可用 positive / negative" if has_zero else ""
        lines.append(f"  - {name}: {desc}{zero}")
    lines += [
        "",
        "可用運算子:",
        "  - top_pct(value):    該欄位數值前 value% 的股票",
        "  - bottom_pct(value): 該欄位數值後 value% 的股票",
        "  - positive:          該欄位大於 0",
        "  - negative:          該欄位小於 0",
        "",
        f"n 必須是 {MIN_N} 到 {MAX_N} 之間的整數。filters 可以是空的。",
        "多個 filter 之間是 AND(必須同時滿足),而且每個百分位都是相對於「完整的 "
        "50 檔股票池」計算,不是相對於前一個 filter 篩剩下的。所以 filter 的先後"
        "順序不影響結果。",
        f"篩選後若少於 {MIN_SELECTED} 檔,這條規則會被判為無效。",
    ]
    return "\n".join(lines)


def random_rule(rng=None):
    """Sample a rule uniformly from the DSL.

    This is the control arm: if the evolved playbook cannot beat rules drawn
    from the same space, nothing was learned. It exists only because the space
    is bounded.
    """
    rng = rng or random
    filters = []
    for field in rng.sample(list(FIELD_NAMES), rng.randint(0, 3)):
        if FIELD_SPECS[field][2] and rng.random() < 0.5:
            filters.append({"field": field, "op": rng.choice(["positive", "negative"])})
        else:
            filters.append({"field": field,
                            "op": rng.choice(list(PCT_OPS)),
                            "value": rng.choice([20, 30, 50, 70])})
    return {
        "filters": filters,
        "rank_by": rng.choice(list(FIELD_NAMES)),
        "direction": rng.choice(["high", "low"]),
        "n": rng.choice([5, 10, 15]),
    }
