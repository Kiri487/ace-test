"""Exercise the DataProcessor without ACE, so the adapter can be checked before
finlab is installed alongside the ACE dependencies.

    .venv-stock-tests/bin/python -m eval.twstock.test_adapter
"""

import json

from . import market
from .data_processor import DataProcessor


def main():
    market.login()
    dates = market.decision_dates("2023-01-01", 3)
    raw = [{"date": str(d)[:10]} for d in dates]

    proc = DataProcessor()
    samples = proc.process_task_data(raw)

    print("=" * 74)
    print("  what the LLM actually sees for sample 0")
    print("=" * 74)
    print("--- context ---")
    print(samples[0]["context"])
    print("\n--- question (truncated) ---")
    print(samples[0]["question"][:900])
    print("\n--- target ---")
    print(samples[0]["target"])

    print("\n" + "=" * 74)
    print("  leakage check: does any stock id appear in the prompt?")
    print("=" * 74)
    members = market.universe(dates[0])
    blob = samples[0]["context"] + samples[0]["question"]
    leaked = [s for s in members if s in blob]
    print(f"   universe has {len(members)} ids; {len(leaked)} of them appear: {leaked}")
    print(f"   -> {'PASS' if not leaked else 'FAIL - identity is reaching the prompt'}")

    print("\n" + "=" * 74)
    print("  a well-formed answer, the way extract_answer would hand it over")
    print("=" * 74)
    rule = {"filters": [{"field": "foreign_buy", "op": "positive"},
                        {"field": "roe", "op": "top_pct", "value": 50}],
            "rank_by": "rev_yoy", "direction": "high", "n": 10}
    # extract_answer does str(parsed["final_answer"]) - a dict arrives as a
    # Python repr, not JSON. Feed it exactly that shape.
    predicted = str(rule)
    tgt = samples[0]["target"]
    print(f"   predicted (repr form): {predicted[:90]}...")
    print(f"   answer_is_correct    : {proc.answer_is_correct(predicted, tgt)}")
    print("   environment_feedback :")
    for line in proc.environment_feedback(predicted, tgt).splitlines():
        print("      " + line)

    print("\n" + "=" * 74)
    print("  malformed answers must score, not crash")
    print("=" * 74)
    for bad in ["No final answer found",
                '{"filters": [{"field": "pe_ratio", "op": "top_pct", "value": 10}], '
                '"rank_by": "roe", "direction": "high", "n": 10}',
                '{"filters": [], "rank_by": "roe", "direction": "sideways", "n": 10}']:
        ok = proc.answer_is_correct(bad, tgt)
        fb = proc.environment_feedback(bad, tgt)
        print(f"   {bad[:64]:<66} correct={ok}")
        print(f"      -> {fb[:110]}")

    print("\n" + "=" * 74)
    print("  evaluate_accuracy across the 3 samples (this selects best_playbook)")
    print("=" * 74)
    answers = [predicted] * len(samples)
    targets = [s["target"] for s in samples]
    acc = proc.evaluate_accuracy(answers, targets)
    for s in samples:
        r = proc._evaluate(predicted, s["target"])
        print(f"   {r.get('date')}  rule {r.get('rule_return', 0)*100:+.2f}%"
              f"  bench {r.get('benchmark_return', 0)*100:+.2f}%"
              f"  -> {'win' if r.get('correct') else 'loss'}")
    print(f"   win rate = {acc:.3f}   (in [0,1], so it can beat best_accuracy=0.0)")
    print(f"   backtests cached: {len(proc._cache)}")


if __name__ == "__main__":
    main()
