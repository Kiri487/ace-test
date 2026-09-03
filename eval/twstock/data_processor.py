"""The ACE adapter: three methods plus one optional hook.

    process_task_data   build {context, question, target} per decision date
    answer_is_correct   compile the rule, backtest it, beat the benchmark?
    evaluate_accuracy   win rate across samples - this selects best_playbook
    environment_feedback  what the Reflector is told about *why* it was wrong

The last one is not part of the upstream interface. `Reflector.reflect()`
declares an `environment_feedback` argument but ace.py hardcodes it to
"Predicted answer does not match ground truth", which is enough for a task with
a string ground truth and useless for one whose feedback is a number. The paper
does feed real execution feedback (AppWorld), so populating it here restores the
paper's design rather than departing from it. The hook is optional: a processor
without it falls back to the hardcoded string, so FiNER is unaffected.

Two deliberate asymmetries, both load-bearing:

* `answer_is_correct` binarises as "beat the same-window equal-weight
  benchmark", not "made money". Scoring raw profit would mark every rule
  correct in a rising market and teach the Reflector to explain market moves as
  if they were rule quality.
* `evaluate_accuracy` returns a win rate, so it stays in [0, 1]. ace.py
  initialises `best_accuracy = 0.0` and only ever replaces on `>`, so a metric
  that can go negative (excess return, Sharpe) would never beat the floor and
  `best_playbook` would silently remain the empty starting playbook.
"""

import ast
import json
import logging
from typing import Any, Dict, List

from . import market
from .evaluate import evaluate_rule
from .rules import describe as describe_dsl, InvalidRule

logger = logging.getLogger(__name__)

TASK_INSTRUCTION = """你要為一個台股選股任務提出一組「選股規則」。

規則會被套用在「市值前 50 大」的股票池上,篩選出一小批股票,等權買進並持有 {hold} 個交易日,
然後和「同一段期間、同一個股票池、等權持有全部 50 檔」的報酬相比。**贏過它才算成功。**

{dsl}

**你只會看到整體市場狀態,看不到任何個股代號或個股數值。** 請提出在這種市場環境下你認為
有效的篩選邏輯,而不是去猜特定公司。

final_answer 必須是一個 JSON 物件,格式如下:
{{"filters": [{{"field": "<欄位>", "op": "<運算子>", "value": <數字,positive/negative 不需要>}}],
  "rank_by": "<欄位>", "direction": "high" 或 "low", "n": <整數>}}"""


def _parse_rule(text: Any) -> Dict:
    """Recover the rule object from whatever extract_answer handed back.

    `extract_answer` does `str(parsed["final_answer"])`, so a nested object
    arrives as a Python repr with single quotes rather than JSON. Accept both,
    and let anything else surface as InvalidRule so it is scored, not crashed.
    """
    if isinstance(text, dict):
        return text
    s = str(text).strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s.split("\n", 1)[-1] if "\n" in s else s
    for loader in (json.loads, ast.literal_eval):
        try:
            out = loader(s)
            if isinstance(out, dict):
                return out
        except Exception:
            continue
    raise InvalidRule(f"could not parse a rule object from: {s[:200]}")


class DataProcessor:
    """ACE DataProcessor for the TW screening task."""

    def __init__(self, task_name: str = "twstock"):
        self.task_name = task_name
        # (rule json, date) -> evaluation. answer_is_correct is called more than
        # once per sample, and each miss costs two real backtests.
        self._cache: Dict[str, Dict] = {}
        self._last: Dict[str, Dict] = {}

    # -- required interface -------------------------------------------------

    def process_task_data(self, raw_data: List[Dict]) -> List[Dict]:
        processed = []
        for item in raw_data:
            date = item["date"]
            ctx = market.market_context(date)
            context = (
                f"決策日:{date}\n"
                f"股票池:市值前 {market.UNIVERSE_SIZE} 大(該日重新計算)\n"
                f"持有期:{market.HOLD_DAYS} 個交易日,期間不換股\n"
                f"此決策日前 {market.HOLD_DAYS} 個交易日,股票池的表現:\n"
                f"  - 平均報酬 {ctx['universe_return_pct']:+.2f}%\n"
                f"  - 個股報酬離散度(標準差) {ctx['dispersion_pct']:.2f}%"
            )
            processed.append({
                "context": context,
                "question": TASK_INSTRUCTION.format(
                    hold=market.HOLD_DAYS, dsl=describe_dsl()),
                # target carries what is knowable before the rule exists: which
                # window to score in. The outcome cannot live here because it
                # depends on the rule.
                "target": json.dumps({"date": date}, ensure_ascii=False),
                "others": {"task": self.task_name, "date": date},
            })
        return processed

    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        return bool(self._evaluate(predicted, ground_truth).get("correct"))

    def evaluate_accuracy(self, out: List[str], target: List[str]) -> float:
        """Win rate over the samples. Non-negative by construction - see the
        module docstring for why that matters."""
        if not out:
            return 0.0
        wins = sum(1 for p, g in zip(out, target)
                   if self._evaluate(p, g).get("correct"))
        return wins / len(out)

    # -- optional hook ------------------------------------------------------

    def environment_feedback(self, predicted: str, ground_truth: str) -> str:
        r = self._evaluate(predicted, ground_truth)
        if not r.get("valid"):
            return (f"規則無法執行:{r.get('error')}。"
                    f"請確認欄位、運算子與 n 都在允許範圍內,且篩選後至少剩 3 檔。")
        return (
            f"回測結果({r['entry']} ~ {r['exit']},{r['n_selected']} 檔等權持有):\n"
            f"  你的規則報酬     {r['rule_return']*100:+.2f}%\n"
            f"  基準(全 50 檔等權)  {r['benchmark_return']*100:+.2f}%\n"
            f"  超額報酬         {r['excess']*100:+.2f}%  "
            f"({'贏過' if r['correct'] else '輸給'}基準)"
        )

    # -- internals ----------------------------------------------------------

    def _evaluate(self, predicted: str, ground_truth: str) -> Dict:
        try:
            date = json.loads(ground_truth)["date"]
        except Exception:
            date = str(ground_truth).strip()

        try:
            rule = _parse_rule(predicted)
        except InvalidRule as e:
            return {"valid": False, "error": str(e), "correct": False}

        key = json.dumps([rule, date], sort_keys=True, ensure_ascii=False)
        if key not in self._cache:
            try:
                self._cache[key] = evaluate_rule(rule, date)
            except Exception as e:                       # a backtest blew up
                logger.warning("evaluate_rule failed for %s: %s", date, e)
                self._cache[key] = {"valid": False,
                                    "error": f"{type(e).__name__}: {e}",
                                    "correct": False}
        self._last = self._cache[key]
        return self._cache[key]
