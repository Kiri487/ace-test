#!/usr/bin/env python3
"""Run ACE on the TW screening task.

Mirrors eval/finance/run.py, except the samples are decision dates generated
from the trading-day calendar rather than rows in a JSONL file - the "data" here
is a schedule, and the labels do not exist until a rule has been backtested.

    .venv/bin/python -m eval.twstock.run --api_provider clinepass \\
        --generator_model cline-pass/deepseek-v4-flash ... --json_mode

The splits are chronological with a gap of one full holding period between them,
so a sample's outcome window never overlaps the next split's decision dates.
"""

import argparse
import json
import os

from ace.ace import ACE

from . import market
from .data_processor import DataProcessor


def parse_args():
    p = argparse.ArgumentParser(description="ACE on TW stock screening rules")
    p.add_argument("--task_name", default="twstock")
    p.add_argument("--mode", default="offline",
                   choices=["offline", "online", "eval_only"])

    p.add_argument("--api_provider", default="clinepass")
    p.add_argument("--generator_model", default="cline-pass/deepseek-v4-flash")
    p.add_argument("--reflector_model", default="cline-pass/deepseek-v4-flash")
    p.add_argument("--curator_model", default="cline-pass/deepseek-v4-flash")
    p.add_argument("--max_tokens", type=int, default=4096)

    p.add_argument("--train_start", default="2023-01-01",
                   help="first decision date is the first trading day on or after this")
    p.add_argument("--n_train", type=int, default=5)
    p.add_argument("--n_val", type=int, default=5)
    p.add_argument("--n_test", type=int, default=5)

    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--max_num_rounds", type=int, default=3)
    p.add_argument("--curator_frequency", type=int, default=1)
    p.add_argument("--eval_steps", type=int, default=5)
    p.add_argument("--online_eval_frequency", type=int, default=15)
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--playbook_token_budget", type=int, default=6000)
    p.add_argument("--json_mode", action="store_true")
    p.add_argument("--no_ground_truth", action="store_true")
    p.add_argument("--save_path", default="results")
    p.add_argument("--test_workers", type=int, default=2)
    p.add_argument("--initial_playbook_path", default=None)
    p.add_argument("--use_bulletpoint_analyzer", action="store_true")
    p.add_argument("--bulletpoint_analyzer_threshold", type=float, default=0.90)
    p.add_argument("--bulletpoint_merge", action="store_true")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--curator_batch_size", type=int, default=None)
    p.add_argument("--curator_num_groups", type=int, default=None)
    p.add_argument("--augmented_shuffling", action="store_true", default=True)
    return p.parse_args()


def build_schedule(train_start, n_train, n_val, n_test):
    """Chronological splits separated by one holding period.

    Asking for n+1 dates from the end of the previous split's holding window and
    dropping the first is what creates the gap: the dropped date is the one that
    would have overlapped.
    """
    train = market.decision_dates(train_start, n_train)
    if len(train) < n_train:
        raise SystemExit(f"only {len(train)} train dates available from {train_start}")
    after_train = market.holding_window(train[-1])[1]
    val = market.decision_dates(after_train, n_val + 1)[1:]
    after_val = market.holding_window(val[-1])[1]
    test = market.decision_dates(after_val, n_test + 1)[1:]
    if len(val) < n_val or len(test) < n_test:
        raise SystemExit("not enough trading days left for the requested splits")
    return train, val, test


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("ACE SYSTEM - TW STOCK SCREENING")
    print("=" * 60)
    print(f"Mode: {args.mode.upper()}")
    print(f"Generator Model: {args.generator_model}")
    print("=" * 60 + "\n")

    market.login()
    train_d, val_d, test_d = build_schedule(
        args.train_start, args.n_train, args.n_val, args.n_test)
    for label, dates in (("train", train_d), ("val", val_d), ("test", test_d)):
        print(f"  {label:5} {[str(d)[:10] for d in dates]}")
    print()

    processor = DataProcessor(task_name=args.task_name)
    to_raw = lambda ds: [{"date": str(d)[:10]} for d in ds]      # noqa: E731
    train_samples = processor.process_task_data(to_raw(train_d))
    val_samples = processor.process_task_data(to_raw(val_d))
    test_samples = processor.process_task_data(to_raw(test_d))

    initial_playbook = None
    if args.initial_playbook_path and os.path.exists(args.initial_playbook_path):
        initial_playbook = open(args.initial_playbook_path, encoding="utf-8").read()
        print(f"Loaded initial playbook from {args.initial_playbook_path}\n")
    else:
        print("Using empty playbook as initial playbook\n")

    ace_system = ACE(
        api_provider=args.api_provider,
        generator_model=args.generator_model,
        reflector_model=args.reflector_model,
        curator_model=args.curator_model,
        max_tokens=args.max_tokens,
        initial_playbook=initial_playbook,
        use_bulletpoint_analyzer=args.use_bulletpoint_analyzer,
        bulletpoint_analyzer_threshold=args.bulletpoint_analyzer_threshold,
        bulletpoint_merge=args.bulletpoint_merge,
    )

    config = {
        "num_epochs": args.num_epochs,
        "max_num_rounds": args.max_num_rounds,
        "curator_frequency": args.curator_frequency,
        "eval_steps": args.eval_steps,
        "online_eval_frequency": args.online_eval_frequency,
        "save_steps": args.save_steps,
        "playbook_token_budget": args.playbook_token_budget,
        "task_name": args.task_name,
        "mode": args.mode,
        "json_mode": args.json_mode,
        "no_ground_truth": args.no_ground_truth,
        "save_dir": args.save_path,
        "test_workers": args.test_workers,
        "initial_playbook_path": args.initial_playbook_path,
        "use_bulletpoint_analyzer": args.use_bulletpoint_analyzer,
        "bulletpoint_analyzer_threshold": args.bulletpoint_analyzer_threshold,
        "bulletpoint_merge": args.bulletpoint_merge,
        "api_provider": args.api_provider,
        "batch_size": args.batch_size,
        "curator_batch_size": args.curator_batch_size,
        "curator_num_groups": args.curator_num_groups,
        "augmented_shuffling": args.augmented_shuffling,
        "max_samples": max(args.n_train, args.n_val, args.n_test),
        # Recorded so a set of results can be traced back to the code that made
        # it - run_config.json otherwise stores no version at all.
        "decision_dates": {
            "train": [str(d)[:10] for d in train_d],
            "val": [str(d)[:10] for d in val_d],
            "test": [str(d)[:10] for d in test_d],
        },
    }

    ace_system.run(
        mode=args.mode,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        data_processor=processor,
        config=config,
    )

    print("\nEvaluations cached this run:", len(processor._cache))
    print(json.dumps(
        {k[:60]: {kk: v.get(kk) for kk in ("rule_return", "benchmark_return", "correct")}
         for k, v in list(processor._cache.items())[:5]},
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
