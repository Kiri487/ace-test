"""A1 reflection vs A2 Reflector output length, side by side. No LLM call, no threshold.

    .venv/bin/python -m eval.twstock.reflection_lengths --purpose result RUN_DIR [RUN_DIR ...]

A1's reflection length is measured, not constrained (arm_protocol docstring, 2026-09-15): ACE's
REFLECTOR_PROMPT has no length instruction, so neither has A1's. This reads llm_calls.parquet of each run
directory, through records.read_run with the given purpose (PILOT_LOCK), and reports the completion tokens
of answered calls per role - n, median, p90, max. What to do about the numbers is decided after they exist.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import arm_protocol as ap
from . import records

ROLES = {"A1 reflection": ap.A1_REFLECTION_ROLE, "A2 Reflector": "reflector"}


def compare(run_dirs, purpose):
    frames = []
    for d in run_dirs:
        d = Path(d)
        records.read_run(d, purpose)
        if (d / "llm_calls.parquet").exists():
            frames.append(pd.read_parquet(d / "llm_calls.parquet"))
    calls = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["role", "ok_call", "completion_tokens"])
    out = {}
    for label, role in ROLES.items():
        c = calls[(calls["role"] == role) & calls["ok_call"].fillna(False).astype(bool)]["completion_tokens"]
        c = c.dropna().astype(float).to_numpy()
        out[label] = {"role": role, "n": int(len(c)),
                      "median": float(np.median(c)) if len(c) else None,
                      "p90": float(np.percentile(c, 90)) if len(c) else None,
                      "max": float(c.max()) if len(c) else None}
    out["budget_reference_tokens"] = ap.A1_REFLECTION_REFERENCE_TOKENS
    out["note"] = "measured completion tokens of answered calls; no cap, no threshold"
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--purpose", required=True, choices=records.READ_PURPOSES)
    p.add_argument("run_dirs", nargs="+")
    args = p.parse_args()
    print(json.dumps(compare(args.run_dirs, args.purpose), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
