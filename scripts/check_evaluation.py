from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Enforce release evaluation thresholds")
    parser.add_argument("result", type=Path)
    parser.add_argument("--min-token-f1", type=float, default=0.35)
    parser.add_argument("--min-fact-recall", type=float, default=0.75)
    parser.add_argument("--min-full-facts", type=float, default=0.60)
    parser.add_argument("--min-source-hit", type=float, default=0.85)
    parser.add_argument("--min-source-at-1", type=float, default=0.75)
    parser.add_argument("--min-source-mrr", type=float, default=0.85)
    parser.add_argument("--min-no-answer", type=float, default=0.80)
    args = parser.parse_args()

    payload = json.loads(args.result.read_text(encoding="utf-8"))
    aggregate = payload.get("aggregate") or payload["metrics"]
    checks = {
        "mean_token_f1": (aggregate["mean_token_f1"], args.min_token_f1),
        "mean_required_fact_recall": (
            aggregate["mean_required_fact_recall"],
            args.min_fact_recall,
        ),
        "full_required_fact_coverage": (
            aggregate["full_required_fact_coverage"],
            args.min_full_facts,
        ),
        "source_hit_rate": (aggregate["source_hit_rate"], args.min_source_hit),
        "source_hit_at_1": (aggregate["source_hit_at_1"], args.min_source_at_1),
        "source_mrr": (aggregate["source_mrr"], args.min_source_mrr),
        "no_answer_accuracy": (aggregate["no_answer_accuracy"], args.min_no_answer),
    }
    failed = [
        name
        for name, (actual, minimum) in checks.items()
        if actual is None or actual < minimum
    ]
    for name, (actual, minimum) in checks.items():
        print(f"{name}: actual={actual!r}, required>={minimum}")
    if failed:
        print("Release gate failed: " + ", ".join(failed), file=sys.stderr)
        return 1
    print("Release gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
