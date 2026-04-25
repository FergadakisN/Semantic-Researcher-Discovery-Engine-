"""
Evaluation script: Hit@k and MRR over a benchmark file.

Benchmark format (evaluation/benchmark.json):
[
  {
    "query": "...",
    "expected_researcher_ids": ["Q123", "Q456"]   // at least one must appear in top-k
  },
  ...
]

Run:
    python evaluation/evaluate.py
    python evaluation/evaluate.py --field title_topics --strategy max
"""

import json
import sys
import argparse
from pathlib import Path

# allow imports from src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from retrieve import RetrievalIndex
from aggregate import aggregate

BENCHMARK_PATH = Path(__file__).parent / "benchmark.json"


def evaluate(
    benchmark: list[dict],
    index: RetrievalIndex,
    strategy: str,
    ks: list[int],
    paper_pool: int = 100,
) -> dict:
    hits = {k: 0 for k in ks}
    reciprocal_ranks = []

    for item in benchmark:
        query = item["query"]
        expected_ids = set(item["expected_researcher_ids"])

        paper_matches = index.query(query, top_n=paper_pool)
        ranked = aggregate(paper_matches, strategy=strategy, top_k_researchers=max(ks))

        ranked_ids = [r.researcher_id for r in ranked]

        # MRR: first relevant rank
        rr = 0.0
        for rank, rid in enumerate(ranked_ids, start=1):
            if rid in expected_ids:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        # Hit@k
        for k in ks:
            if any(rid in expected_ids for rid in ranked_ids[:k]):
                hits[k] += 1

    n = len(benchmark)
    results = {
        "n_queries": n,
        "MRR": round(sum(reciprocal_ranks) / n, 4) if n else 0.0,
    }
    for k in ks:
        results[f"Hit@{k}"] = round(hits[k] / n, 4) if n else 0.0

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default=str(BENCHMARK_PATH))
    parser.add_argument(
        "--field",
        choices=["title_only", "title_topics"],
        default="title_only",
    )
    parser.add_argument(
        "--strategy",
        default="sum_top_3",
        help="Aggregation strategy: max | sum_top_3 | mean_top_3",
    )
    parser.add_argument("--paper-pool", type=int, default=100)
    parser.add_argument(
        "--compare-all",
        action="store_true",
        help="Run all field × strategy combinations and print a comparison table",
    )
    args = parser.parse_args()

    benchmark = json.loads(Path(args.benchmark).read_text())
    print(f"Loaded {len(benchmark)} benchmark queries")

    ks = [1, 3, 5, 10]

    if args.compare_all:
        fields = ["title_only", "title_topics"]
        strategies = ["max", "sum_top_3", "mean_top_3"]
        for field in fields:
            index = RetrievalIndex(field=field)
            for strategy in strategies:
                res = evaluate(benchmark, index, strategy, ks, args.paper_pool)
                label = f"{field} / {strategy}"
                metrics = "  ".join(
                    f"Hit@{k}={res[f'Hit@{k}']}" for k in ks
                )
                print(f"{label:35s}  MRR={res['MRR']}  {metrics}")
    else:
        index = RetrievalIndex(field=args.field)
        res = evaluate(benchmark, index, args.strategy, ks, args.paper_pool)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
