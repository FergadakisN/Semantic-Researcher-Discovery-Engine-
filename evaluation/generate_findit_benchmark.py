"""Generate a realistic benchmark from findit supervisor-thesis data.

The professor provided findit exports mapping supervisor Wikidata IDs to the
English titles of student theses they supervised.  The benchmark task:
  query  = thesis title
  target = the supervisor must appear in the top-k results

This yields a harder, more realistic test than the LLM-rewritten queries
(benchmark_llm.json), because the thesis titles are not derived from the
supervisor's own publications.

Input:  evaluation/supervisors.json
Output: evaluation/benchmark_findit.json
"""

import json
from pathlib import Path

SUPERVISORS_PATH = Path(__file__).parent / "supervisors.json"
OUTPUT_PATH = Path(__file__).parent / "benchmark_findit.json"


def main():
    supervisors = json.loads(SUPERVISORS_PATH.read_text())

    benchmark = []
    for researcher_id, info in supervisors.items():
        for title in info["titles"]:
            benchmark.append(
                {
                    "query": title,
                    "expected_researcher_ids": [researcher_id],
                }
            )

    OUTPUT_PATH.write_text(json.dumps(benchmark, indent=2, ensure_ascii=False))
    print(f"Generated {len(benchmark)} benchmark entries → {OUTPUT_PATH}")
    for rid, info in supervisors.items():
        count = sum(1 for e in benchmark if rid in e["expected_researcher_ids"])
        print(f"  {info['name']} ({rid}): {count} theses")


if __name__ == "__main__":
    main()
