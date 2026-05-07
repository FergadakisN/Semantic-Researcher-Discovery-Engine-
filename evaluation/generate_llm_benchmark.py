"""
Generate LLM-rewritten query benchmark for evaluation.

Samples N papers from data/papers.json, rewrites each title as a student query
using the CampusAI API, and saves the result to evaluation/benchmark_llm.json.

Usage:
    export CAMPUSAI_API_KEY="your-project-key"
    python evaluation/generate_llm_benchmark.py
    python evaluation/generate_llm_benchmark.py --n 200 --seed 0
    python evaluation/generate_llm_benchmark.py --output evaluation/benchmark_llm_debug.json --n 10
    python evaluation/generate_llm_benchmark.py --fix-existing evaluation/benchmark_llm.json
"""

import json
import os
import random
import argparse
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
from tqdm import tqdm

PROMPT_TEMPLATE = (
    'You are a master\'s student looking for a research supervisor at a technical university.\n'
    'Rewrite the following paper title as a short, natural student query (1-2 sentences).\n'
    'Do not copy the title verbatim. Use plain language, not academic jargon.\n'
    'Only output the query, nothing else.\n\n'
    'Paper title: "{title}"\n\n'
    'Student query:'
)

CAMPUSAI_URL = "https://api.campusai.compute.dtu.dk/v1/chat/completions"


def _is_thinking_leak(text: str) -> bool:
    """Return True if the response contains reasoning instead of a clean query."""
    markers = ["Thinking Process:", "Analyze the Request:", "**Role:**", "**Task:**"]
    return any(m in text for m in markers)


def _call_api(title: str, headers: dict) -> str | None:
    """
    Call the CampusAI API and return the rewritten query, or None on failure.

    Only the `content` field is used. The `reasoning_content` field contains
    the model's internal chain-of-thought and must never be treated as output.
    """
    prompt = PROMPT_TEMPLATE.format(title=title)
    response = requests.post(
        CAMPUSAI_URL,
        headers=headers,
        json={
            "model": os.environ.get("CAMPUSAI_MODEL", "Gemma 4"),
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": 300,
        },
        timeout=60,
    )
    if not response.ok:
        raise requests.HTTPError(
            f"{response.status_code} {response.reason} — {response.text[:300]}",
            response=response,
        )
    msg = response.json()["choices"][0]["message"]
    # Never fall back to reasoning_content — that is the thinking trace, not output.
    content = (msg.get("content") or "").strip()
    return content if content else None


def generate(papers: list[dict], headers: dict) -> list[dict]:
    benchmark = []
    for paper in tqdm(papers, desc="Generating queries"):
        title = paper["title"].strip()
        try:
            query = _call_api(title, headers)
        except Exception as exc:
            print(f"\nWarning: skipping {paper['paper_id']!r} — {exc}")
            continue

        if not query:
            print(f"\nWarning: skipping {paper['paper_id']!r} — empty response")
            continue

        if _is_thinking_leak(query):
            print(f"\nWarning: skipping {paper['paper_id']!r} — response contains thinking text")
            continue

        benchmark.append({
            "original_title": title,
            "query": query,
            "expected_researcher_ids": paper["researcher_ids"],
        })
    return benchmark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300, help="Number of papers to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent / "benchmark_llm.json"),
        help="Output path for the benchmark JSON",
    )
    parser.add_argument(
        "--fix-existing",
        metavar="PATH",
        help=(
            "Path to an existing benchmark file that contains corrupted entries "
            "(reasoning leaks). Re-queries the API for all bad entries and "
            "writes a cleaned file back to the same path."
        ),
    )
    args = parser.parse_args()

    api_key = os.environ.get("CAMPUSAI_API_KEY")
    if not api_key:
        raise RuntimeError("CAMPUSAI_API_KEY environment variable is not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    root = Path(__file__).parent.parent

    if args.fix_existing:
        _fix_existing(Path(args.fix_existing), headers)
        return

    papers = json.loads((root / "data" / "papers.json").read_text())
    eligible = [
        p for p in papers
        if p.get("title") and p["title"].strip() and p.get("researcher_ids")
    ]

    random.seed(args.seed)
    sample_size = min(args.n, len(eligible))
    sampled = random.sample(eligible, sample_size)

    benchmark = generate(sampled, headers)

    if not benchmark:
        print("\nNo queries generated — output file not written.")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(benchmark, indent=2, ensure_ascii=False))
    print(f"\nGenerated {len(benchmark)} queries → {output_path}")


def _fix_existing(path: Path, headers: dict) -> None:
    existing = json.loads(path.read_text())
    good = [e for e in existing if not _is_thinking_leak(e["query"])]
    bad = [e for e in existing if _is_thinking_leak(e["query"])]

    print(f"Loaded {len(existing)} entries: {len(good)} clean, {len(bad)} corrupted")
    if not bad:
        print("Nothing to fix.")
        return

    repaired = []
    for entry in tqdm(bad, desc="Re-querying corrupted entries"):
        title = entry["original_title"].strip()
        try:
            query = _call_api(title, headers)
        except Exception as exc:
            print(f"\nWarning: skipping {title!r:.60} — {exc}")
            continue

        if not query or _is_thinking_leak(query):
            print(f"\nWarning: skipping {title!r:.60} — still bad after retry")
            continue

        repaired.append({
            "original_title": title,
            "query": query,
            "expected_researcher_ids": entry["expected_researcher_ids"],
        })

    cleaned = good + repaired
    path.write_text(json.dumps(cleaned, indent=2, ensure_ascii=False))
    print(f"\nFixed {len(repaired)}/{len(bad)} corrupted entries → {path}")
    print(f"Final benchmark: {len(cleaned)} entries")


if __name__ == "__main__":
    main()
