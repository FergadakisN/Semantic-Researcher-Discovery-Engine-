# Evaluation Approach: LLM-Rewritten Query Benchmark

## Overview

This project uses an **LLM-based query rewriting** strategy to evaluate the semantic researcher search system. It bridges the gap between two extremes:

| Approach | Realism | Scale | Cost |
|---|---|---|---|
| Raw leave-one-out | Low — uses exact paper titles | High — 50k+ queries | Free |
| Human annotation | High — real user intent | Low — 6–20 queries | Expensive |
| **LLM-rewritten queries** | **Medium-high — mimics student phrasing** | **Medium — 200–500 queries** | **Low** |

---

## The Problem with Raw Leave-One-Out

In leave-one-out evaluation, the query is the exact paper title:

> *"Artificial intelligence for natural product drug discovery"*

No student or researcher would type this verbatim into a search box. The phrasing is too close to the indexed text, making the task artificially easy and inflating performance metrics.

---

## The LLM-Rewriting Approach

### Step 1 — Sample papers

Randomly sample N papers (e.g. 200–500) from `data/papers.json`. Each paper has a title and one or more author IDs — these author IDs become the ground truth.

### Step 2 — Rewrite with an LLM

For each paper title, prompt an LLM to rewrite it as a realistic student query:

```
You are a master's student looking for a supervisor.
Rewrite this paper title as a natural student query (1-2 sentences).
Do not use academic jargon. Do not copy the title.

Paper title: "{title}"

Student query:
```

### Step 3 — Save as a benchmark file

The output is saved to `evaluation/benchmark_llm.json`:

```json
[
  {
    "original_title": "Artificial intelligence for natural product drug discovery",
    "query": "I want to work on AI to find new antibiotics",
    "expected_researcher_ids": ["Q19753684"]
  }
]
```

### Step 4 — Run evaluation

```bash
python evaluation/evaluate.py --benchmark evaluation/benchmark_llm.json
python evaluation/evaluate.py --benchmark evaluation/benchmark_llm.json --compare-all
```

---

## Example Rewrites

| Original paper title | LLM-rewritten student query |
|---|---|
| Deep learning for antibiotic resistance prediction | "I want to work on AI for antibiotics" |
| Transformer models for clinical text mining | "I am interested in NLP for healthcare" |
| Optimization of wind turbine control systems | "I want to research machine learning for wind energy" |

---

## What the Metrics Tell You

Running evaluation on both benchmarks produces a meaningful comparison:

```
benchmark_loo.json  (raw leave-one-out)   Hit@5 = 0.91   MRR = 0.80
benchmark_llm.json  (LLM-rewritten)       Hit@5 = 0.74   MRR = 0.61
```

The gap between the two scores quantifies how much natural query phrasing degrades retrieval performance. This is a real finding: it shows how sensitive the system is to phrasing variation, and motivates potential improvements (e.g. query expansion, topic-enriched embeddings).

---

## Reproducibility

The benchmark file (`benchmark_llm.json`) is committed to git alongside the generation script. This means:

- The professor can run `evaluate.py` directly without API access
- The exact queries used are transparent and inspectable
- Results are deterministic (temperature=0 was used during generation)

To regenerate the benchmark from scratch:

```bash
python evaluation/generate_llm_benchmark.py
```

---

## Limitations

- **LLM style bias** — all rewritten queries may sound stylistically similar, since they come from the same model and prompt
- **Prompt sensitivity** — different prompts produce different rewrites; the chosen prompt should be reported
- **Optimistic compared to real users** — the LLM knows the paper topic and rewrites faithfully; a real student may have a vague or imprecise idea
- **Spot-check recommended** — manually inspect ~20 generated queries to confirm they look realistic before trusting the metrics

---

## Files

| File | Description |
|---|---|
| `evaluation/generate_llm_benchmark.py` | Script to sample papers and generate rewritten queries via LLM |
| `evaluation/benchmark_llm.json` | Generated benchmark (committed to git for reproducibility) |
| `evaluation/benchmark.json` | Original small hand-curated benchmark (6 queries) |
| `evaluation/evaluate.py` | Evaluation script — works with any benchmark file via `--benchmark` flag |
