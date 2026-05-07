# DTU Researcher Finder — Semantic Search System

A semantic search API that finds the right DTU research supervisor or collaborator from a free-text natural language query. Built as an NLP course project at the Technical University of Denmark.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [System Architecture](#2-system-architecture)
3. [Stage 1 — Data Acquisition](#3-stage-1--data-acquisition)
4. [Stage 2 — Preprocessing](#4-stage-2--preprocessing)
5. [Stage 3 — Embedding](#5-stage-3--embedding)
6. [Stage 4 — Retrieval](#6-stage-4--retrieval)
7. [Stage 5 — Aggregation](#7-stage-5--aggregation)
8. [Stage 6 — REST API](#8-stage-6--rest-api)
9. [Evaluation](#9-evaluation)
10. [Key Design Decisions](#10-key-design-decisions)
11. [How to Run — Local Setup](#11-how-to-run--local-setup)
12. [How to Run — Docker](#12-how-to-run--docker)
13. [Project Structure](#13-project-structure)
14. [Operational Characteristics](#14-operational-characteristics)

---

## 1. Problem Statement

Finding the right research supervisor or collaborator at a large technical university is a real information-retrieval problem. DTU has roughly **1,810 researchers** publishing across every engineering and science domain. There is no existing tool that lets a student type a description of their research interest and receive a ranked list of relevant supervisors.

Simple keyword search fails because it requires exact vocabulary overlap between the query and the document. A student who writes _"AI for renewable energy"_ will miss papers titled _"Neural network control of offshore wind parks"_ — even though the semantic meaning is closely related.

The solution is **dense semantic retrieval**: represent both the query and every paper as a point in a high-dimensional vector space where proximity means semantic similarity, then rank papers by their distance to the query vector.

The product is a REST API. There is no frontend UI; interaction is through HTTP requests or the auto-generated Swagger docs at `http://localhost:8000/docs`.

---

## 2. System Architecture

The system is built as a five-stage offline pipeline followed by a real-time query engine:

```
Wikidata (SPARQL)
       │
       ▼
  1. Data Acquisition     src/acquire.py       — fetch papers and researchers
       │
       ▼
  2. Preprocessing        src/preprocess.py    — clean and structure data
       │
       ▼
  3. Embedding            src/embed.py         — convert text → dense vectors
       │
       ▼
  4. Retrieval Index      src/retrieve.py      ◄── invoked at query time
       │
       ▼
  5. Aggregation          src/aggregate.py     ◄── invoked at query time
       │
       ▼
  6. REST API             src/api.py           ◄── exposed to the user
```

**Stages 1–3 are run once offline** to build the index. They are slow (minutes) but only need to run again when the data is refreshed. **Stages 4–6 serve live queries** and complete in approximately 38 milliseconds.

---

## 3. Stage 1 — Data Acquisition

**File:** [src/acquire.py](src/acquire.py)

### Source

All data comes from [Wikidata](https://www.wikidata.org), a free and open structured knowledge base. Wikidata records researcher affiliations, publication authorship, and paper subject topics. DTU's Wikidata entity ID is `Q1269766`.

Queries are issued against the [QLever SPARQL endpoint](https://qlever.cs.uni-freiburg.de/api/wikidata), a high-performance SPARQL engine that hosts the full Wikidata graph.

### Why Two Separate SPARQL Queries?

Two queries are run instead of one joined query, for an important reason.

**Query 1 — Paper titles:** Retrieves all publications authored by any currently DTU-affiliated researcher. Affiliation is matched via two Wikidata properties:

- `P108` (employer): direct employment at DTU
- `P1416` (affiliation): affiliation with DTU or any sub-unit (departments, institutes)

Past affiliations — those with an end date recorded via property `P582` — are excluded using a `MINUS` clause. This ensures only currently active DTU researchers are included.

**Query 2 — Paper topics:** For those same papers, fetches their `P921` (main subject) topic labels. Topics are optional annotations in Wikidata and are present for only about 54% of papers.

If the two queries were merged into a single SPARQL query with a `JOIN` on topics, all papers without topics would be silently dropped from the results. Keeping them separate means papers with no topic annotations are still included in the index; they simply have an empty topic list.

### Dataset Statistics

| Metric                                  | Value          |
| --------------------------------------- | -------------- |
| Unique researchers                      | 1,810          |
| Unique papers                           | 32,900         |
| Papers with at least one topic          | 17,733 (53.9%) |
| Average topics per paper (when present) | 1.9            |

Output is saved to `data/raw_sparql.json`.

---

## 4. Stage 2 — Preprocessing

**File:** [src/preprocess.py](src/preprocess.py)

The raw SPARQL results contain multiple rows per paper (one per author, one per topic). The preprocessing step normalizes this into a clean, structured format.

### Steps

1. **Topic aggregation:** All topics for a given paper are grouped into a list. Duplicate topics are removed case-insensitively to avoid _"Deep Learning"_ and _"deep learning"_ appearing twice.
2. **Paper deduplication:** Multiple SPARQL rows referencing the same paper (different co-authors) are collapsed into a single record with all co-author IDs and names collected together.
3. **Validation:** Papers with missing titles or titles shorter than 5 characters are discarded.
4. **Text variant generation:** Two text representations are pre-computed for each paper and stored alongside the record.

### Text Variants

| Variant             | Content                                    | Rationale                                         |
| ------------------- | ------------------------------------------ | ------------------------------------------------- |
| `text_title_only`   | The raw paper title                        | Clean signal, unaffected by sparse topic coverage |
| `text_title_topics` | `"Title: <title> Topics: <t1>, <t2>, ..."` | Potentially richer when topics are present        |

Both variants are stored so evaluation can determine which produces better retrieval.

### Outputs

- `data/papers.json` — 32,900 paper records, each with paper ID, title, topics, researcher IDs, researcher names, and both text variants.
- `data/researchers.json` — 1,810 researcher records, each with researcher ID and name.

---

## 5. Stage 3 — Embedding

**File:** [src/embed.py](src/embed.py)

### The Model: `all-MiniLM-L6-v2`

This is a lightweight transformer model from the [sentence-transformers](https://www.sbert.net/) library with 22 million parameters. It is specifically trained to produce **semantically meaningful sentence embeddings**: texts with similar meanings produce vectors that point in similar directions in the 384-dimensional embedding space.

The model was chosen for three reasons:

- **Small size:** 22M parameters vs 110M for full BERT — much faster inference
- **Strong semantic quality:** Trained on diverse sentence-pair tasks including paraphrase detection and natural language inference
- **Output dimension 384:** A practical sweet spot — large enough to capture semantic nuance, small enough for fast dot-product search

### Encoding Process

All 32,900 paper texts (both title-only and title+topics variants) are encoded in batches of 256 papers at a time. After encoding, every vector is **L2-normalized** to unit length (magnitude = 1).

This normalization step is critical because it allows cosine similarity to be computed as a simple dot product:

```
cosine_similarity(a, b) = (a · b) / (||a|| × ||b||) = a · b    when ||a|| = ||b|| = 1
```

At query time this means finding the most similar papers reduces to a single matrix–vector multiplication, which is fast and can be vectorized efficiently by NumPy.

### Output Files

| File                                     | Shape                 | Size  |
| ---------------------------------------- | --------------------- | ----- |
| `embeddings/embeddings_title_only.npy`   | (32900, 384) float32  | 48 MB |
| `embeddings/embeddings_title_topics.npy` | (32900, 384) float32  | 48 MB |
| `embeddings/paper_ids.json`              | list of 32900 strings | —     |

The `paper_ids.json` file maps each row index in the matrices to a Wikidata paper ID, allowing metadata to be looked up after retrieval.

**Timing on MacBook M-series (no GPU):** 30.8 seconds for 32,900 titles (~1,069 titles/second).

---

## 6. Stage 4 — Retrieval

**File:** [src/retrieve.py](src/retrieve.py)

### The `RetrievalIndex` Class

At startup, the `RetrievalIndex` class loads into memory:

- The 48 MB embedding matrix
- The paper ID row index
- The paper metadata dictionary (for title/topic/author lookup)
- The SentenceTransformer model (shared with the embedding stage)

This takes a few seconds on first load but keeps everything in RAM for fast subsequent queries.

### Query Processing

Given a user query string, the retrieval process runs the following steps:

1. **Encode the query:** The query string is passed through the same `all-MiniLM-L6-v2` model and L2-normalized, producing a vector of shape `(384,)`.

2. **Score all papers at once:**

   ```
   scores = embedding_matrix @ query_vector    # shape: (32900,)
   ```

   This single matrix multiplication computes the cosine similarity between the query and every paper in the index simultaneously.

3. **Find top-N efficiently:** Rather than sorting all 32,900 scores, `numpy.argpartition` is used to find the indices of the N highest scores. This is O(n) rather than O(n log n) — a meaningful speedup for the paper pool sizes used (50–500 papers).

4. **Sort and look up:** The top-N indices are sorted by descending score. For each index, metadata is retrieved and a `PaperMatch` object is constructed.

### Output: `PaperMatch`

Each match record contains:

- `paper_id`: Wikidata QID
- `title`: paper title
- `topics`: list of subject topics
- `researcher_ids` and `researcher_names`: all DTU co-authors
- `score`: cosine similarity score (typically in the range 0.4–0.95)

**Query latency:** approximately 38 ms on MacBook M-series, dominated by model encoding time rather than matrix multiplication.

---

## 7. Stage 5 — Aggregation

**File:** [src/aggregate.py](src/aggregate.py)

### The Problem

Retrieval returns paper-level results. A paper can have multiple DTU co-authors, and a researcher can have many papers. The aggregation step converts a list of paper matches into a ranked list of researchers.

### The Strategies

| Strategy    | Formula                                        | Effect                                                     |
| ----------- | ---------------------------------------------- | ---------------------------------------------------------- |
| `max`       | `researcher_score = best single paper score`   | Rewards the researcher with the single most relevant paper |
| `sum_top_3` | `researcher_score = sum of top 3 paper scores` | Rewards researchers with multiple relevant papers          |

### Which Strategy Wins and Why

**`max` achieves the best MRR (0.912) on the LLM benchmark.** The reason becomes clear when you consider what a student query like _"AI for renewable energy"_ actually signals: the student wants to find the expert whose work is closest to their interest. That is best captured by the single most relevant paper, not an accumulation of loosely related papers.

`sum_top_3` creates a **prolific-researcher bias**: a researcher with three papers scoring 0.70, 0.68, 0.65 would receive a total of 2.03 and outrank a researcher with one paper scoring 0.85. The latter researcher is clearly more relevant to this specific query.

On the harder `findit` benchmark, `sum_top_3` outperforms `max`. This is because the findit task involves matching a thesis title to a supervisor by topic area rather than by paper-to-paper similarity — in this case, broader topic coverage across multiple papers is a better signal than any single paper.

### Evidence Collection

For the top researchers, the aggregation step also collects:

- The titles of their best-matching publications (up to 5, deduplicated)
- The union of all topics across their matching papers (sorted alphabetically)

This gives the API response the evidence needed for a user to understand why a researcher was ranked highly.

---

## 8. Stage 6 — REST API

**File:** [src/api.py](src/api.py)

### Framework

The API is built with [FastAPI](https://fastapi.tiangolo.com/), an ASGI Python framework with automatic request validation via Pydantic and automatic OpenAPI documentation.

Both embedding indexes are loaded into memory once at startup via FastAPI's `lifespan` context manager. This means the 38 ms query latency is consistent from the first request; there is no cold-start cost per request.

### Endpoints

| Method | Path                           | Description                                                                        |
| ------ | ------------------------------ | ---------------------------------------------------------------------------------- |
| `GET`  | `/health`                      | Returns `{"status": "ok", "loaded_fields": [...]}` — shows which indexes are ready |
| `GET`  | `/version`                     | Returns the current API version                                                    |
| `POST` | `/v1/researcher-search`        | Main search endpoint                                                               |
| `POST` | `/v1/debug-publication-search` | Returns raw paper-level matches before aggregation                                 |

### Main Endpoint: `POST /v1/researcher-search`

**Request body:**

| Field             | Type                               | Default        | Description                         |
| ----------------- | ---------------------------------- | -------------- | ----------------------------------- |
| `query`           | string (≥3 chars)                  | required       | The user's natural language query   |
| `top_k`           | int (1–50)                         | 5              | Number of researchers to return     |
| `embedding_field` | `"title_only"` or `"title_topics"` | `"title_only"` | Which index to search               |
| `aggregation`     | `"max"` or `"sum_top_3"`           | `"sum_top_3"`  | How to score researchers            |
| `paper_pool`      | int (10–500)                       | 50             | Papers retrieved before aggregation |

**Example request:**

```json
POST /v1/researcher-search
{
  "query": "deep learning for wind turbine fault detection",
  "top_k": 5,
  "embedding_field": "title_only",
  "aggregation": "max"
}
```

**Example response:**

```json
{
  "query": "deep learning for wind turbine fault detection",
  "embedding_field": "title_only",
  "aggregation": "max",
  "top_k": 5,
  "results": [
    {
      "researcher_name": "Finn Gunnar Nielsen",
      "researcher_id": "Q47502345",
      "score": 0.8821,
      "matched_publications": [
        "Deep learning for condition monitoring of wind turbines",
        "Neural network approaches to offshore wind farm diagnostics"
      ],
      "matched_topics": ["deep learning", "wind energy", "condition monitoring"]
    }
  ]
}
```

### Debug Endpoint: `POST /v1/debug-publication-search`

Returns the raw paper-level matches before aggregation. Useful when a final researcher result looks unexpected: you can inspect which papers were retrieved and how they were scored, to determine whether the problem is in retrieval or in aggregation.

---

## 9. Evaluation

**Files:** [evaluation/evaluate.py](evaluation/evaluate.py), [evaluation/benchmark_llm.json](evaluation/benchmark_llm.json), [evaluation/benchmark_findit.json](evaluation/benchmark_findit.json)

### Metrics

Two standard information retrieval metrics are used:

**Hit@k:** The fraction of queries for which the correct researcher appears in the top-k results.

- Hit@1 = 0.857 means the correct researcher is ranked first in 85.7% of queries.
- Hit@5 = 0.993 means the correct researcher appears in the top 5 in 99.3% of queries.

**MRR (Mean Reciprocal Rank):** For each query, compute 1/rank where rank is the position of the first correct researcher; then average across all queries.

- MRR = 1.0: always ranks the correct researcher first.
- MRR = 0.5: the correct researcher is ranked second on average.

### Why Not Use Exact Paper Titles as Queries?

A naive evaluation — called **leave-one-out** — would use the actual paper title as the query. This is unrealistically easy: the query text would be nearly identical to the indexed text, inflating performance numbers. No real student types a verbatim paper title when searching for a supervisor.

### Benchmark 1 — LLM-Rewritten Queries (`evaluation/benchmark_llm.json`)

**Construction:** 300 papers were sampled randomly from the dataset. For each paper, the title was rewritten into a natural student query using an LLM (Gemma 4 via DTU's CampusAI API), with the following prompt:

> _"You are a master's student looking for a research supervisor at a technical university. Rewrite the following paper title as a short, natural student query (1-2 sentences). Do not copy the title verbatim. Use plain language, not academic jargon."_

Example rewrites:

| Original Paper Title                                                                    | LLM-Rewritten Student Query                                                                                                          |
| --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Metabolic characterization of high- and low-yielding strains of Penicillium chrysogenum | I'm interested in studying how the metabolism differs between high-performing and low-performing strains of Penicillium chrysogenum. |
| Unity Makes Strength: Exploring Intraspecies and Interspecies Toxin Synergism           | I'm interested in how different toxins work together to increase their overall toxicity.                                             |
| Byssochlamys: significance of heat resistance and mycotoxin production                  | I'm interested in studying how certain fungi survive high temperatures and produce toxins.                                           |

The benchmark file is committed to the repository (alongside the generation script) so results can be reproduced without re-calling the LLM API.

**Results:**

| Embedding field | Strategy   | MRR        | Hit@1      | Hit@3      | Hit@5      | Hit@10     |
| --------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| **title_only**  | **max**    | **0.9121** | **0.8567** | **0.9600** | **0.9933** | **0.9933** |
| title_topics    | max        | 0.9061     | 0.8533     | 0.9533     | 0.9800     | 0.9833     |
| title_only      | sum_top_3  | 0.8311     | 0.7667     | 0.8967     | 0.9167     | 0.9300     |
| title_topics    | sum_top_3  | 0.8217     | 0.7533     | 0.8867     | 0.9067     | 0.9333     |

**Best overall configuration: `title_only / max` with MRR = 0.912, Hit@1 = 85.7%, Hit@5 = 99.3%.**

This benchmark represents an **upper bound** on real-world performance because the LLM that rewrote the queries had access to the paper title and produced faithful paraphrases.

### Benchmark 2 — Findit Thesis Titles (`evaluation/benchmark_findit.json`)

**Construction:** Student thesis titles were downloaded from [findit.dtu.dk](https://findit.dtu.dk) for two supervisors:

- Lone Gram — 74 thesis titles
- Finn Årup Nielsen — 26 thesis titles

The task: given a thesis title as the query, does the system return the correct supervisor? This is a harder and more realistic benchmark because thesis titles are not derived from the supervisor's own publications — there is no guaranteed vocabulary overlap.

**Results:**

| Embedding field | Strategy      | MRR        | Hit@1      | Hit@3      | Hit@5      | Hit@10     |
| --------------- | ------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| title_only      | max           | 0.4833     | 0.3700     | 0.5700     | 0.6400     | 0.7100     |
| **title_only**  | **sum_top_3** | **0.5895** | **0.5100** | **0.6300** | **0.6600** | **0.8000** |
| title_topics    | max           | 0.4815     | 0.3600     | 0.5800     | 0.6300     | 0.7100     |
| title_topics    | sum_top_3     | 0.5608     | 0.4800     | 0.5800     | 0.6600     | 0.7900     |

**Interpretation:**

- Performance drops substantially compared to the LLM benchmark (MRR ~0.59 vs 0.91), confirming that the LLM benchmark is an optimistic upper bound.
- `sum_top_3` outperforms `max` here because the task requires matching broad research topics rather than specific paper content.
- `title_only` still outperforms `title_topics`, confirming that sparse Wikidata topic coverage adds noise rather than signal.

---

## 10. Key Design Decisions

### Why `title_only` beats `title_topics`?

Adding Wikidata topics to the embedded text does not improve retrieval for two reasons:

1. **Sparse coverage:** 46.1% of papers have zero topics. For those papers, both variants are identical, so `title_topics` can never outperform `title_only` on them. Since these papers make up nearly half the index, the potential gains from the other 53.9% must be large to overcome this structural disadvantage.

2. **Noisy topics:** When present, topics average only 1.9 per paper and use broad labels like _"applied mathematics"_ or _"bioengineering"_. These broad labels add little semantic precision beyond what the title already provides.

### Why `max` aggregation is better for the LLM benchmark?

The LLM benchmark queries are paraphrases of specific paper titles. The correct researcher is identified because they wrote that specific paper. Their relevance to the query is best captured by their single highest-scoring paper — not by an average or sum across multiple papers. `max` isolates this signal cleanly.

`sum_top_3` punishes the correct researcher when they have only one highly relevant paper but several moderately relevant ones, by allowing a more prolific researcher with many medium-relevance papers to outscore them.

### Why the findit benchmark flips the winner to `sum_top_3`?

Thesis titles describe a research area rather than a specific paper. A supervisor is identified not because they wrote that exact thesis, but because their research area overlaps with the thesis topic. Across many theses, the correct supervisor may not have a single paper that closely matches any given thesis title, but they will have many papers that are moderately relevant. Summing the top-3 scores captures this broader topic coverage.

### Why L2-normalize the embeddings?

Normalizing all vectors to unit length converts cosine similarity into a plain dot product. This allows the entire paper scoring step to be a single NumPy matrix multiplication (`scores = matrix @ query_vec`) instead of computing norms and dividing — faster and simpler.

### Why two separate SPARQL queries instead of one joined query?

A single SPARQL `JOIN` on topics would silently drop all papers that have no topic annotations (46.1% of the dataset). Separate queries allow the two result sets to be merged in application code, keeping all papers regardless of whether they have topics.

---

## 11. How to Run — Local Setup

### Prerequisites

- Python 3.11 or higher
- `pip`

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Build the index (run once)

These three steps must be completed before the API can start. They populate the `data/` and `embeddings/` directories, which are gitignored and not included in the repository.

```bash
# Fetch papers and researchers from Wikidata (~2 minutes)
python src/acquire.py

# Clean and structure the raw data (~5 seconds)
python src/preprocess.py

# Generate embedding matrices (~31 seconds on MacBook M-series)
python src/embed.py
```

### Step 3 — Start the API

```bash
uvicorn src.api:app --reload
```

The API is available at `http://localhost:8000`.  
Interactive Swagger documentation is at `http://localhost:8000/docs`.

### Step 4 — Run evaluation

```bash
# LLM benchmark — all embedding × strategy combinations
python evaluation/evaluate.py --benchmark evaluation/benchmark_llm.json --compare-all

# Findit benchmark — all combinations
python evaluation/evaluate.py --benchmark evaluation/benchmark_findit.json --compare-all

# Single configuration
python evaluation/evaluate.py --benchmark evaluation/benchmark_llm.json --field title_only --strategy max
```

### Step 5 — Run tests

```bash
pytest tests/
```

All tests are fully synthetic — no network calls and no embedding files required.

### Environment file (only needed for benchmark regeneration)

The `.env` file is only required if you want to regenerate the LLM benchmark by calling the CampusAI API. Create a `.env` file in the project root with your key:

```
CAMPUSAI_API_KEY=your-api-key-here
CAMPUSAI_MODEL="Gemma 4"
```

To regenerate the LLM benchmark:

```bash
python evaluation/generate_llm_benchmark.py
```

---

## 12. How to Run — Docker

Docker is the recommended way to run the API in a reproducible, isolated environment. The Docker image contains only the source code and dependencies; the data and embeddings are mounted as volumes from your local machine.

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- The `data/` and `embeddings/` directories must already exist locally (generated by running the pipeline in [Step 2](#step-2--build-the-index-run-once) above)

### Option A — Docker Compose (recommended)

```bash
# Build the image and start the container
docker compose up --build

# To stop
docker compose down
```

The API is available at `http://localhost:8000`.  
The `data/` and `embeddings/` directories from your local machine are mounted into the container, so no rebuild is needed if you refresh the data.

### Option B — Docker without Compose

If you prefer to run without `docker compose`:

```bash
# Build the image
docker build -t dtu-researcher-finder .

# Run the container, mounting local data and embeddings
docker run -p 8000:8000 \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/embeddings:/app/embeddings" \
  dtu-researcher-finder
```

### Verify the API is running

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{ "status": "ok", "loaded_fields": ["title_only", "title_topics"] }
```

### Make a search request

```bash
curl -X POST http://localhost:8000/v1/researcher-search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning for wind turbine fault detection",
    "top_k": 5,
    "embedding_field": "title_only",
    "aggregation": "max"
  }'
```

### Notes on Docker image size

The Docker image includes only the Python dependencies and source code (~600 MB for sentence-transformers + PyTorch CPU). The embedding matrices (96 MB total) and data files are kept outside the image and mounted as volumes, so rebuilding the image does not require regenerating the index.

---

## 13. Project Structure

```
project/
├── src/
│   ├── acquire.py           — SPARQL queries to fetch DTU papers and researchers
│   ├── preprocess.py        — Raw data cleaning and structuring
│   ├── embed.py             — all-MiniLM-L6-v2 encoding and L2 normalization
│   ├── retrieve.py          — RetrievalIndex class (matrix dot product search)
│   ├── aggregate.py         — Paper-level → researcher-level scoring
│   └── api.py               — FastAPI REST application
│
├── data/                    — (gitignored; generated by pipeline)
│   ├── raw_sparql.json      — Raw SPARQL output
│   ├── papers.json          — 32,900 cleaned paper records
│   └── researchers.json     — 1,810 researcher records
│
├── embeddings/              — (gitignored; generated by embed.py)
│   ├── embeddings_title_only.npy    — (32900 × 384) float32
│   ├── embeddings_title_topics.npy  — (32900 × 384) float32
│   └── paper_ids.json               — Row-to-paper-ID mapping
│
├── evaluation/
│   ├── evaluate.py                    — Hit@k / MRR evaluation script
│   ├── generate_llm_benchmark.py      — LLM query rewriting script (uses CampusAI)
│   ├── generate_findit_benchmark.py   — Findit thesis benchmark builder
│   ├── benchmark.json                 — 6 hand-curated queries (sanity check)
│   ├── benchmark_llm.json             — 300 LLM-rewritten queries
│   ├── benchmark_findit.json          — 100 thesis title queries
│   ├── supervisors.json               — Supervisor + thesis title source data
│   └── EVALUATION_APPROACH.md         — Evaluation methodology documentation
│
├── tests/
│   ├── test_retrieve.py    — Retrieval unit tests (synthetic embeddings)
│   └── test_aggregate.py   — Aggregation strategy unit tests
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 14. Operational Characteristics

Measured on MacBook M-series CPU (no GPU):

| Operation                              | Measurement                    |
| -------------------------------------- | ------------------------------ |
| Fetch data from Wikidata via SPARQL    | ~2 minutes                     |
| Preprocess raw data                    | ~5 seconds                     |
| Embed 32,900 titles (batch_size=256)   | 30.8 seconds (~1,069 titles/s) |
| Embedding matrix size (384-d, float32) | 48 MB per variant              |
| Query latency (encode + dot product)   | ~38 ms                         |
| Runtime memory (both indexes + model)  | ~200–300 MB                    |

---

## Summary

| Component            | Choice                                      | Why                                                              |
| -------------------- | ------------------------------------------- | ---------------------------------------------------------------- |
| Data source          | Wikidata via SPARQL                         | Structured, open, machine-readable, covers DTU affiliations      |
| SPARQL endpoint      | QLever                                      | High-performance; handles full Wikidata graph                    |
| Embedding model      | `all-MiniLM-L6-v2`                          | Lightweight (22M params), strong semantic quality, 384-d output  |
| Text representation  | Title only                                  | Outperforms title+topics — sparse Wikidata topics add noise      |
| Aggregation strategy | `max` (LLM benchmark), `sum_top_3` (findit) | Task-dependent: single best paper vs. broad topic coverage       |
| Similarity metric    | Cosine (via L2-normalized dot product)      | Measures semantic direction, length-invariant, fast at scale     |
| Evaluation method    | LLM-rewritten queries + thesis titles       | More realistic than leave-one-out; cheaper than human annotation |
| API framework        | FastAPI                                     | Async, fast, automatic validation and OpenAPI docs               |
| Deployment           | Docker + volume mounts                      | Keeps image small; data updated without rebuild                  |
