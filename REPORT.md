# DTU Researcher Finder — Project Report

## 1. Problem Statement

Finding the right research supervisor or collaborator at a large technical university is a difficult information-retrieval problem. A student interested in, say, *"machine learning for wind energy"* cannot easily search DTU's publication database with that phrase and get back a ranked list of relevant researchers.

This project builds a **semantic search system** that solves exactly this problem. Given a free-text natural language query, the system returns a ranked list of DTU researchers whose published work best matches the **meaning** of the query — not just the keywords.

The final product is a REST API that accepts a query and returns researchers with their scores and the specific publications that justify their ranking.

---

## 2. System Architecture

The system is built as a five-stage offline pipeline followed by a real-time query engine:

```
Wikidata (SPARQL)
       │
       ▼
  1. Data Acquisition     acquire.py
       │
       ▼
  2. Preprocessing        preprocess.py
       │
       ▼
  3. Embedding            embed.py
       │
       ▼
  4. Retrieval Index      retrieve.py      ◄── query at runtime
       │
       ▼
  5. Aggregation          aggregate.py     ◄── query at runtime
       │
       ▼
  6. REST API             api.py           ◄── exposed to the user
```

Stages 1–3 are run once offline to build the index. Stages 4–6 serve live queries.

---

## 3. Data Acquisition

**Source:** [Wikidata](https://www.wikidata.org), queried via the [QLever SPARQL endpoint](https://qlever.cs.uni-freiburg.de/api/wikidata).

Wikidata is a free, structured knowledge base that contains information about researchers, their institutional affiliations, and their publications. DTU (Technical University of Denmark) has the Wikidata entity ID `Q1269766`.

Two SPARQL queries are executed:

**Query 1 — Paper titles:** Finds all publications authored by any researcher currently affiliated with DTU. Affiliation is matched via two Wikidata properties: `P108` (employer) and `P1416` (affiliation). Past affiliations — those with an end date (`P582`) — are excluded so only current employees are included.

**Query 2 — Paper topics:** Fetches the `P921` (main subject) topic labels for those same papers.

The two queries are kept separate because topic coverage in Wikidata is sparse — merging them at the SPARQL level would silently drop all papers that have no topics assigned.

**Result:** The raw data is saved to `data/raw_sparql.json`.

### Dataset statistics

| Metric | Value |
|---|---|
| Unique researchers | 1,810 |
| Unique papers | 32,900 |
| Papers with topics | 17,733 (53.9%) |
| Average topics per paper (when present) | 1.9 |

---

## 4. Preprocessing

The raw SPARQL results are cleaned and structured into two output files:

- **`data/papers.json`** — one record per unique paper, containing the paper ID, title, topic list, and the IDs and names of all DTU-affiliated co-authors.
- **`data/researchers.json`** — one record per researcher with their Wikidata ID and name.

A key design decision here is preparing **two different text representations** for each paper:

| Variant | Text format | Purpose |
|---|---|---|
| `text_title_only` | The raw paper title | Clean signal, no noise from sparse topics |
| `text_title_topics` | `"Title: <title> Topics: <t1>, <t2>, ..."` | Richer representation when topics are available |

Both variants are stored so that downstream steps can compare which one produces better retrieval quality.

---

## 5. Embedding

**Model:** `all-MiniLM-L6-v2` from the [sentence-transformers](https://www.sbert.net/) library.

This is a lightweight transformer model (22M parameters) specifically trained to produce semantically meaningful sentence embeddings — that is, texts with similar meanings are mapped to nearby points in vector space.

All 32,900 paper texts are encoded into dense vectors. The vectors are **L2-normalized** (scaled to unit length) before saving. This is important because it means similarity between two vectors can be computed as a simple **dot product**, which is mathematically equivalent to cosine similarity but much faster at query time.

The resulting embedding matrices are saved as NumPy arrays:

| File | Shape | Description |
|---|---|---|
| `embeddings_title_only.npy` | (32900, 384) | One 384-dimensional vector per paper, title only |
| `embeddings_title_topics.npy` | (32900, 384) | One 384-dimensional vector per paper, title + topics |
| `paper_ids.json` | list of 32900 IDs | Row index shared by both matrices |

The embedding dimension is 384 because that is the output size of `all-MiniLM-L6-v2`.

---

## 6. Retrieval

At query time, the `RetrievalIndex` class handles the following steps:

1. The user's query string is embedded by the same model into a 384-dimensional unit vector.
2. A single matrix–vector multiplication scores all 32,900 papers simultaneously:

```
scores = embedding_matrix @ query_vector     # shape: (32900,)
```

3. The top-N highest-scoring papers are returned using `numpy.argpartition`, which finds the best N results without sorting the full array (faster for large N).

Each result is a `PaperMatch` object containing the paper title, topics, author IDs, author names, and similarity score.

**Why cosine similarity?** Cosine similarity measures the angle between two vectors, ignoring their magnitude. This means it compares the *direction* of meaning rather than the intensity, which is the right measure for semantic similarity between texts of different lengths.

---

## 7. Aggregation

A single paper can have multiple co-authors. The retrieval step returns paper-level results, but the user wants researcher-level results. The aggregation step collapses paper matches into researcher scores.

Each researcher's score is computed from their top-matching papers using one of two strategies:

| Strategy | Formula | Intuition |
|---|---|---|
| `max` | `score = best single paper score` | The researcher's most relevant work defines their relevance |
| `sum_top_3` | `score = sum of top 3 paper scores` | Rewards researchers with multiple strong matches |

The output is a ranked list of `ResearcherResult` objects, each with the researcher's name, ID, final score, and the matching publications and topics used as evidence.

---

## 8. API

The system is exposed as a REST API built with [FastAPI](https://fastapi.tiangolo.com/). Both embedding indexes are loaded into memory once at startup and kept resident so that individual queries do not hit the disk.

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Returns which indexes are loaded |
| `GET` | `/version` | Returns API version |
| `POST` | `/v1/researcher-search` | Main search endpoint |
| `POST` | `/v1/debug-publication-search` | Returns raw paper-level matches for debugging |

### Example request

```json
POST /v1/researcher-search
{
  "query": "deep learning for wind turbine fault detection",
  "top_k": 5,
  "embedding_field": "title_only",
  "aggregation": "max"
}
```

### Example response

```json
{
  "query": "deep learning for wind turbine fault detection",
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

---

## 9. Evaluation

### 9.1 Metrics

The system is evaluated using two standard information retrieval metrics:

- **Hit@k:** The fraction of queries for which the correct researcher appears in the top-k results. A Hit@5 of 0.95 means 95% of queries are answered correctly when you look at 5 results.
- **MRR (Mean Reciprocal Rank):** The average of 1/rank, where rank is the position of the first correct researcher. A perfect system scores MRR=1.0 (always ranks the right person first). MRR=0.5 would mean the correct researcher is ranked second on average.

### 9.2 Benchmark construction

A naive approach to evaluation — called **leave-one-out** — would use the exact paper title as the query. This is unrealistic: no student types a verbatim paper title when searching for a supervisor. It inflates performance because the query is nearly identical to the indexed text.

Instead, this project uses an **LLM-rewriting** approach:

1. **Sample** 300 papers randomly from the dataset.
2. **Rewrite** each paper title as a realistic student query using an LLM (Gemma 4 via the DTU CampusAI API), prompted as follows:

   > *"You are a master's student looking for a research supervisor at a technical university. Rewrite the following paper title as a short, natural student query (1-2 sentences). Do not copy the title verbatim. Use plain language, not academic jargon."*

3. **Store** the rewritten query alongside the original paper's author IDs as ground truth.

This produces queries like:

| Original paper title | LLM-rewritten student query |
|---|---|
| Metabolic characterization of high- and low-yielding strains of Penicillium chrysogenum | I'm interested in studying how the metabolism differs between high-performing and low-performing strains of Penicillium chrysogenum. |
| Unity Makes Strength: Exploring Intraspecies and Interspecies Toxin Synergism between Phospholipases A2 and Cytotoxins | I'm interested in how different toxins work together to increase their overall toxicity. Do you have openings for a student to research these synergistic effects? |
| Byssochlamys: significance of heat resistance and mycotoxin production | I'm interested in studying how certain fungi survive high temperatures and produce toxins. |

The benchmark file (`evaluation/benchmark_llm.json`, 300 entries) is committed to the repository alongside the generation script, so results are reproducible without calling the LLM API again.

A small hand-curated benchmark of 6 queries (`evaluation/benchmark.json`) was also created as a sanity check during development.

### 9.3 Results

The evaluation was run across all combinations of embedding variant (`title_only`, `title_topics`) and aggregation strategy (`max`, `sum_top_3`) on the 300-query LLM benchmark:

| Configuration | MRR | Hit@1 | Hit@3 | Hit@5 | Hit@10 |
|---|---|---|---|---|---|
| **title_only / max** | **0.9121** | **0.8567** | **0.9600** | **0.9933** | **0.9933** |
| title_topics / max | 0.9061 | 0.8533 | 0.9533 | 0.9800 | 0.9833 |
| title_only / sum_top_3 | 0.8311 | 0.7667 | 0.8967 | 0.9167 | 0.9300 |
| title_topics / sum_top_3 | 0.8217 | 0.7533 | 0.8867 | 0.9067 | 0.9333 |

**Best configuration: `title_only / max`**

---

## 10. Analysis of Results

### The `max` strategy outperforms `sum` and `mean`

`sum_top_3` accumulates the scores of a researcher's top 3 papers. This creates a bias toward **prolific researchers**: someone with three moderately relevant papers scores higher than someone with one highly relevant paper. In practice, for topic-based queries, the single most relevant paper is the strongest signal of expertise, so `max` is the right choice.

### `title_only` marginally outperforms `title_topics`

Adding Wikidata topics to the embedded text does not improve retrieval, for two reasons:

1. **Sparse coverage:** Only 53.9% of papers have any topics at all. For the remaining 46.1%, the two variants are identical, so `title_topics` cannot gain on them.
2. **Noise:** When topics are present, they average only 1.9 per paper, and the labels (e.g. *"applied mathematics"*) are often broader than the paper title itself. The title already carries more precise semantic content.

### Overall performance is strong

With the best configuration, the system achieves:

- **Hit@1 = 85.7%:** The correct researcher is ranked first in 85.7% of queries.
- **Hit@5 = 99.3%:** The correct researcher is found in the top 5 in virtually every query.
- **MRR = 0.912:** The correct researcher appears at rank ~1.1 on average.

These results are on LLM-rewritten queries that paraphrase paper titles into student language — a meaningfully harder task than matching exact titles.

### Caveat: optimism bias in the benchmark

The LLM that generated the queries had access to the paper title and rewrote it faithfully. A real student with only a vague research interest would produce queries that are harder to match. The reported numbers should therefore be interpreted as an upper bound on real-world performance.

---

## 11. Summary

| Component | Choice | Rationale |
|---|---|---|
| Data source | Wikidata via SPARQL | Structured, open, covers DTU affiliations |
| Embedding model | all-MiniLM-L6-v2 | Fast, small, strong semantic quality |
| Text representation | Title only | Outperforms title+topics due to sparse Wikidata coverage |
| Aggregation strategy | Max | Best single paper score avoids prolific-researcher bias |
| Evaluation | LLM-rewritten queries | More realistic than leave-one-out, cheaper than human annotation |
