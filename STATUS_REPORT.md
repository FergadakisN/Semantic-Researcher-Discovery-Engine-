# Project Status Report — Find DTU Researcher Given a Text

## Overview

I have implemented a working end-to-end prototype of the system. Given a research query (e.g. a project title or abstract), the system returns a ranked list of relevant DTU researchers based on the semantic similarity of their publications.

---

## What has been implemented

### 1. Data acquisition

I wrote a script that queries the QLever Wikidata SPARQL endpoint to collect publications authored by DTU researchers (DTU = `wd:Q1269766`). Two queries are issued:

- one to retrieve paper titles and researcher names/IDs,
- one to retrieve paper topics.

The data is saved locally so the system does not depend on live SPARQL queries at runtime.

### 2. Preprocessing

The raw data is cleaned and structured into a dataset of unique papers. Each paper record contains its title, associated topics, and the list of researchers who authored it. Papers with missing or very short titles are discarded. Duplicate paper-author records are handled correctly.

Two text representations are built per paper for comparison:

- **Title only** — just the publication title
- **Title + Topics** — `"Title: ... Topics: ..."` concatenated string

### 3. Embedding

All unique papers are embedded using **sentence-transformers** (`all-MiniLM-L6-v2`), producing a matrix of dense vector representations stored on disk. Embeddings are L2-normalized so that similarity can be computed as a dot product (equivalent to cosine similarity).

### 4. Retrieval

At query time, the input text is embedded with the same model and compared against all paper embeddings using cosine similarity. The top-N most similar papers are retrieved.

### 5. Researcher-level aggregation

Since the search is over papers, the paper-level matches are aggregated into a researcher-level ranking. Three strategies are implemented for comparison:

- **max** — researcher score = highest single paper score
- **sum_top_3** — researcher score = sum of top 3 paper scores
- **mean_top_3** — researcher score = mean of top 3 paper scores

### 6. API

The system is exposed through a FastAPI web service with the following endpoints:

- `POST /v1/researcher-search` — main endpoint, returns ranked researchers with matched publications and topics
- `POST /v1/debug-publication-search` — returns raw paper-level matches for inspection
- `GET /health` and `GET /version`

### 7. Evaluation framework

I set up an evaluation script that measures **Hit@1**, **Hit@3**, **Hit@5**, **Hit@10**, and **MRR** over a benchmark of queries with known expected researchers. The script supports a `--compare-all` flag that runs all combinations of text representation × aggregation strategy and prints a comparison table. I have started populating the benchmark with 6 queries and verified the results manually.

### 8. Tests and packaging

Unit tests are written for the aggregation and retrieval components. The system is containerized with Docker for reproducibility.

---

## Example output

Query: `"deep learning for medical imaging"`

| Rank | Researcher           | Score | Top matched publication                                                             |
| ---- | -------------------- | ----- | ----------------------------------------------------------------------------------- |
| 1    | Ole Winther          | 1.52  | Deep Learning for Diagnostic Binary Classification of Multiple-Lesion Skin Diseases |
| 2    | Tim B. Dyrby         | 1.37  | Accelerated Microstructure Imaging via Convex Optimization from diffusion MRI data  |
| 3    | Koen Van Leemput     | 1.34  | A Contrast Augmentation Approach to Improve Multi-Scanner Generalization in MRI     |
| 4    | Aasa Feragen         | 1.32  | Shortcut Learning in Medical Image Segmentation                                     |
| 5    | Jørgen Arendt Jensen | 1.30  | Medical ultrasound imaging                                                          |

---

## What remains

- Expand the evaluation benchmark with more queries
- Run the full `--compare-all` evaluation and report which text representation and aggregation strategy performs best
- Write the final report
- Optionally: swap the embedding model for CampusAI embeddings and compare results
