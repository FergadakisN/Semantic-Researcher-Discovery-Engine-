# Project Progress

## What we built

A semantic retrieval system that takes a research query and returns ranked DTU researchers based on the similarity of their publications.

## Pipeline

```
QLever (Wikidata SPARQL)
        ↓
src/acquire.py        → data/raw_sparql.json
        ↓
src/preprocess.py     → data/papers.json, data/researchers.json
        ↓
src/embed.py          → embeddings/embeddings_title_only.npy
                         embeddings/embeddings_title_topics.npy
        ↓
src/retrieve.py       cosine similarity search over the embedding matrix
        ↓
src/aggregate.py      paper matches → researcher-level ranking
        ↓
src/api.py            FastAPI service
```

## Key design decisions

- **Unique papers are embedded**, not paper-author pairs — avoids duplicates in the index
- **Two text variants** are compared: `title_only` vs `title + topics`
- **Three aggregation strategies** are compared: `max`, `sum_top_3`, `mean_top_3`
- Scores are cosine similarities (0–1 per paper); researcher score under `sum_top_3` can exceed 1

## API

```
POST /v1/researcher-search         main endpoint
POST /v1/debug-publication-search  inspect paper-level scores
GET  /health
GET  /version
```

Interactive docs at `http://localhost:8000/docs`.

## Evaluation

Benchmark: `evaluation/benchmark.json` — 6 queries with known expected researcher QIDs.

Run full comparison:
```bash
python evaluation/evaluate.py --compare-all
```

Metrics: Hit@1, Hit@3, Hit@5, Hit@10, MRR.

## Tests

```bash
pytest tests/
```

7 unit tests covering aggregation logic and retrieval output properties.

## How to run

```bash
pip install -r requirements.txt
python src/acquire.py         # fetch data from Wikidata
python src/preprocess.py      # clean and structure
python src/embed.py           # embed papers
uvicorn src.api:app --reload  # start API
```

## What's next

- Add more benchmark queries to `evaluation/benchmark.json`
- Run `evaluate.py --compare-all` and compare strategies
- Optionally: swap sentence-transformers for CampusAI embeddings
- Write the project report
