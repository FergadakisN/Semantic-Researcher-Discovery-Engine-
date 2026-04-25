"""
FastAPI service exposing the researcher search pipeline.

Start:
    uvicorn src.api:app --reload
or via Docker.

Endpoints:
    GET  /health
    GET  /version
    POST /v1/researcher-search
    POST /v1/debug-publication-search
"""

import sys
from pathlib import Path

# Allow running from project root: python src/api.py
sys.path.insert(0, str(Path(__file__).parent))

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from aggregate import aggregate, ResearcherResult
from retrieve import RetrievalIndex

VERSION = "0.1.0"

# Supported embedding fields
EmbeddingField = Literal["title_only", "title_topics"]
AggStrategy = Literal["max", "sum_top_3", "mean_top_3"]

# Global indexes (loaded once at startup)
_indexes: dict[str, RetrievalIndex] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load retrieval indexes once at startup so requests do not repeatedly hit
    # disk or recreate the embedding model.
    for field in ("title_only", "title_topics"):
        emb_path = Path("embeddings") / f"embeddings_{field}.npy"
        if emb_path.exists():
            print(f"Loading index: {field} ...")
            _indexes[field] = RetrievalIndex(field=field)
        else:
            print(f"Warning: embedding file not found for field '{field}', skipping.")
    if not _indexes:
        raise RuntimeError(
            "No embedding files found. Run src/embed.py first."
        )
    yield
    _indexes.clear()


app = FastAPI(
    title="DTU Researcher Finder",
    version=VERSION,
    lifespan=lifespan,
)


# --- request / response models ---

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Research query text")
    top_k: int = Field(5, ge=1, le=50, description="Number of researchers to return")
    embedding_field: EmbeddingField = Field(
        "title_only", description="Which embedding variant to use"
    )
    aggregation: AggStrategy = Field(
        "sum_top_3", description="Researcher scoring strategy"
    )
    paper_pool: int = Field(
        50, ge=10, le=500,
        description="How many top papers to retrieve before aggregation"
    )


class ResearcherOut(BaseModel):
    researcher_name: str
    researcher_id: str
    score: float
    matched_publications: list[str]
    matched_topics: list[str]


class SearchResponse(BaseModel):
    query: str
    embedding_field: str
    aggregation: str
    top_k: int
    results: list[ResearcherOut]


class PaperOut(BaseModel):
    paper_id: str
    title: str
    topics: list[str]
    researcher_names: list[str]
    score: float


class PaperSearchResponse(BaseModel):
    query: str
    top_n: int
    results: list[PaperOut]


# --- endpoints ---

@app.get("/health")
def health():
    return {"status": "ok", "loaded_fields": list(_indexes.keys())}


@app.get("/version")
def version():
    return {"version": VERSION}


@app.post("/v1/researcher-search", response_model=SearchResponse)
def researcher_search(req: SearchRequest):
    index = _indexes.get(req.embedding_field)
    if index is None:
        raise HTTPException(
            status_code=400,
            detail=f"Embedding field '{req.embedding_field}' is not loaded.",
        )

    # First retrieve relevant papers, then collapse those paper-level matches
    # into a researcher-level ranking.
    paper_matches = index.query(req.query, top_n=req.paper_pool)
    ranked = aggregate(
        paper_matches,
        strategy=req.aggregation,
        top_k_researchers=req.top_k,
    )

    return SearchResponse(
        query=req.query,
        embedding_field=req.embedding_field,
        aggregation=req.aggregation,
        top_k=req.top_k,
        results=[
            ResearcherOut(
                researcher_name=r.researcher_name,
                researcher_id=r.researcher_id,
                score=round(r.score, 4),
                matched_publications=r.matched_publications,
                matched_topics=r.matched_topics,
            )
            for r in ranked
        ],
    )


@app.post("/v1/debug-publication-search", response_model=PaperSearchResponse)
def debug_publication_search(
    query: str,
    top_n: int = 10,
    embedding_field: EmbeddingField = "title_only",
):
    # This endpoint exposes the paper retrieval stage directly, which is useful
    # for checking whether bad final results come from retrieval or aggregation.
    index = _indexes.get(embedding_field)
    if index is None:
        raise HTTPException(
            status_code=400,
            detail=f"Embedding field '{embedding_field}' is not loaded.",
        )

    matches = index.query(query, top_n=top_n)
    return PaperSearchResponse(
        query=query,
        top_n=top_n,
        results=[
            PaperOut(
                paper_id=m.paper_id,
                title=m.title,
                topics=m.topics,
                researcher_names=m.researcher_names,
                score=round(m.score, 4),
            )
            for m in matches
        ],
    )
