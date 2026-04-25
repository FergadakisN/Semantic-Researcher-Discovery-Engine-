"""
Retrieval index: loads the embedding matrix once and answers queries.

The index is built over unique papers (normalized vectors), so similarity
is computed as a dot product (== cosine similarity after L2-normalization).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

EMB_DIR = Path(__file__).parent.parent / "embeddings"
DATA_DIR = Path(__file__).parent.parent / "data"

DEFAULT_MODEL = "all-MiniLM-L6-v2"


@dataclass
class PaperMatch:
    paper_id: str
    title: str
    topics: list[str]
    researcher_ids: list[str]
    researcher_names: list[str]
    score: float


class RetrievalIndex:
    def __init__(
        self,
        field: str = "title_only",
        model_name: str = DEFAULT_MODEL,
        emb_dir: Path = EMB_DIR,
        data_dir: Path = DATA_DIR,
    ):
        emb_path = emb_dir / f"embeddings_{field}.npy"
        ids_path = emb_dir / "paper_ids.json"
        papers_path = data_dir / "papers.json"

        # The matrix and paper_ids.json must stay in the same order; otherwise
        # retrieved rows would point to the wrong paper metadata.
        self._matrix = np.load(emb_path)  # [n_papers, dim], float32, unit vectors
        self._paper_ids: list[str] = json.loads(ids_path.read_text())

        papers_raw = json.loads(papers_path.read_text())
        self._paper_lookup: dict[str, dict] = {p["paper_id"]: p for p in papers_raw}

        self._model = SentenceTransformer(model_name)
        self._field = field

    def query(self, text: str, top_n: int = 50) -> list[PaperMatch]:
        """Return the top_n most similar papers to the input text."""
        q_vec = self._model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)  # [1, dim]

        # Because both query and paper vectors are normalized, this dot product
        # is equivalent to cosine similarity.
        scores = (self._matrix @ q_vec.T).squeeze()  # [n_papers]
        # argpartition is faster than sorting the full array when we only need
        # the best-scoring subset.
        top_indices = np.argpartition(scores, -top_n)[-top_n:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = []
        for idx in top_indices:
            pid = self._paper_ids[idx]
            paper = self._paper_lookup.get(pid, {})
            results.append(
                PaperMatch(
                    paper_id=pid,
                    title=paper.get("title", ""),
                    topics=paper.get("topics", []),
                    researcher_ids=paper.get("researcher_ids", []),
                    researcher_names=paper.get("researcher_names", []),
                    score=float(scores[idx]),
                )
            )
        return results
