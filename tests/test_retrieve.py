"""
Unit tests for the retrieval layer.
These use synthetic in-memory data — no real embedding files required.
"""

import sys
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class _FakeModel:
    """Returns a deterministic vector based on the hash of the text."""

    DIM = 8

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True, **_):
        vecs = []
        for t in texts:
            seed = abs(hash(t)) % (2**31)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self.DIM).astype(np.float32)
            if normalize_embeddings:
                v /= np.linalg.norm(v) + 1e-9
            vecs.append(v)
        return np.stack(vecs)


def _build_temp_index(tmp_path: Path):
    """Create a minimal synthetic dataset + embedding matrix on disk."""
    papers = [
        {
            "paper_id": "Q1",
            "title": "Deep learning for image segmentation",
            "topics": ["deep learning", "computer vision"],
            "text_title_only": "Deep learning for image segmentation",
            "text_title_topics": "Title: Deep learning for image segmentation Topics: deep learning, computer vision",
            "researcher_ids": ["R1"],
            "researcher_names": ["Alice"],
        },
        {
            "paper_id": "Q2",
            "title": "Natural language processing with transformers",
            "topics": ["NLP", "transformers"],
            "text_title_only": "Natural language processing with transformers",
            "text_title_topics": "Title: Natural language processing with transformers Topics: NLP, transformers",
            "researcher_ids": ["R2"],
            "researcher_names": ["Bob"],
        },
        {
            "paper_id": "Q3",
            "title": "Reinforcement learning for robotics",
            "topics": ["reinforcement learning", "robotics"],
            "text_title_only": "Reinforcement learning for robotics",
            "text_title_topics": "Title: Reinforcement learning for robotics Topics: reinforcement learning, robotics",
            "researcher_ids": ["R1", "R3"],
            "researcher_names": ["Alice", "Carol"],
        },
    ]

    model = _FakeModel()
    texts = [p["text_title_only"] for p in papers]
    matrix = model.encode(texts, normalize_embeddings=True)

    data_dir = tmp_path / "data"
    emb_dir = tmp_path / "embeddings"
    data_dir.mkdir()
    emb_dir.mkdir()

    (data_dir / "papers.json").write_text(json.dumps(papers))
    (emb_dir / "paper_ids.json").write_text(json.dumps([p["paper_id"] for p in papers]))
    np.save(emb_dir / "embeddings_title_only.npy", matrix.astype(np.float32))

    return data_dir, emb_dir


def test_query_returns_correct_count(tmp_path, monkeypatch):
    from retrieve import RetrievalIndex
    from sentence_transformers import SentenceTransformer

    data_dir, emb_dir = _build_temp_index(tmp_path)
    monkeypatch.setattr(SentenceTransformer, "__init__", lambda self, *a, **kw: None)
    monkeypatch.setattr(SentenceTransformer, "encode", lambda self, *a, **kw: _FakeModel().encode(*a, **kw))

    idx = RetrievalIndex.__new__(RetrievalIndex)
    idx._matrix = np.load(emb_dir / "embeddings_title_only.npy")
    idx._paper_ids = json.loads((emb_dir / "paper_ids.json").read_text())
    idx._paper_lookup = {p["paper_id"]: p for p in json.loads((data_dir / "papers.json").read_text())}
    idx._model = _FakeModel()
    idx._field = "title_only"

    results = idx.query("deep learning image", top_n=2)
    assert len(results) == 2


def test_query_scores_are_bounded(tmp_path, monkeypatch):
    from retrieve import RetrievalIndex

    data_dir, emb_dir = _build_temp_index(tmp_path)

    idx = RetrievalIndex.__new__(RetrievalIndex)
    idx._matrix = np.load(emb_dir / "embeddings_title_only.npy")
    idx._paper_ids = json.loads((emb_dir / "paper_ids.json").read_text())
    idx._paper_lookup = {p["paper_id"]: p for p in json.loads((data_dir / "papers.json").read_text())}
    idx._model = _FakeModel()
    idx._field = "title_only"

    results = idx.query("some query text", top_n=3)
    for r in results:
        assert -1.01 <= r.score <= 1.01


def test_query_results_sorted_descending(tmp_path):
    from retrieve import RetrievalIndex

    data_dir, emb_dir = _build_temp_index(tmp_path)

    idx = RetrievalIndex.__new__(RetrievalIndex)
    idx._matrix = np.load(emb_dir / "embeddings_title_only.npy")
    idx._paper_ids = json.loads((emb_dir / "paper_ids.json").read_text())
    idx._paper_lookup = {p["paper_id"]: p for p in json.loads((data_dir / "papers.json").read_text())}
    idx._model = _FakeModel()
    idx._field = "title_only"

    results = idx.query("natural language transformers", top_n=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
