import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aggregate import aggregate, ResearcherResult
from retrieve import PaperMatch


def _make_match(paper_id, title, researcher_ids, researcher_names, score, topics=None):
    return PaperMatch(
        paper_id=paper_id,
        title=title,
        topics=topics or [],
        researcher_ids=researcher_ids,
        researcher_names=researcher_names,
        score=score,
    )


def test_max_strategy():
    matches = [
        _make_match("P1", "Paper A", ["R1"], ["Alice"], 0.9),
        _make_match("P2", "Paper B", ["R1"], ["Alice"], 0.5),
        _make_match("P3", "Paper C", ["R2"], ["Bob"],   0.8),
    ]
    results = aggregate(matches, strategy="max", top_k_researchers=5)
    assert results[0].researcher_id == "R1"
    assert abs(results[0].score - 0.9) < 1e-6
    assert results[1].researcher_id == "R2"


def test_sum_top_3():
    matches = [
        _make_match("P1", "Paper A", ["R1"], ["Alice"], 0.9),
        _make_match("P2", "Paper B", ["R1"], ["Alice"], 0.5),
        _make_match("P3", "Paper C", ["R1"], ["Alice"], 0.4),
        _make_match("P4", "Paper D", ["R2"], ["Bob"],   0.8),
    ]
    results = aggregate(matches, strategy="sum_top_3", top_k_researchers=5)
    # R1: 0.9 + 0.5 + 0.4 = 1.8; R2: 0.8
    assert results[0].researcher_id == "R1"
    assert abs(results[0].score - 1.8) < 1e-6


def test_mean_top_3():
    matches = [
        _make_match("P1", "Paper A", ["R1"], ["Alice"], 0.9),
        _make_match("P2", "Paper B", ["R1"], ["Alice"], 0.6),
        _make_match("P3", "Paper C", ["R2"], ["Bob"],   0.8),
        _make_match("P4", "Paper D", ["R2"], ["Bob"],   0.7),
    ]
    results = aggregate(matches, strategy="mean_top_3", top_k_researchers=5)
    # R1 mean: (0.9+0.6)/2 = 0.75; R2 mean: (0.8+0.7)/2 = 0.75 — tie, order unspecified
    assert len(results) == 2


def test_top_k_respected():
    matches = [
        _make_match(f"P{i}", f"Paper {i}", [f"R{i}"], [f"Researcher {i}"], 1.0 - i * 0.1)
        for i in range(10)
    ]
    results = aggregate(matches, strategy="max", top_k_researchers=3)
    assert len(results) == 3


def test_shared_paper_multiple_authors():
    matches = [
        _make_match("P1", "Shared Paper", ["R1", "R2"], ["Alice", "Bob"], 0.95),
    ]
    results = aggregate(matches, strategy="max", top_k_researchers=5)
    ids = {r.researcher_id for r in results}
    assert "R1" in ids and "R2" in ids


def test_matched_publications_included():
    matches = [
        _make_match("P1", "Great Paper", ["R1"], ["Alice"], 0.9),
    ]
    results = aggregate(matches, strategy="max", top_k_researchers=1)
    assert "Great Paper" in results[0].matched_publications


def test_empty_input():
    results = aggregate([], strategy="max", top_k_researchers=5)
    assert results == []
