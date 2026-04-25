"""
Aggregate paper-level matches into a researcher-level ranking.

Three strategies are implemented and can be compared:
  - max:        researcher score = best single paper score
  - sum_top_k:  researcher score = sum of top-k paper scores
  - mean_top_k: researcher score = mean of top-k paper scores
"""

from collections import defaultdict
from dataclasses import dataclass, field

from retrieve import PaperMatch


@dataclass
class ResearcherResult:
    researcher_id: str
    researcher_name: str
    score: float
    matched_publications: list[str] = field(default_factory=list)
    matched_topics: list[str] = field(default_factory=list)


def aggregate(
    paper_matches: list[PaperMatch],
    strategy: str = "sum_top_3",
    top_k_researchers: int = 10,
) -> list[ResearcherResult]:
    """
    Convert a ranked list of PaperMatch objects into a ranked list of
    ResearcherResult objects.

    strategy options: "max", "sum_top_3", "mean_top_3"
                      or "sum_top_N" / "mean_top_N" for any integer N.
    """
    # A single paper can belong to multiple researchers, so one PaperMatch may
    # contribute evidence to several researcher candidates.
    researcher_papers: dict[str, list[PaperMatch]] = defaultdict(list)
    researcher_name_map: dict[str, str] = {}

    for pm in paper_matches:
        for rid, rname in zip(pm.researcher_ids, pm.researcher_names):
            researcher_papers[rid].append(pm)
            researcher_name_map[rid] = rname

    # Parse top-k from strategy string
    k = _parse_k(strategy)

    results = []
    for rid, matches in researcher_papers.items():
        # Sort matches by descending score
        sorted_matches = sorted(matches, key=lambda m: m.score, reverse=True)
        top_matches = sorted_matches[:k] if k else sorted_matches

        # Score the researcher from their strongest matching papers rather than
        # from every paper in the corpus.
        score = _compute_score(strategy, [m.score for m in sorted_matches], k)

        # Collect supporting evidence for the API response. Titles are kept in
        # score order and deduplicated; topics are merged across top matches.
        pub_titles: list[str] = []
        topic_set: set[str] = set()
        seen_titles: set[str] = set()
        for m in top_matches:
            if m.title not in seen_titles:
                pub_titles.append(m.title)
                seen_titles.add(m.title)
            for t in m.topics:
                topic_set.add(t)

        results.append(
            ResearcherResult(
                researcher_id=rid,
                researcher_name=researcher_name_map[rid],
                score=score,
                matched_publications=pub_titles,
                matched_topics=sorted(topic_set),
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k_researchers]


def _parse_k(strategy: str) -> int | None:
    """Extract k from 'sum_top_3' -> 3, 'max' -> None (use all)."""
    if strategy == "max":
        return None
    parts = strategy.split("_")
    try:
        return int(parts[-1])
    except (ValueError, IndexError):
        return 3


def _compute_score(strategy: str, scores: list[float], k: int | None) -> float:
    if not scores:
        return 0.0
    if strategy == "max":
        return scores[0]
    top = scores[:k] if k else scores
    if strategy.startswith("mean"):
        return sum(top) / len(top)
    return sum(top)  # sum_top_k default; rewards multiple strong matches
