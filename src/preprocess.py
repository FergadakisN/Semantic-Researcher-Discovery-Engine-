"""
Preprocess raw SPARQL results into a clean local dataset.

Input:  data/raw_sparql.json  (produced by acquire.py)
Output: data/papers.json       (one record per unique paper)
        data/researchers.json  (one record per researcher)

Run:
    python src/preprocess.py
"""

import json
import re
import argparse
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def normalize(text: str) -> str:
    # Normalize whitespace early so duplicate detection is more reliable.
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def strip_uri(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def build_paper_text(title: str, topics: list[str], use_topics: bool = True) -> str:
    # This is the exact text that will later be embedded for semantic search.
    if use_topics and topics:
        topic_str = ", ".join(topics)
        return f"Title: {title} Topics: {topic_str}"
    return title


def preprocess(raw_path: Path, papers_out: Path, researchers_out: Path) -> None:
    raw = json.loads(raw_path.read_text())
    title_rows = raw["title_rows"]
    topic_rows = raw["topic_rows"]

    # --- build topic lookup: paper_id -> [topic labels] ---
    paper_topics: dict[str, list[str]] = defaultdict(list)
    for row in topic_rows:
        paper_id = strip_uri(row["paper"])
        label = normalize(row.get("topicLabel", ""))
        if label:
            paper_topics[paper_id].append(label)

    # deduplicate topic lists
    for pid in paper_topics:
        seen: set[str] = set()
        deduped = []
        for t in paper_topics[pid]:
            tl = t.lower()
            if tl not in seen:
                seen.add(tl)
                deduped.append(t)
        paper_topics[pid] = deduped

    # --- build paper records ---
    # Multiple researchers can point to the same paper, so we collapse rows
    # into one record per paper and keep all linked researchers on that record.
    paper_data: dict[str, dict] = {}
    researcher_meta: dict[str, str] = {}  # researcher_id -> name

    skipped = 0
    for row in title_rows:
        paper_id = strip_uri(row["paper"])
        researcher_id = strip_uri(row["researcher"])
        title = normalize(row.get("paperTitle", ""))
        researcher_name = normalize(row.get("researcherLabel", ""))

        if not title or len(title) < 5:
            skipped += 1
            continue

        researcher_meta[researcher_id] = researcher_name

        if paper_id not in paper_data:
            paper_data[paper_id] = {
                "paper_id": paper_id,
                "title": title,
                "topics": paper_topics.get(paper_id, []),
                "researcher_ids": [],
                "researcher_names": [],
            }

        rec = paper_data[paper_id]
        if researcher_id not in rec["researcher_ids"]:
            rec["researcher_ids"].append(researcher_id)
            rec["researcher_names"].append(researcher_name)

    print(f"Skipped {skipped} rows with missing/short titles")
    print(f"Unique papers: {len(paper_data)}")
    print(f"Unique researchers: {len(researcher_meta)}")

    # Precompute both text variants so the embedding step can choose whether
    # to index title-only text or title-plus-topic text.
    papers = []
    for rec in paper_data.values():
        rec["text_title_only"] = rec["title"]
        rec["text_title_topics"] = build_paper_text(
            rec["title"], rec["topics"], use_topics=True
        )
        papers.append(rec)

    researchers = [
        {"researcher_id": rid, "researcher_name": name}
        for rid, name in researcher_meta.items()
    ]

    papers_out.parent.mkdir(parents=True, exist_ok=True)
    papers_out.write_text(json.dumps(papers, ensure_ascii=False, indent=2))
    researchers_out.write_text(json.dumps(researchers, ensure_ascii=False, indent=2))
    print(f"Saved {len(papers)} papers  -> {papers_out}")
    print(f"Saved {len(researchers)} researchers -> {researchers_out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=str(DATA_DIR / "raw_sparql.json"))
    parser.add_argument("--papers-out", default=str(DATA_DIR / "papers.json"))
    parser.add_argument("--researchers-out", default=str(DATA_DIR / "researchers.json"))
    args = parser.parse_args()

    preprocess(
        Path(args.raw),
        Path(args.papers_out),
        Path(args.researchers_out),
    )


if __name__ == "__main__":
    main()
