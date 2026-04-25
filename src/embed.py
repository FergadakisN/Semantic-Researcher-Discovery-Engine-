"""
Embed all unique papers and persist the embedding matrix.

Output (in embeddings/):
    embeddings_title_only.npy      float32 matrix [n_papers, dim]
    embeddings_title_topics.npy    float32 matrix [n_papers, dim]
    paper_ids.json                 list of paper_ids (row order)

Run:
    python src/embed.py
    python src/embed.py --field title_only      # only title variant
    python src/embed.py --model all-mpnet-base-v2
"""

import json
import argparse
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data"
EMB_DIR = Path(__file__).parent.parent / "embeddings"

DEFAULT_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 256


def embed_texts(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    # Embeddings are normalized so retrieval can use a fast dot product while
    # still behaving like cosine similarity.
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # unit vectors -> dot product == cosine similarity
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers", default=str(DATA_DIR / "papers.json"))
    parser.add_argument("--out-dir", default=str(EMB_DIR))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--field",
        choices=["title_only", "title_topics", "both"],
        default="both",
    )
    args = parser.parse_args()

    papers = json.loads(Path(args.papers).read_text())
    print(f"Loaded {len(papers)} papers")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The paper id list defines the row order shared by all saved matrices.
    paper_ids = [p["paper_id"] for p in papers]
    (out_dir / "paper_ids.json").write_text(json.dumps(paper_ids))

    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model)

    fields_to_embed = (
        ["title_only", "title_topics"] if args.field == "both" else [args.field]
    )

    for field in fields_to_embed:
        key = f"text_{field}"
        texts = [p[key] for p in papers]
        print(f"\nEmbedding field '{field}' ({len(texts)} texts) ...")
        matrix = embed_texts(texts, model)
        # Each row in the matrix corresponds to the paper id at the same index
        # in paper_ids.json.
        out_path = out_dir / f"embeddings_{field}.npy"
        np.save(out_path, matrix)
        print(f"Saved {matrix.shape} matrix -> {out_path}")


if __name__ == "__main__":
    main()
