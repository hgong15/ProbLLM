#!/usr/bin/env python
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def normalize_token(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def build_documents(meta: pd.DataFrame):
    docs = []
    for row in meta[["title", "genres"]].itertuples(index=False):
        title = str(row.title)
        genres = [g for g in str(row.genres).split("|") if g]
        genre_tokens = [f"genre_{normalize_token(g)}" for g in genres]
        year_match = re.search(r"\((\d{4})\)", title)
        year_token = f"year_{year_match.group(1)}" if year_match else ""
        docs.append(" ".join([title.replace("|", " "), *genre_tokens, year_token]).strip())
    return docs


def main():
    parser = argparse.ArgumentParser(description="Build clean MovieLens side content from title and genre metadata only.")
    parser.add_argument("--data_dir", type=Path, default=Path("./data/ml-1m"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max_features", type=int, default=200)
    args = parser.parse_args()

    raw_path = args.data_dir / "raw-data.csv"
    meta = pd.read_csv(raw_path)
    expected = ["item", "raw_item", "title", "genres"]
    missing = [col for col in expected if col not in meta.columns]
    if missing:
        raise ValueError(f"{raw_path} missing columns: {missing}")
    if not np.array_equal(meta["item"].to_numpy(), np.arange(len(meta))):
        raise ValueError("MovieLens item ids in raw-data.csv are not contiguous from 0.")

    docs = build_documents(meta)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=args.max_features,
        norm="l2",
        token_pattern=r"(?u)\b[a-zA-Z0-9_']{2,}\b",
    )
    content = vectorizer.fit_transform(docs).astype(np.float32).toarray()

    output = args.output or (args.data_dir / "ml-1m_item_content_clean_tfidf.npy")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, content)
    meta_path = output.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "source": str(raw_path),
                "rows": int(content.shape[0]),
                "dim": int(content.shape[1]),
                "feature_type": "clean_metadata_tfidf",
                "metadata_fields": ["title", "genres", "year parsed from title"],
                "max_features": int(args.max_features),
                "uses_interactions": False,
                "uses_llm_embeddings": False,
                "uses_trained_adapter": False,
                "vectorizer_vocabulary_size": int(len(vectorizer.vocabulary_)),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"Saved clean MovieLens side content: {output} shape={content.shape}")
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()
