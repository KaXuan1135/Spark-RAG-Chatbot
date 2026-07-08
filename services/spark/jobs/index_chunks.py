from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer


def batched(items: list[dict], size: int) -> list[list[dict]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-path", required=True)
    parser.add_argument("--qdrant-url", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    chunks_path = Path(args.chunks_path)
    rows = [
        json.loads(line)
        for line in chunks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        print(f"No chunks found in {chunks_path}")
        return

    model = SentenceTransformer(args.embedding_model)
    client = QdrantClient(url=args.qdrant_url)

    sample_vector = model.encode(["dimension probe"], normalize_embeddings=True)[0]
    client.recreate_collection(
        collection_name=args.collection,
        vectors_config=VectorParams(size=len(sample_vector), distance=Distance.COSINE),
    )

    total = 0
    for group in batched(rows, args.batch_size):
        vectors = model.encode(
            [row["chunk_text"] for row in group],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        points = [
            PointStruct(
                id=str(uuid.UUID(row["content_hash"][:32])),
                vector=vector.tolist(),
                payload=row,
            )
            for row, vector in zip(group, vectors)
        ]
        client.upsert(collection_name=args.collection, points=points)
        total += len(points)

    print(f"Indexed {total} chunks into Qdrant collection {args.collection}")


if __name__ == "__main__":
    main()
