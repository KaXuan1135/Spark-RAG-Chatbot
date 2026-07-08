from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from pyspark.sql import SparkSession

from rag_core.chunking import chunk_markdown_file


def chunk_one(markdown_path_str: str, max_words: int, overlap_words: int) -> list[dict]:
    markdown_path = Path(markdown_path_str)
    markdown_text = markdown_path.read_text(encoding="utf-8")
    return [
        asdict(chunk)
        for chunk in chunk_markdown_file(
            markdown_path,
            markdown_text,
            max_words=max_words,
            overlap_words=overlap_words,
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown-dir", required=True)
    parser.add_argument("--chunks-dir", required=True)
    parser.add_argument("--max-words", type=int, default=420)
    parser.add_argument("--overlap-words", type=int, default=60)
    parser.add_argument("--master", default="local[*]")
    args = parser.parse_args()

    markdown_dir = Path(args.markdown_dir)
    chunks_dir = Path(args.chunks_dir)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    output_path = chunks_dir / "chunks.jsonl"

    markdown_files = sorted(str(path) for path in markdown_dir.glob("*.md"))
    spark = SparkSession.builder.appName("markdown-chunking").master(args.master).getOrCreate()
    try:
        if not markdown_files:
            print(f"No Markdown files found in {markdown_dir}")
            output_path.write_text("", encoding="utf-8")
            return

        rows = (
            spark.sparkContext.parallelize(markdown_files, min(len(markdown_files), 8))
            .flatMap(lambda path: chunk_one(path, args.max_words, args.overlap_words))
            .collect()
        )
        with output_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        spark.createDataFrame(rows).write.mode("overwrite").parquet(str(chunks_dir / "chunks.parquet"))
        print(f"Wrote {len(rows)} chunks to {output_path}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
