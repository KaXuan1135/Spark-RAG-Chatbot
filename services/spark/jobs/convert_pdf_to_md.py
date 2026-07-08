from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pymupdf4llm
from pyspark.sql import SparkSession


def convert_one(pdf_path_str: str, markdown_dir_str: str) -> dict[str, str]:
    pdf_path = Path(pdf_path_str)
    markdown_dir = Path(markdown_dir_str)
    markdown_dir.mkdir(parents=True, exist_ok=True)
    output_path = markdown_dir / f"{pdf_path.stem}.md"

    markdown = pymupdf4llm.to_markdown(str(pdf_path))
    front_matter = "\n".join(
        [
            "---",
            f"source_file: {pdf_path.name}",
            f"document_id: {pdf_path.stem}",
            f"converted_at: {datetime.now(timezone.utc).isoformat()}",
            "---",
            "",
        ]
    )
    output_path.write_text(front_matter + markdown, encoding="utf-8")
    return {
        "source_file": pdf_path.name,
        "markdown_path": str(output_path),
        "status": "converted",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-pdf-dir", required=True)
    parser.add_argument("--markdown-dir", required=True)
    parser.add_argument("--master", default="local[*]")
    args = parser.parse_args()

    raw_pdf_dir = Path(args.raw_pdf_dir)
    pdfs = sorted(str(path) for path in raw_pdf_dir.glob("*.pdf"))

    spark = SparkSession.builder.appName("pdf-to-markdown").master(args.master).getOrCreate()
    try:
        if not pdfs:
            print(f"No PDF files found in {raw_pdf_dir}")
            return

        results = spark.sparkContext.parallelize(pdfs, min(len(pdfs), 8)).map(
            lambda path: convert_one(path, args.markdown_dir)
        )
        for row in results.collect():
            print(row)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
