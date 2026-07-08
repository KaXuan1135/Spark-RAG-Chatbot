from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    data_dir: Path = Path("/app/data")
    raw_pdf_dir: Path = Path("/app/data/raw_pdfs")
    markdown_dir: Path = Path("/app/data/markdown")
    chunks_dir: Path = Path("/app/data/chunks")

    spark_master_url: str = "local[*]"

    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "rag_chunks"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_max_words: int = 420
    chunk_overlap_words: int = 60
    top_k: int = 5

    llm_base_url: str = "http://sglang:30000/v1"
    llm_api_key: str = "EMPTY"
    llm_model: str = "Qwen/Qwen3.6-35B-A3B"


settings = Settings()
