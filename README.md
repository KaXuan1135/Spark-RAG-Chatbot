# Spark + SGLang RAG Demo

Dockerized, plug-and-play RAG over PDF documents. The pipeline deliberately converts every PDF to Markdown before any RAG work happens.

## Architecture

```text
PDF files
  -> Spark job: PDF to Markdown
  -> Spark job: Markdown-aware chunking
  -> Spark job: local embedding + Qdrant indexing
  -> FastAPI retrieval API
  -> SGLang OpenAI-compatible chat server
```

SGLang is used for chat inference. Embeddings are generated with a local SentenceTransformers model so ingestion and retrieval stay independent from the chat model's supported API surface.

## Services

- `api`: FastAPI app, Spark job launcher, retrieval, streaming, and multi-turn query endpoints.
- `frontend`: Streamlit GUI for uploads, ingestion, multi-turn chat, sources, and debug metrics.
- `qdrant`: Vector database.
- `sglang`: Optional GPU-backed OpenAI-compatible chat server, enabled with the `llm` Compose profile.

## Quick Start

```bash
cd /home/ka_xuan/.Workspace/spark-sglang-rag-demo
cp .env.example .env
docker compose up --build
```

If Docker requires root privileges on your DGX Spark device, prefix the Compose commands with `sudo`.

This starts the API, frontend, and Qdrant. To also start SGLang:

```bash
docker compose --profile llm up --build
```

The SGLang service expects NVIDIA GPU access. Adjust `SGLANG_MODEL` in `.env` to a model that fits your hardware.


## Frontend GUI

The demo includes a simple Streamlit GUI for uploading PDFs, running ingestion, querying the RAG API, keeping a multi-turn conversation, and inspecting sources.

Start the stack:

```bash
docker compose up --build
```

For query answering, start SGLang as well:

```bash
docker compose --profile llm up --build
```

After code-only changes to the API or frontend, rebuild just those services:

```bash
docker compose up --build api frontend
```

Open the GUI at:

```text
http://localhost:8501
```

The frontend service talks to the API service through `API_BASE_URL=http://api:8000` inside Docker.

## Add PDFs

Either copy PDFs into:

```text
data/raw_pdfs/
```

or upload one through the API:

```bash
curl -X POST http://localhost:8000/documents \
  -F "file=@/path/to/document.pdf"
```

## Ingest

```bash
curl -X POST http://localhost:8000/ingest
```

This runs:

1. `convert_pdf_to_md.py`
2. `chunk_markdown.py`
3. `index_chunks.py`

Generated artifacts land in:

```text
data/markdown/
data/chunks/chunks.jsonl
data/chunks/chunks.parquet
```

## Query

The query endpoint needs an OpenAI-compatible chat server. Start the full stack with SGLang before querying:

```bash
docker compose --profile llm up --build
```

Then send a question:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What does the document say about refunds?"}'
```

For multi-turn follow-up questions, include recent conversation turns in `history`:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What about during probation?",
    "history": [
      {"role": "user", "content": "What is the outpatient medical claim limit for employees in Kuala Lumpur?"},
      {"role": "assistant", "content": "Confirmed employees in Kuala Lumpur have an annual outpatient limit of RM 3,500 [1]."}
    ]
  }'
```

The API uses conversation history only to resolve follow-up references. Policy facts still have to come from retrieved document context.

Response shape:

```json
{
  "answer": "The answer with citations like [1].",
  "sources": [
    {
      "source_file": "policy.pdf",
      "chunk_id": "policy:00003",
      "section_title": "Refunds",
      "heading_path": "Policy > Refunds",
      "score": 0.78
    }
  ]
}
```

## Configuration

Key values in `.env`:

```text
SGLANG_MODEL=Qwen/Qwen3.6-35B-A3B
SPARK_MASTER_URL=local[*]
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_MAX_WORDS=420
CHUNK_OVERLAP_WORDS=60
TOP_K=5
```

`SPARK_MASTER_URL=local[*]` keeps the demo simple while still running the preprocessing jobs through Spark. You can later point this at a standalone Spark cluster once shared volumes and dependencies are mirrored on workers.

## Notes

- First ingestion downloads the embedding model inside the API container.
- First SGLang startup downloads the chat model into your Hugging Face cache.
- PDF extraction quality depends on source PDF quality. Scanned PDFs need OCR support added as a separate preprocessing stage.




## Streaming Answers

The Streamlit GUI uses `/query/stream`, a JSON-lines streaming endpoint. It renders answer tokens as they arrive from SGLang, then fills in sources and debug metrics when generation completes. The original `/query` endpoint remains available for scripts and evaluation.

## Multi-Turn Conversation

The Streamlit GUI stores the current conversation in `st.session_state["chat_history"]` and sends it with each `/query/stream` request. The backend keeps only the most recent six turns in the prompt to limit context growth. Use the sidebar `Clear conversation` button to start a fresh chat.


## Qwen Thinking Mode

Qwen3-style models can generate explicit `<think>...</think>` reasoning by default. The API disables this by default for faster, direct RAG answers by passing SGLang's Qwen control:

```json
{"chat_template_kwargs": {"enable_thinking": false}}
```

The Streamlit GUI includes an `Enable Qwen thinking mode` toggle. Leave it off for normal policy QA; turn it on for questions that genuinely need deeper reasoning.

## GUI Debug Metrics

After each question, the Streamlit GUI shows a `Debug Metrics` panel. The query metrics include:

- `Embedding`: time to embed the user question.
- `Retrieval`: time spent searching Qdrant.
- `Filtering`: time spent applying score-based chunk filtering.
- `Prompt Build`: time to assemble the RAG prompt.
- `TTFT`: time to first streamed token from the LLM server.
- `LLM Total`: total generation time.
- `TPS`: generated tokens per second, using backend-reported token counts when available and an estimate otherwise.
- `Retrieved` / `Used`: chunks returned by vector search versus chunks actually sent to the model.

After ingestion, the GUI also shows ingestion-stage timing:

- `PDF to Markdown`
- `Chunking`
- `Indexing`
- `Total`

Token counts may be estimated when the OpenAI-compatible backend does not return usage metadata.

## Evaluation Script

After ingestion and after SGLang is running, run the policy QA checks:

```bash
python3 scripts/evaluate_rag.py --verbose
```

Optional JSON output:

```bash
python3 scripts/evaluate_rag.py --json
```

The script checks three dimensions:

- Faithfulness and safety guardrails for sovereign inference fallback.
- Tabular parsing accuracy for probation medical claims.
- Mathematical reasoning and parameter extraction for mileage reimbursement.

## Git Tracking

This directory is not currently initialized as a Git repository. To start tracking code changes locally:

```bash
cd /home/ka_xuan/.Workspace/spark-sglang-rag-demo
git init
git add README.md docker-compose.yml scripts services .env.example .gitignore .dockerignore data/raw_pdfs/.gitkeep data/markdown/.gitkeep data/chunks/.gitkeep data/qdrant/.gitkeep
git commit -m "Initial RAG demo project"
```

The existing `.gitignore` keeps generated data, Qdrant storage, local `.env`, and Python caches out of version control. Commit source code and configuration templates; avoid committing PDFs, generated chunks, vector database files, model caches, or secrets.
