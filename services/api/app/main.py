from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from openai import APIConnectionError, BadRequestError, OpenAI
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from rag_core.config import settings


logger = logging.getLogger("rag_api")

app = FastAPI(title="Spark + SGLang RAG Demo")
embedding_model: SentenceTransformer | None = None


class IngestDebug(BaseModel):
    pdf_to_markdown_ms: int | None = None
    chunking_ms: int | None = None
    indexing_ms: int | None = None
    total_ms: int | None = None


latest_ingest_debug: IngestDebug | None = None


SYSTEM_PROMPT = (
    "You are a policy QA assistant. Answer using only the supplied context. "
    "Use the recent conversation only to resolve follow-up references, not as a source of policy facts. "
    "Give the direct answer first. Keep non-calculation answers to one or two concise sentences. "
    "When the user asks for a total amount, reimbursement, limit, or claim value, compute the final numeric answer from the retrieved parameters. "
    "For calculations, include one compact formula line with the substituted values and final total. "
    "Do not quote long policy passages, dump retrieved context, reproduce tables, or summarize entire tables unless the user explicitly asks for the table. "
    "For table questions, identify the single matching row by checking all constraints expressed in the user question against the row headers and row values; answer only from the relevant columns. "
    "Do not mention or use values from non-matching rows, countries, employee statuses, tiers, or examples, even if they appear in the retrieved context. "
    "Do not stop after quoting rates or parameters when arithmetic is required. "
    "For compliance, sovereignty, data residency, or fallback questions, state the policy rule, the allowed or forbidden action, and required error handling; do not infer or propose backup routes unless the policy explicitly permits them. "
    "When the user names a proposed provider or endpoint, explicitly say whether that named route is allowed or forbidden. "
    "If the context is insufficient, say you do not know. "
    "Cite sources with bracket numbers like [1]."
)


ANSWER_INSTRUCTIONS = (
    "Answer directly. Include only the facts needed to answer the question, plus citations. "
    "Do not include unrelated table entries or long quotes. "
    "If the question asks for a computed amount, calculate it and show the minimal formula. "
    "For fallback or compliance questions, do not speculate about alternatives; answer only what the policy allows or forbids, "
    "and explicitly address the proposed provider or endpoint named in the question."
)


class IngestResponse(BaseModel):
    status: str
    detail: str
    debug: IngestDebug | None = None


class ChatTurn(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None
    max_tokens: int | None = Field(default=None, ge=32, le=4096)
    enable_thinking: bool = False
    history: list[ChatTurn] = Field(default_factory=list)


class Source(BaseModel):
    source_file: str
    chunk_id: str
    section_title: str
    heading_path: str
    score: float | None = None


class QueryDebug(BaseModel):
    embedding_ms: int
    retrieval_ms: int
    filtering_ms: int
    prompt_build_ms: int
    llm_total_ms: int
    ttft_ms: int | None = None
    total_ms: int
    retrieved_chunks: int
    used_chunks: int
    prompt_chars: int
    completion_chars: int
    estimated_prompt_tokens: int
    estimated_completion_tokens: int
    reported_prompt_tokens: int | None = None
    reported_completion_tokens: int | None = None
    reported_total_tokens: int | None = None
    tokens_per_second: float | None = None
    notes: list[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    reasoning_content: str | None = None
    debug: QueryDebug | None = None
    ingest_debug: IngestDebug | None = None


def format_history(history: list[ChatTurn], max_turns: int = 6) -> str:
    allowed_roles = {"user", "assistant"}
    lines = []
    for turn in history[-max_turns:]:
        role = turn.role.lower().strip()
        content = turn.content.strip()
        if role not in allowed_roles or not content:
            continue
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {content}")
    return "\n".join(lines)


def build_chat_messages(context_blocks: list[str], question: str, history: list[ChatTurn]) -> list[dict[str, str]]:
    history_text = format_history(history)
    user_parts = ["Context:\n\n" + "\n\n".join(context_blocks)]
    if history_text:
        user_parts.append("Recent conversation:\n\n" + history_text)
    user_parts.append(f"Question: {question}")
    user_parts.append(ANSWER_INSTRUCTIONS)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def split_inline_thinking(text: str) -> tuple[str | None, str]:
    patterns = [
        re.compile(r"<think>(?P<thinking>.*?)</think>", re.IGNORECASE | re.DOTALL),
        re.compile(r"Here's a thinking process:\s*(?P<thinking>.*?)</think>", re.IGNORECASE | re.DOTALL),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if not match:
            continue
        thinking = match.group("thinking").strip()
        answer = (text[: match.start()] + text[match.end() :]).strip()
        return thinking or None, answer
    return None, text



def estimate_tokens(text: str) -> int:
    return max(1, round(len(text) / 4)) if text else 0


def elapsed_ms(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


def get_embedding_model() -> SentenceTransformer:
    global embedding_model
    if embedding_model is None:
        embedding_model = SentenceTransformer(settings.embedding_model)
    return embedding_model


def run_job(script_name: str, args: list[str]) -> str:
    command = ["spark-submit", "--master", settings.spark_master_url, f"/app/jobs/{script_name}", *args]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    output = "\n".join(part for part in [completed.stdout, completed.stderr] if part)
    if completed.returncode != 0:
        raise HTTPException(status_code=500, detail=output[-4000:])
    return output



@app.on_event("startup")
def load_embedding_model_on_startup() -> None:
    logger.info("Loading embedding model: %s", settings.embedding_model)
    started = time.perf_counter()
    get_embedding_model()
    logger.info("Embedding model loaded in %s ms", elapsed_ms(started))



@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)) -> dict[str, str]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported")

    settings.raw_pdf_dir.mkdir(parents=True, exist_ok=True)
    destination = settings.raw_pdf_dir / Path(file.filename).name
    with destination.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)

    return {"status": "uploaded", "path": str(destination)}


@app.get("/documents")
def list_documents() -> dict[str, list[str]]:
    settings.raw_pdf_dir.mkdir(parents=True, exist_ok=True)
    return {"pdfs": sorted(path.name for path in settings.raw_pdf_dir.glob("*.pdf"))}


@app.post("/ingest", response_model=IngestResponse)
def ingest() -> IngestResponse:
    global latest_ingest_debug
    ingest_start = time.perf_counter()
    settings.markdown_dir.mkdir(parents=True, exist_ok=True)
    settings.chunks_dir.mkdir(parents=True, exist_ok=True)

    stage_start = time.perf_counter()
    convert_output = run_job(
        "convert_pdf_to_md.py",
        [
            "--raw-pdf-dir",
            str(settings.raw_pdf_dir),
            "--markdown-dir",
            str(settings.markdown_dir),
        ],
    )
    convert_ms = elapsed_ms(stage_start)

    stage_start = time.perf_counter()
    chunk_output = run_job(
        "chunk_markdown.py",
        [
            "--markdown-dir",
            str(settings.markdown_dir),
            "--chunks-dir",
            str(settings.chunks_dir),
            "--max-words",
            str(settings.chunk_max_words),
            "--overlap-words",
            str(settings.chunk_overlap_words),
        ],
    )
    chunking_ms = elapsed_ms(stage_start)

    stage_start = time.perf_counter()
    index_output = run_job(
        "index_chunks.py",
        [
            "--chunks-path",
            str(settings.chunks_dir / "chunks.jsonl"),
            "--qdrant-url",
            settings.qdrant_url,
            "--collection",
            settings.qdrant_collection,
            "--embedding-model",
            settings.embedding_model,
        ],
    )
    indexing_ms = elapsed_ms(stage_start)

    latest_ingest_debug = IngestDebug(
        pdf_to_markdown_ms=convert_ms,
        chunking_ms=chunking_ms,
        indexing_ms=indexing_ms,
        total_ms=elapsed_ms(ingest_start),
    )

    return IngestResponse(
        status="ok",
        detail="\n".join([convert_output, chunk_output, index_output])[-4000:],
        debug=latest_ingest_debug,
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    total_start = time.perf_counter()
    notes: list[str] = []
    logger.info("/query start top_k=%s max_tokens=%s enable_thinking=%s", request.top_k, request.max_tokens, request.enable_thinking)

    embed_start = time.perf_counter()
    model = get_embedding_model()
    vector = model.encode([request.question], normalize_embeddings=True)[0].tolist()
    embedding_ms = elapsed_ms(embed_start)
    logger.info("/query embedding done in %s ms", embedding_ms)

    retrieval_start = time.perf_counter()
    qdrant = QdrantClient(url=settings.qdrant_url)
    hits = qdrant.search(
        collection_name=settings.qdrant_collection,
        query_vector=vector,
        limit=request.top_k or settings.top_k,
        with_payload=True,
    )
    retrieval_ms = elapsed_ms(retrieval_start)
    logger.info("/query retrieval done in %s ms hits=%s", retrieval_ms, len(hits))

    if not hits:
        raise HTTPException(status_code=404, detail="No indexed chunks found. Run /ingest first.")

    filter_start = time.perf_counter()
    top_score = hits[0].score or 0.0
    relevant_hits = [
        hit
        for hit in hits
        if (hit.score or 0.0) >= 0.3 and (hit.score or 0.0) >= top_score * 0.75
    ]
    if not relevant_hits:
        relevant_hits = hits[:1]
        notes.append("No chunks passed the score filter; used the top retrieved chunk.")
    filtering_ms = elapsed_ms(filter_start)

    prompt_start = time.perf_counter()
    context_blocks = []
    sources: list[Source] = []
    for index, hit in enumerate(relevant_hits, start=1):
        payload = hit.payload or {}
        context_blocks.append(
            "\n".join(
                [
                    f"[{index}] {payload.get('source_file')} :: {payload.get('heading_path')}",
                    payload.get("chunk_text", ""),
                ]
            )
        )
        sources.append(
            Source(
                source_file=payload.get("source_file", ""),
                chunk_id=payload.get("chunk_id", ""),
                section_title=payload.get("section_title", ""),
                heading_path=payload.get("heading_path", ""),
                score=hit.score,
            )
        )

    messages = build_chat_messages(context_blocks, request.question, request.history)
    prompt_text = "\n".join(message["content"] for message in messages)
    prompt_build_ms = elapsed_ms(prompt_start)
    logger.info("/query prompt built in %s ms used_chunks=%s", prompt_build_ms, len(relevant_hits))
    max_tokens = request.max_tokens or 300
    extra_body = {
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": request.enable_thinking},
    }
    if not request.enable_thinking:
        notes.append("Qwen thinking mode disabled via chat_template_kwargs.enable_thinking=false.")

    client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    llm_start = time.perf_counter()
    ttft_ms: int | None = None
    answer_parts: list[str] = []
    reasoning_parts: list[str] = []
    reported_prompt_tokens: int | None = None
    reported_completion_tokens: int | None = None
    reported_total_tokens: int | None = None

    def consume_stream(stream) -> None:
        nonlocal reported_prompt_tokens, reported_completion_tokens, reported_total_tokens, ttft_ms
        for event in stream:
            usage = getattr(event, "usage", None)
            if usage is not None:
                reported_prompt_tokens = getattr(usage, "prompt_tokens", None)
                reported_completion_tokens = getattr(usage, "completion_tokens", None)
                reported_total_tokens = getattr(usage, "total_tokens", None)

            if not event.choices:
                continue

            delta = event.choices[0].delta
            reasoning_content = getattr(delta, "reasoning_content", None)
            if reasoning_content:
                reasoning_parts.append(reasoning_content)

            content = getattr(delta, "content", None)
            if content:
                if ttft_ms is None:
                    ttft_ms = elapsed_ms(llm_start)
                answer_parts.append(content)

    try:
        try:
            stream = client.chat.completions.create(
                model=settings.llm_model,
                messages=messages,
                temperature=0.1,
                max_tokens=max_tokens,
                stream=True,
                stream_options={"include_usage": True},
                extra_body=extra_body,
            )
        except (TypeError, BadRequestError):
            notes.append("Streaming usage metadata was unavailable; token counts are estimated.")
            stream = client.chat.completions.create(
                model=settings.llm_model,
                messages=messages,
                temperature=0.1,
                max_tokens=max_tokens,
                stream=True,
                extra_body=extra_body,
            )
        consume_stream(stream)
    except APIConnectionError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Could not connect to LLM endpoint {settings.llm_base_url}. "
                "Start SGLang with `docker compose --profile llm up --build`, "
                "or point LLM_BASE_URL at a running OpenAI-compatible server."
            ),
        ) from exc

    llm_total_ms = elapsed_ms(llm_start)
    logger.info("/query llm stream done in %s ms ttft=%s", llm_total_ms, ttft_ms)
    answer = "".join(answer_parts)
    reasoning_content = "".join(reasoning_parts).strip() or None
    inline_reasoning, answer = split_inline_thinking(answer)
    if inline_reasoning:
        reasoning_content = "\n\n".join(part for part in [reasoning_content, inline_reasoning] if part)

    estimated_prompt_tokens = estimate_tokens(prompt_text)
    estimated_completion_tokens = estimate_tokens(answer)
    output_tokens = reported_completion_tokens or estimated_completion_tokens
    generation_seconds = max((llm_total_ms - (ttft_ms or 0)) / 1000, 0.001)
    tokens_per_second = round(output_tokens / generation_seconds, 2) if output_tokens else None

    if reported_total_tokens is None:
        notes.append("Token counts are estimated from character length because the backend did not return usage metadata.")
    if ttft_ms is None:
        notes.append("TTFT is unavailable because the backend did not stream token deltas.")

    return QueryResponse(
        answer=answer,
        sources=sources,
        reasoning_content=reasoning_content,
        ingest_debug=latest_ingest_debug,
        debug=QueryDebug(
            embedding_ms=embedding_ms,
            retrieval_ms=retrieval_ms,
            filtering_ms=filtering_ms,
            prompt_build_ms=prompt_build_ms,
            llm_total_ms=llm_total_ms,
            ttft_ms=ttft_ms,
            total_ms=elapsed_ms(total_start),
            retrieved_chunks=len(hits),
            used_chunks=len(relevant_hits),
            prompt_chars=len(prompt_text),
            completion_chars=len(answer),
            estimated_prompt_tokens=estimated_prompt_tokens,
            estimated_completion_tokens=estimated_completion_tokens,
            reported_prompt_tokens=reported_prompt_tokens,
            reported_completion_tokens=reported_completion_tokens,
            reported_total_tokens=reported_total_tokens,
            tokens_per_second=tokens_per_second,
            notes=notes,
        ),
    )


@app.post("/query/stream")
def query_stream(request: QueryRequest) -> StreamingResponse:
    def event(payload: dict) -> str:
        return json.dumps(payload, ensure_ascii=False) + "\n"

    def generate():
        total_start = time.perf_counter()
        notes: list[str] = []
        logger.info(
            "/query/stream start top_k=%s max_tokens=%s enable_thinking=%s",
            request.top_k,
            request.max_tokens,
            request.enable_thinking,
        )

        try:
            embed_start = time.perf_counter()
            model = get_embedding_model()
            vector = model.encode([request.question], normalize_embeddings=True)[0].tolist()
            embedding_ms = elapsed_ms(embed_start)
            yield event({"type": "stage", "stage": "embedding", "latency_ms": embedding_ms})

            retrieval_start = time.perf_counter()
            qdrant = QdrantClient(url=settings.qdrant_url)
            hits = qdrant.search(
                collection_name=settings.qdrant_collection,
                query_vector=vector,
                limit=request.top_k or settings.top_k,
                with_payload=True,
            )
            retrieval_ms = elapsed_ms(retrieval_start)
            yield event({"type": "stage", "stage": "retrieval", "latency_ms": retrieval_ms, "hits": len(hits)})

            if not hits:
                yield event({"type": "error", "detail": "No indexed chunks found. Run /ingest first."})
                return

            filter_start = time.perf_counter()
            top_score = hits[0].score or 0.0
            relevant_hits = [
                hit
                for hit in hits
                if (hit.score or 0.0) >= 0.3 and (hit.score or 0.0) >= top_score * 0.75
            ]
            if not relevant_hits:
                relevant_hits = hits[:1]
                notes.append("No chunks passed the score filter; used the top retrieved chunk.")
            filtering_ms = elapsed_ms(filter_start)

            prompt_start = time.perf_counter()
            context_blocks = []
            sources: list[Source] = []
            for index, hit in enumerate(relevant_hits, start=1):
                payload = hit.payload or {}
                context_blocks.append(
                    "\n".join(
                        [
                            f"[{index}] {payload.get('source_file')} :: {payload.get('heading_path')}",
                            payload.get("chunk_text", ""),
                        ]
                    )
                )
                sources.append(
                    Source(
                        source_file=payload.get("source_file", ""),
                        chunk_id=payload.get("chunk_id", ""),
                        section_title=payload.get("section_title", ""),
                        heading_path=payload.get("heading_path", ""),
                        score=hit.score,
                    )
                )

            yield event({"type": "sources", "sources": [source.model_dump() for source in sources]})

            messages = build_chat_messages(context_blocks, request.question, request.history)
            prompt_text = "\n".join(message["content"] for message in messages)
            prompt_build_ms = elapsed_ms(prompt_start)
            max_tokens = request.max_tokens or 300
            extra_body = {
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": request.enable_thinking},
            }
            if not request.enable_thinking:
                notes.append("Qwen thinking mode disabled via chat_template_kwargs.enable_thinking=false.")

            client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
            llm_start = time.perf_counter()
            ttft_ms: int | None = None
            answer_parts: list[str] = []
            reasoning_parts: list[str] = []
            reported_prompt_tokens: int | None = None
            reported_completion_tokens: int | None = None
            reported_total_tokens: int | None = None

            try:
                try:
                    stream = client.chat.completions.create(
                        model=settings.llm_model,
                        messages=messages,
                        temperature=0.1,
                        max_tokens=max_tokens,
                        stream=True,
                        stream_options={"include_usage": True},
                        extra_body=extra_body,
                    )
                except (TypeError, BadRequestError):
                    notes.append("Streaming usage metadata was unavailable; token counts are estimated.")
                    stream = client.chat.completions.create(
                        model=settings.llm_model,
                        messages=messages,
                        temperature=0.1,
                        max_tokens=max_tokens,
                        stream=True,
                        extra_body=extra_body,
                    )

                for chunk in stream:
                    usage = getattr(chunk, "usage", None)
                    if usage is not None:
                        reported_prompt_tokens = getattr(usage, "prompt_tokens", None)
                        reported_completion_tokens = getattr(usage, "completion_tokens", None)
                        reported_total_tokens = getattr(usage, "total_tokens", None)

                    if not chunk.choices:
                        continue

                    delta = chunk.choices[0].delta
                    reasoning_delta = getattr(delta, "reasoning_content", None)
                    if reasoning_delta:
                        reasoning_parts.append(reasoning_delta)
                        yield event({"type": "reasoning_delta", "text": reasoning_delta})

                    content_delta = getattr(delta, "content", None)
                    if content_delta:
                        if ttft_ms is None:
                            ttft_ms = elapsed_ms(llm_start)
                            yield event({"type": "stage", "stage": "ttft", "latency_ms": ttft_ms})
                        answer_parts.append(content_delta)
                        yield event({"type": "answer_delta", "text": content_delta})
            except APIConnectionError as exc:
                yield event(
                    {
                        "type": "error",
                        "detail": (
                            f"Could not connect to LLM endpoint {settings.llm_base_url}. "
                            "Start SGLang with `docker compose --profile llm up --build`, "
                            "or point LLM_BASE_URL at a running OpenAI-compatible server."
                        ),
                    }
                )
                return

            llm_total_ms = elapsed_ms(llm_start)
            answer = "".join(answer_parts)
            reasoning_content = "".join(reasoning_parts).strip() or None
            inline_reasoning, answer = split_inline_thinking(answer)
            if inline_reasoning:
                reasoning_content = "\n\n".join(part for part in [reasoning_content, inline_reasoning] if part)

            estimated_prompt_tokens = estimate_tokens(prompt_text)
            estimated_completion_tokens = estimate_tokens(answer)
            output_tokens = reported_completion_tokens or estimated_completion_tokens
            generation_seconds = max((llm_total_ms - (ttft_ms or 0)) / 1000, 0.001)
            tokens_per_second = round(output_tokens / generation_seconds, 2) if output_tokens else None

            if reported_total_tokens is None:
                notes.append("Token counts are estimated from character length because the backend did not return usage metadata.")
            if ttft_ms is None:
                notes.append("TTFT is unavailable because the backend did not stream token deltas.")

            debug = QueryDebug(
                embedding_ms=embedding_ms,
                retrieval_ms=retrieval_ms,
                filtering_ms=filtering_ms,
                prompt_build_ms=prompt_build_ms,
                llm_total_ms=llm_total_ms,
                ttft_ms=ttft_ms,
                total_ms=elapsed_ms(total_start),
                retrieved_chunks=len(hits),
                used_chunks=len(relevant_hits),
                prompt_chars=len(prompt_text),
                completion_chars=len(answer),
                estimated_prompt_tokens=estimated_prompt_tokens,
                estimated_completion_tokens=estimated_completion_tokens,
                reported_prompt_tokens=reported_prompt_tokens,
                reported_completion_tokens=reported_completion_tokens,
                reported_total_tokens=reported_total_tokens,
                tokens_per_second=tokens_per_second,
                notes=notes,
            )
            yield event(
                {
                    "type": "done",
                    "answer": answer,
                    "reasoning_content": reasoning_content,
                    "debug": debug.model_dump(),
                    "ingest_debug": latest_ingest_debug.model_dump() if latest_ingest_debug else None,
                }
            )
        except Exception as exc:
            logger.exception("/query/stream failed")
            yield event({"type": "error", "detail": str(exc)})

    return StreamingResponse(generate(), media_type="application/x-ndjson")
