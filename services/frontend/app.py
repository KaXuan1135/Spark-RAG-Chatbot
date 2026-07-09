from __future__ import annotations

import json
import os
from typing import Any

import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000").rstrip("/")


st.set_page_config(page_title="Spark SGLang RAG", page_icon="", layout="wide")
st.title("Spark + SGLang RAG")
st.caption("Upload PDFs, run the Spark ingestion pipeline, and query the indexed documents.")

st.markdown(
    """
    <style>
    [data-testid="stMetric"] {
        padding: 0.1rem 0;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.72rem;
        line-height: 1.1;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.05rem;
        line-height: 1.15;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


def api_get(path: str) -> Any:
    response = requests.get(f"{API_BASE_URL}{path}", timeout=30)
    response.raise_for_status()
    return response.json()


def api_post(path: str, *, json: dict | None = None, files: dict | None = None, timeout: int = 120) -> Any:
    response = requests.post(f"{API_BASE_URL}{path}", json=json, files=files, timeout=timeout)
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(detail)
    return response.json()


def api_post_stream(path: str, *, json_payload: dict, timeout: int = 300):
    with requests.post(
        f"{API_BASE_URL}{path}",
        json=json_payload,
        stream=True,
        timeout=timeout,
    ) as response:
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise RuntimeError(detail)

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            yield json.loads(line)



def split_inline_thinking(answer: str, reasoning_content: str | None) -> tuple[str | None, str]:
    if reasoning_content:
        return reasoning_content.strip(), answer.strip()

    lower = answer.lower()
    end_marker = "</think>"
    end = lower.find(end_marker)
    if end == -1:
        return None, answer

    starts = [
        ("<think>", len("<think>")),
        ("here's a thinking process:", len("here's a thinking process:")),
    ]
    for marker, offset in starts:
        start = lower.find(marker)
        if start == -1 or start > end:
            continue
        thinking = answer[start + offset : end].strip()
        final_answer = (answer[:start] + answer[end + len(end_marker) :]).strip()
        return thinking or None, final_answer

    thinking = answer[:end].strip()
    final_answer = answer[end + len(end_marker) :].strip()
    return thinking or None, final_answer


def render_answer(answer: str, reasoning_content: str | None) -> None:
    thinking, final_answer = split_inline_thinking(answer, reasoning_content)
    if thinking:
        with st.expander("Thinking process", expanded=False):
            st.markdown(thinking)
    st.write(final_answer or answer)




def format_ms(value: int | float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:,} ms"


def format_number(value: int | float | None, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:,.2f}{suffix}"
    return f"{value:,}{suffix}"


def render_ingest_debug(debug: dict[str, Any] | None) -> None:
    if not debug:
        return
    st.markdown("#### Ingestion Debug")
    cols = st.columns(4)
    cols[0].metric("PDF to Markdown", format_ms(debug.get("pdf_to_markdown_ms")))
    cols[1].metric("Chunking", format_ms(debug.get("chunking_ms")))
    cols[2].metric("Indexing", format_ms(debug.get("indexing_ms")))
    cols[3].metric("Total", format_ms(debug.get("total_ms")))


def render_query_debug(debug: dict[str, Any] | None) -> None:
    if not debug:
        st.info("No debug metadata returned by the API.")
        return

    st.markdown("#### Query Latency")
    cols = st.columns(6)
    cols[0].metric("Embedding", format_ms(debug.get("embedding_ms")))
    cols[1].metric("Retrieval", format_ms(debug.get("retrieval_ms")))
    cols[2].metric("Filtering", format_ms(debug.get("filtering_ms")))
    cols[3].metric("Prompt Build", format_ms(debug.get("prompt_build_ms")))
    cols[4].metric("TTFT", format_ms(debug.get("ttft_ms")))
    cols[5].metric("Total", format_ms(debug.get("total_ms")))

    st.markdown("#### Generation")
    cols = st.columns(5)
    cols[0].metric("LLM Total", format_ms(debug.get("llm_total_ms")))
    cols[1].metric("TPS", format_number(debug.get("tokens_per_second"), " tok/s"))
    cols[2].metric("Output Tokens", format_number(debug.get("reported_completion_tokens") or debug.get("estimated_completion_tokens")))
    cols[3].metric("Prompt Tokens", format_number(debug.get("reported_prompt_tokens") or debug.get("estimated_prompt_tokens")))
    cols[4].metric("Total Tokens", format_number(debug.get("reported_total_tokens")))

    st.markdown("#### Retrieval")
    cols = st.columns(4)
    cols[0].metric("Retrieved", format_number(debug.get("retrieved_chunks")))
    cols[1].metric("Used", format_number(debug.get("used_chunks")))
    cols[2].metric("Prompt Chars", format_number(debug.get("prompt_chars")))
    cols[3].metric("Completion Chars", format_number(debug.get("completion_chars")))

    notes = debug.get("notes") or []
    if notes:
        st.markdown("#### Notes")
        for note in notes:
            st.caption(note)




def render_sidebar() -> str:
    with st.sidebar:
        st.title("Spark RAG")
        if hasattr(st, "segmented_control"):
            page = st.segmented_control("Page", ["Chat", "Documents"], default="Chat", label_visibility="collapsed")
        else:
            page = st.radio("Page", ["Chat", "Documents"], label_visibility="collapsed")

        st.divider()
        st.subheader("Connection")
        st.code(API_BASE_URL, language="text")

        if st.button("Check API", use_container_width=True):
            try:
                health = api_get("/health")
                st.success(f"API status: {health.get('status', 'ok')}")
            except Exception as exc:
                st.error(f"API check failed: {exc}")

        if page == "Chat" and st.button("Clear conversation", use_container_width=True):
            st.session_state["chat_history"] = []
            st.rerun()

    return page


def render_documents_page() -> None:
    st.header("Documents")

    upload_notice = st.session_state.pop("upload_notice", None)
    upload_path = st.session_state.pop("upload_path", "")
    if upload_notice:
        st.success(upload_notice)
        if upload_path:
            st.caption(upload_path)

    try:
        documents = api_get("/documents").get("pdfs", [])
    except Exception as exc:
        st.warning(f"Could not load documents: {exc}")
        documents = []

    st.subheader("PDF Documents")
    if documents:
        for document in documents:
            st.write(f"- {document}")
    else:
        st.info("No PDFs uploaded yet.")

    st.divider()
    upload_col, ingest_col = st.columns([1, 1])

    with upload_col:
        st.subheader("Upload PDF")
        uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"], label_visibility="collapsed")
        if uploaded_file and st.button("Upload PDF", type="primary", use_container_width=True):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                result = api_post("/documents", files=files, timeout=120)
                st.session_state["upload_notice"] = f"Uploaded {uploaded_file.name}"
                st.session_state["upload_path"] = result.get("path", "")
                st.rerun()
            except Exception as exc:
                st.error(f"Upload failed: {exc}")

    with ingest_col:
        st.subheader("Run Ingestion")
        st.write("Converts PDFs to Markdown, chunks with Spark, embeds, and indexes into Qdrant.")
        if st.button("Run ingestion", type="primary", use_container_width=True):
            with st.spinner("Running Spark ingestion jobs..."):
                try:
                    result = api_post("/ingest", timeout=600)
                    st.success("Ingestion completed")
                    st.session_state["last_ingest_debug"] = result.get("debug")
                    render_ingest_debug(result.get("debug"))
                    with st.expander("Job output"):
                        st.text(result.get("detail", ""))
                except Exception as exc:
                    st.error(f"Ingestion failed: {exc}")


def render_chat_page() -> None:
    st.header("Chat")

    if st.session_state["chat_history"]:
        for turn in st.session_state["chat_history"]:
            with st.chat_message(turn["role"]):
                st.write(turn["content"])

    examples = [
        "What is this document about?",
        "What is the annual outpatient medical claim limit for a newly hired engineer in Kuala Lumpur on probation?",
        "If an employee at Astra Malaysia drives 650 kilometers for an approved corporate business trip, how much reimbursement can they claim?",
        "Can Malaysian GLC payloads fall back to Anthropic Claude if h100astra is down?",
        "During an S2 disaster recovery event in Kuala Lumpur, we need to activate our standby server h100backup-02 to host our Tier-1 client inference endpoints. What hardware limitation on this standby server violates our standard GPU deployment policy, and whose approval do we need to bypass this limitation?",
        "What are the differences in the preemption rules between Tier-2 and Tier-3 workloads on our GPU platform?",
    ]

    selected = st.selectbox("Example questions", examples)
    question = st.text_area("Message", value=selected, height=100)

    settings_cols = st.columns([1, 1, 1])
    top_k = settings_cols[0].number_input("Retrieved chunks", min_value=1, max_value=10, value=5, step=1)
    max_tokens = settings_cols[1].number_input("Max generated tokens", min_value=32, max_value=4096, value=300, step=32)
    enable_thinking = settings_cols[2].toggle(
        "Enable Qwen thinking mode",
        value=False,
        help="Off by default for faster, direct RAG answers. Turn on for harder reasoning questions.",
    )

    if st.button("Ask", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            try:
                payload = {
                    "question": question,
                    "top_k": int(top_k),
                    "max_tokens": int(max_tokens),
                    "enable_thinking": enable_thinking,
                    "history": st.session_state["chat_history"],
                }
                st.markdown("### Answer")
                thinking_box = st.empty()
                answer_box = st.empty()
                status_box = st.empty()

                streamed_answer = ""
                streamed_reasoning = ""
                sources = []
                debug = None
                ingest_debug = None
                completed_answer = ""

                status_box.caption("Retrieving context...")
                for message in api_post_stream("/query/stream", json_payload=payload, timeout=300):
                    message_type = message.get("type")

                    if message_type == "stage":
                        stage = message.get("stage", "working")
                        latency = message.get("latency_ms")
                        if latency is None:
                            status_box.caption(stage)
                        else:
                            status_box.caption(f"{stage}: {latency} ms")
                    elif message_type == "sources":
                        sources = message.get("sources", [])
                        status_box.caption("Generating answer...")
                    elif message_type == "reasoning_delta":
                        streamed_reasoning += message.get("text", "")
                        with thinking_box.expander("Thinking process", expanded=False):
                            st.markdown(streamed_reasoning)
                    elif message_type == "answer_delta":
                        streamed_answer += message.get("text", "")
                        thinking, final_answer = split_inline_thinking(streamed_answer, streamed_reasoning or None)
                        if thinking:
                            with thinking_box.expander("Thinking process", expanded=False):
                                st.markdown(thinking)
                        answer_box.write(final_answer or streamed_answer)
                    elif message_type == "done":
                        streamed_answer = message.get("answer", streamed_answer)
                        streamed_reasoning = message.get("reasoning_content") or streamed_reasoning
                        debug = message.get("debug")
                        ingest_debug = message.get("ingest_debug")
                        thinking, final_answer = split_inline_thinking(streamed_answer, streamed_reasoning or None)
                        if thinking:
                            with thinking_box.expander("Thinking process", expanded=False):
                                st.markdown(thinking)
                        completed_answer = final_answer or streamed_answer
                        answer_box.write(completed_answer)
                        status_box.empty()
                    elif message_type == "error":
                        raise RuntimeError(message.get("detail", "Unknown streaming error"))

                if completed_answer:
                    st.session_state["chat_history"].append({"role": "user", "content": question.strip()})
                    st.session_state["chat_history"].append({"role": "assistant", "content": completed_answer})

                if sources:
                    st.markdown("### Sources")
                    for idx, source in enumerate(sources, start=1):
                        with st.expander(f"[{idx}] {source.get('source_file', '')} - {source.get('section_title', '')}"):
                            st.write(source.get("heading_path", ""))
                            st.caption(f"chunk={source.get('chunk_id', '')} | score={source.get('score')}")

                with st.expander("Debug Metrics", expanded=True):
                    render_query_debug(debug)
                    last_ingest_debug = ingest_debug or st.session_state.get("last_ingest_debug")
                    if last_ingest_debug:
                        st.divider()
                        render_ingest_debug(last_ingest_debug)
            except Exception as exc:
                st.error(f"Query failed: {exc}")
                st.info("If this is an LLM connection error, start SGLang with `docker compose --profile llm up -d sglang`.")


page = render_sidebar()
if page == "Chat":
    render_chat_page()
else:
    render_documents_page()
