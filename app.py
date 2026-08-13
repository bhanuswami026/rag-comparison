import os
import hashlib

import streamlit as st

from config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_TOP_K,
    EMBEDDING_MODELS,
    LLM_PROVIDERS,
    MAX_TOP_K,
    MIN_TOP_K,
)
from ingestion import ingest_documents
from llm_providers import LLMProviderError, describe_provider
from strategies import (
    agentic_rag_answer,
    build_entity_graph,
    graph_augmented_retrieval,
    graph_rag_answer,
    simple_rag_answer,
)
from vector_store import build_vector_store


st.set_page_config(
    page_title="RAG Architecture Comparison",
    page_icon="🔎",
    layout="wide",
)


def api_key_status(provider_name: str) -> tuple[str, bool]:
    provider_config = LLM_PROVIDERS[provider_name]
    env_name = provider_config["api_key_env"]
    has_key = bool(os.getenv(env_name))
    return env_name, has_key


def chunk_fingerprint(chunks: list[dict]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(str(chunk["id"]).encode("utf-8"))
        digest.update(chunk["source"].encode("utf-8"))
        digest.update(chunk["text"].encode("utf-8"))
    return digest.hexdigest()


def get_shared_vector_store(chunks: list[dict], embedding_model: str, rebuild: bool):
    if not chunks:
        st.session_state.pop("vector_store", None)
        st.session_state.pop("vector_store_key", None)
        return None, None

    store_key = f"{embedding_model}:{chunk_fingerprint(chunks)}"
    existing_key = st.session_state.get("vector_store_key")
    existing_store = st.session_state.get("vector_store")

    if existing_store is not None and existing_key == store_key and not rebuild:
        return existing_store, store_key

    if rebuild:
        with st.spinner(f"Building shared FAISS index with {embedding_model}..."):
            st.session_state.vector_store = build_vector_store(chunks, embedding_model)
            st.session_state.vector_store_key = store_key
        return st.session_state.vector_store, store_key

    return None, store_key


def render_chunk(result: dict, show_score: bool = True, note=None) -> None:
    score = result.get("score")
    score_text = f" · score {score:.3f}" if show_score and isinstance(score, (int, float)) else ""
    rank_text = f"#{result['rank']} · " if "rank" in result else ""
    st.markdown(f"**{rank_text}Chunk {result['id']}{score_text}**")
    st.caption(f"Source: {result['source']}")
    if note:
        st.caption(note)
    st.write(result["text"])


st.title("RAG Architecture Comparison")
st.caption("Simple RAG vs Graph RAG vs Agentic RAG using one shared ingestion and FAISS pipeline.")

with st.sidebar:
    st.header("Project Details")
    st.write("Researcher: **Bhanu Swami**")
    st.write("M. Tech Research Project")

    st.divider()
    st.header("Configuration")

    embedding_model = st.selectbox(
        "Embedding model",
        EMBEDDING_MODELS,
        index=0,
    )

    llm_provider = st.selectbox(
        "LLM provider",
        list(LLM_PROVIDERS.keys()),
        index=0,
    )

    env_name, has_key = api_key_status(llm_provider)
    if has_key:
        st.success(f"{env_name} is configured")
    else:
        st.warning(f"{env_name} is not set")

    provider_details = describe_provider(llm_provider)
    st.caption(f"Model: `{provider_details['model']}`")

    top_k = st.slider(
        "Retrieved chunks",
        min_value=MIN_TOP_K,
        max_value=MAX_TOP_K,
        value=DEFAULT_TOP_K,
    )

    st.divider()
    st.subheader("Ingestion")
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
    )

    use_sample_doc = st.checkbox("Use sample document", value=True)
    rebuild_index = st.button("Build / rebuild shared index", type="primary")

try:
    documents, chunks = ingest_documents(
        uploaded_files=uploaded_files,
        use_sample_doc=use_sample_doc,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )
    ingestion_error = None
except Exception as exc:
    documents = []
    chunks = []
    ingestion_error = str(exc)

if ingestion_error:
    st.error(f"Ingestion failed: {ingestion_error}")
elif chunks:
    all_entities = sorted({entity for chunk in chunks for entity in chunk["entities"]})
    st.success(
        f"Ingested {len(documents)} document(s) into {len(chunks)} chunk(s). "
        f"Found {len(all_entities)} unique lightweight entities."
    )
else:
    st.warning("No documents available yet. Upload files or enable the sample document.")

vector_store = None
vector_store_error = None
if chunks and not ingestion_error:
    try:
        vector_store, _ = get_shared_vector_store(chunks, embedding_model, rebuild_index)
    except Exception as exc:
        vector_store_error = str(exc)

if vector_store_error:
    st.error(f"Vector store build failed: {vector_store_error}")
elif vector_store is not None:
    st.info(
        f"Shared FAISS index ready with {vector_store.size} vector(s), "
        f"{vector_store.dimension} dimensions, using `{vector_store.embedding_model_name}`."
    )
elif chunks:
    st.info("Documents are ingested. Click **Build / rebuild shared index** to create the FAISS store.")

entity_graph = build_entity_graph(chunks) if chunks else None
metric_cols = st.columns(4)
with metric_cols[0]:
    st.metric("Documents", len(documents))
with metric_cols[1]:
    st.metric("Chunks", len(chunks))
with metric_cols[2]:
    st.metric("Graph edges", entity_graph.number_of_edges() if chunks else 0)
with metric_cols[3]:
    st.metric("FAISS vectors", vector_store.size if vector_store else 0)

settings_col, query_col = st.columns([1, 2])

with settings_col:
    st.subheader("Current Settings")
    st.write(f"Embedding: `{embedding_model}`")
    st.write(f"LLM: `{llm_provider}`")
    st.write(f"LLM model: `{provider_details['model']}`")
    st.write(f"Provider API key: `{provider_details['api_key_env']}`")
    st.write(f"Top-k: `{top_k}`")
    st.write(f"Chunk size: `{DEFAULT_CHUNK_SIZE}`")
    st.write(f"Chunk overlap: `{DEFAULT_CHUNK_OVERLAP}`")
    st.write(f"Uploaded files: `{len(uploaded_files or [])}`")
    st.write(f"Sample doc: `{'enabled' if use_sample_doc else 'disabled'}`")
    st.write(f"Rebuild requested: `{'yes' if rebuild_index else 'no'}`")
    st.write(f"Documents: `{len(documents)}`")
    st.write(f"Chunks: `{len(chunks)}`")
    st.write(f"Graph nodes: `{entity_graph.number_of_nodes() if entity_graph else 0}`")
    st.write(f"Graph edges: `{entity_graph.number_of_edges() if entity_graph else 0}`")
    st.write(f"FAISS vectors: `{vector_store.size if vector_store else 0}`")
    st.write(f"Embedding dimensions: `{vector_store.dimension if vector_store else 0}`")

with query_col:
    st.subheader("Query")
    query = st.text_area(
        "Ask a question",
        placeholder="Example: Compare the key tradeoffs between the retrieved approaches.",
        height=120,
    )
    run_comparison = st.button("Run comparison", disabled=not query.strip())

    with st.expander("Ingested chunks", expanded=False):
        if chunks:
            for chunk in chunks:
                st.markdown(f"**Chunk {chunk['id']}** from `{chunk['source']}`")
                st.caption(
                    "Entities: "
                    + (", ".join(chunk["entities"]) if chunk["entities"] else "none detected")
                )
                st.write(chunk["text"])
        else:
            st.write("No chunks to show.")

st.divider()

simple_col, graph_col, agentic_col = st.columns(3)
simple_result = None
simple_error = None
graph_preview = None
graph_result = None
graph_error = None
agentic_result = None
agentic_error = None

if query.strip() and vector_store is not None and chunks:
    graph_preview = graph_augmented_retrieval(
        query=query,
        chunks=chunks,
        vector_store=vector_store,
        top_k=top_k,
    )

simple_context_k = top_k
if graph_preview:
    simple_context_k = max(top_k, len(graph_preview["augmented_chunks"]))
if vector_store is not None:
    simple_context_k = min(simple_context_k, vector_store.size)

retrieved_chunks = []
if query.strip() and vector_store is not None:
    retrieved_chunks = vector_store.search(query, top_k=simple_context_k)

if run_comparison and query.strip():
    if vector_store is None:
        simple_error = "Build the shared FAISS index before running Simple RAG."
        graph_error = "Build the shared FAISS index before running Graph RAG."
        agentic_error = "Build the shared FAISS index before running Agentic RAG."
    else:
        try:
            with st.spinner("Running Simple RAG..."):
                simple_result = simple_rag_answer(
                    query=query,
                    vector_store=vector_store,
                    provider_name=llm_provider,
                    top_k=simple_context_k,
                )
                retrieved_chunks = simple_result["retrieved_chunks"]
        except LLMProviderError as exc:
            simple_error = str(exc)
        except Exception as exc:
            simple_error = f"Simple RAG failed: {exc}"

        try:
            with st.spinner("Running Graph RAG..."):
                graph_result = graph_rag_answer(
                    query=query,
                    chunks=chunks,
                    vector_store=vector_store,
                    provider_name=llm_provider,
                    top_k=top_k,
                )
                graph_preview = graph_result
        except LLMProviderError as exc:
            graph_error = str(exc)
        except Exception as exc:
            graph_error = f"Graph RAG failed: {exc}"

        try:
            with st.spinner("Running Agentic RAG..."):
                agentic_result = agentic_rag_answer(
                    query=query,
                    vector_store=vector_store,
                    provider_name=llm_provider,
                    top_k=top_k,
                )
        except LLMProviderError as exc:
            agentic_error = str(exc)
        except Exception as exc:
            agentic_error = f"Agentic RAG failed: {exc}"

with simple_col:
    st.subheader("Simple RAG")
    st.write("Baseline vector retrieval and answer synthesis.")
    if query.strip() and vector_store is not None:
        st.caption(f"Context budget: {simple_context_k} chunks, matched to Graph RAG.")
    with st.expander("Retrieved chunks", expanded=False):
        if retrieved_chunks:
            for result in retrieved_chunks:
                render_chunk(result)
        elif vector_store is None:
            st.write("Build the shared FAISS index before searching.")
        else:
            st.write("Enter a query to preview FAISS search results.")
    with st.expander("Answer", expanded=True):
        if simple_error:
            st.warning(simple_error)
        elif simple_result:
            st.write(simple_result["answer"])
        else:
            st.write("Run the comparison to synthesize a Simple RAG answer.")

    if simple_result:
        with st.expander("Prompt", expanded=False):
            st.code(simple_result["prompt"], language="text")

with graph_col:
    st.subheader("Graph RAG")
    st.write("Vector retrieval augmented with lightweight entity neighbors.")
    active_graph = graph_result or graph_preview
    if active_graph:
        st.caption(
            f"Context budget: {len(active_graph['augmented_chunks'])} chunks "
            f"({len(active_graph['base_chunks'])} vector + {len(active_graph['graph_chunks'])} graph)."
        )
    with st.expander("Graph relationships", expanded=False):
        if active_graph and active_graph["relationships"]:
            st.dataframe(active_graph["relationships"], use_container_width=True)
        elif active_graph:
            st.write("No matching entity relationships found for this query.")
        elif entity_graph:
            st.write(
                f"Graph ready with {entity_graph.number_of_nodes()} nodes and "
                f"{entity_graph.number_of_edges()} edges."
            )
        else:
            st.write("Ingest documents to build the entity graph.")
    with st.expander("Retrieved chunks", expanded=False):
        if active_graph:
            st.markdown("**Base vector chunks**")
            for result in active_graph["base_chunks"]:
                render_chunk(result)

            st.markdown("**Graph neighbor chunks**")
            if active_graph["graph_chunks"]:
                for result in active_graph["graph_chunks"]:
                    render_chunk(
                        result,
                        show_score=False,
                        note=result.get("graph_reason", "Graph neighbor"),
                    )
            else:
                st.write("No extra chunks were added by graph expansion.")
        elif vector_store is None:
            st.write("Build the shared FAISS index before Graph RAG retrieval.")
        else:
            st.write("Enter a query to preview graph-aware retrieval.")
    with st.expander("Answer", expanded=True):
        if graph_error:
            st.warning(graph_error)
        elif graph_result:
            st.write(graph_result["answer"])
        else:
            st.write("Run the comparison to synthesize a Graph RAG answer.")

    if graph_result:
        with st.expander("Prompt", expanded=False):
            st.code(graph_result["prompt"], language="text")

with agentic_col:
    st.subheader("Agentic RAG")
    st.write("Query decomposition with retrieval per subquestion.")
    with st.expander("Reasoning trace", expanded=False):
        if agentic_result:
            st.markdown("**Subquestions**")
            for item in agentic_result["subquestion_results"]:
                st.write(f"{item['index']}. {item['subquestion']}")

            st.markdown("**Decomposition model output**")
            st.write(agentic_result["decomposition"]["raw_response"])
        elif agentic_error:
            st.warning(agentic_error)
        elif vector_store is None:
            st.write("Build the shared FAISS index before Agentic RAG retrieval.")
        else:
            st.write("Run the comparison to generate subquestions and retrieval steps.")
    with st.expander("Retrieved chunks", expanded=False):
        if agentic_result:
            for item in agentic_result["subquestion_results"]:
                st.markdown(f"**{item['index']}. {item['subquestion']}**")
                if item["retrieved_chunks"]:
                    for result in item["retrieved_chunks"]:
                        render_chunk(result)
                else:
                    st.write("No chunks retrieved for this subquestion.")
        elif vector_store is None:
            st.write("Build the shared FAISS index before searching.")
        else:
            st.write("Run the comparison to retrieve per subquestion.")
    with st.expander("Answer", expanded=True):
        if agentic_error:
            st.warning(agentic_error)
        elif agentic_result:
            st.write(agentic_result["answer"])
        else:
            st.write("Run the comparison to synthesize an Agentic RAG answer.")

    if agentic_result:
        with st.expander("Prompts", expanded=False):
            st.markdown("**Decomposition prompt**")
            st.code(agentic_result["decomposition"]["prompt"], language="text")
            st.markdown("**Synthesis prompt**")
            st.code(agentic_result["synthesis_prompt"], language="text")

if run_comparison and (simple_result or graph_result or agentic_result):
    st.toast("RAG comparison updated.")
