# RAG Architecture Comparison

Lightweight Streamlit app for comparing three retrieval-augmented generation patterns:

- Simple RAG
- Graph RAG
- Agentic RAG

The app is designed to reuse one ingestion pipeline and one FAISS vector index across all three approaches.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set one or both API keys:

```bash
export GEMINI_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
```

Run the app:

```bash
streamlit run app.py
```

## Scope

This project intentionally avoids Docker, Neo4j, agent frameworks, and enterprise-style architecture. The goal is a beginner-friendly comparison that can be implemented and understood quickly.

## Current Features

- Shared ingestion for `.txt`, `.md`, and `.pdf` files
- Shared FAISS vector index
- Configurable embedding model selection
- Configurable Gemini/OpenAI provider selection
- Simple RAG answer synthesis
- Graph RAG with lightweight NetworkX entity relationships
- Agentic RAG with query decomposition and per-subquestion retrieval
- Side-by-side Streamlit comparison UI

## Usage Notes

1. Upload files or keep the sample document enabled.
2. Click **Build / rebuild shared index** after changing documents or the embedding model.
3. Enter a query and click **Run comparison**.
4. Review retrieved chunks, graph relationships, reasoning trace, and prompts in each column.

The first run for an embedding model may download model files from Hugging Face. Subsequent runs should reuse the local cache.
