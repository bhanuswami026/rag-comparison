# Project Overview

This project compares:
1. Simple RAG
2. Graph RAG
3. Agentic RAG

# Goals

- Lightweight implementation
- Achievable in 3-4 hours
- Shared ingestion pipeline
- Shared FAISS vector database
- Configurable embedding models
- Configurable LLMs
- Side-by-side comparison UI

# Tech Stack

- Streamlit
- FAISS
- sentence-transformers
- Gemini API
- OpenAI API
- NetworkX

# Constraints

- Avoid overengineering
- Avoid enterprise architecture
- Avoid Docker
- Avoid Neo4j
- Keep code modular and beginner friendly

# Architecture Requirements

The app must:
- show Simple RAG, Graph RAG, and Agentic RAG side-by-side
- allow model selection from UI
- allow embedding selection from UI
- reuse the same embeddings/index where possible
- keep retrieval strategies modular

# Graph RAG

Graph RAG should:
- use lightweight entity relationships
- use NetworkX
- augment vector retrieval using neighboring entities

# Agentic RAG

Agentic RAG should:
- decompose queries into subquestions
- retrieve per subquestion
- synthesize final answer
- show reasoning steps

# UI

The UI should:
- use Streamlit columns for side-by-side comparison
- show retrieved chunks
- show graph relationships
- show reasoning traces