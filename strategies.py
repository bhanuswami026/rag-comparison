"""Retrieval and answer strategies for the comparison app."""

from __future__ import annotations

import re
from typing import Any

import networkx as nx

from ingestion import extract_entities
from llm_providers import generate_answer


def format_context(chunks: list[dict[str, Any]]) -> str:
    context_blocks = []
    for chunk in chunks:
        score = chunk.get("score")
        score_text = f"{score:.3f}" if isinstance(score, (int, float)) else "graph-expanded"
        context_blocks.append(
            "\n".join(
                [
                    f"Source: {chunk['source']}",
                    f"Chunk ID: {chunk['id']}",
                    f"Similarity score: {score_text}",
                    chunk["text"],
                ]
            )
        )
    return "\n\n---\n\n".join(context_blocks)


def build_simple_rag_prompt(query: str, retrieved_chunks: list[dict[str, Any]]) -> str:
    context = format_context(retrieved_chunks)
    return f"""You are answering a research-project question using retrieved document chunks.

Use only the context below. If the context is not enough, say what is missing.
Keep the answer concise and cite the source filenames or chunk IDs you used.

Question:
{query}

Retrieved context:
{context}

Answer:"""


def simple_rag_answer(
    query: str,
    vector_store: Any,
    provider_name: str,
    top_k: int,
) -> dict[str, Any]:
    retrieved_chunks = vector_store.search(query, top_k=top_k)
    if not retrieved_chunks:
        return {
            "answer": "No relevant chunks were retrieved.",
            "retrieved_chunks": [],
            "prompt": "",
        }

    prompt = build_simple_rag_prompt(query, retrieved_chunks)
    answer = generate_answer(provider_name, prompt)
    return {
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "prompt": prompt,
    }


def chunk_node_id(chunk_id: int) -> str:
    return f"chunk:{chunk_id}"


def entity_node_id(entity: str) -> str:
    return f"entity:{entity.lower()}"


def build_entity_graph(chunks: list[dict[str, Any]]) -> nx.Graph:
    graph = nx.Graph()

    for chunk in chunks:
        chunk_id = chunk_node_id(chunk["id"])
        graph.add_node(
            chunk_id,
            label=f"Chunk {chunk['id']}",
            kind="chunk",
            chunk_id=chunk["id"],
            source=chunk["source"],
        )

        entities = chunk.get("entities", [])
        for entity in entities:
            entity_id = entity_node_id(entity)
            graph.add_node(entity_id, label=entity, kind="entity")
            graph.add_edge(chunk_id, entity_id, relation="mentions")

        triplets = chunk.get("triplets", [])
        for triplet in triplets:
            sub_id = entity_node_id(triplet["subject"])
            obj_id = entity_node_id(triplet["object"])
            rel = triplet["relation"]
            
            graph.add_node(sub_id, label=triplet["subject"], kind="entity")
            graph.add_node(obj_id, label=triplet["object"], kind="entity")
            graph.add_edge(sub_id, obj_id, relation=rel, chunk_id=chunk["id"])

        if not triplets:
            for left_index, left_entity in enumerate(entities):
                for right_entity in entities[left_index + 1 :]:
                    graph.add_edge(
                        entity_node_id(left_entity),
                        entity_node_id(right_entity),
                        relation="CO-OCCURS",
                        chunk_id=chunk["id"],
                    )

    return graph


def graph_relationship_rows(
    graph: nx.Graph,
    seed_entities: list[str],
    max_rows: int = 20,
) -> list[dict[str, str]]:
    rows = []
    seen_edges = set()

    for entity in seed_entities:
        entity_id = entity_node_id(entity)
        if entity_id not in graph:
            continue

        for neighbor_id in graph.neighbors(entity_id):
            edge_key = tuple(sorted([entity_id, neighbor_id]))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            edge_data = graph.get_edge_data(entity_id, neighbor_id) or {}
            rel = edge_data.get("relation", "RELATED_TO")
            if rel == "mentions":
                continue

            rows.append(
                {
                    "from": graph.nodes[entity_id].get("label", entity_id),
                    "relation": rel,
                    "to": graph.nodes[neighbor_id].get("label", neighbor_id),
                }
            )
            if len(rows) >= max_rows:
                return rows

    if not rows:
        for u, v, data in graph.edges(data=True):
            rel = data.get("relation", "RELATED_TO")
            if rel == "mentions":
                continue
            edge_key = tuple(sorted([u, v]))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            rows.append(
                {
                    "from": graph.nodes[u].get("label", u),
                    "relation": rel,
                    "to": graph.nodes[v].get("label", v),
                }
            )
            if len(rows) >= max_rows:
                break

    return rows


def graph_augmented_retrieval(
    query: str,
    chunks: list[dict[str, Any]],
    vector_store: Any,
    top_k: int,
    max_extra_chunks: int = 3,
) -> dict[str, Any]:
    graph = build_entity_graph(chunks)
    base_chunks = vector_store.search(query, top_k=top_k)
    query_entities = extract_entities(query)
    seed_entities = []

    for entity in query_entities:
        if entity not in seed_entities:
            seed_entities.append(entity)

    for chunk in base_chunks:
        for entity in chunk.get("entities", []):
            if entity not in seed_entities:
                seed_entities.append(entity)

    chunk_by_id = {chunk["id"]: chunk for chunk in chunks}
    base_ids = {chunk["id"] for chunk in base_chunks}
    extra_chunks = []

    for entity in seed_entities:
        entity_id = entity_node_id(entity)
        if entity_id not in graph:
            continue

        for neighbor_id in graph.neighbors(entity_id):
            neighbor = graph.nodes[neighbor_id]

            if neighbor.get("kind") == "chunk":
                candidate_id = neighbor["chunk_id"]
                if candidate_id not in base_ids and candidate_id in chunk_by_id:
                    extra_chunks.append(
                        {
                            **chunk_by_id[candidate_id],
                            "score": None,
                            "rank": len(base_chunks) + len(extra_chunks) + 1,
                            "graph_reason": f"Neighbor of entity '{entity}'",
                        }
                    )

            if neighbor.get("kind") == "entity":
                for second_hop_id in graph.neighbors(neighbor_id):
                    second_hop = graph.nodes[second_hop_id]
                    if second_hop.get("kind") != "chunk":
                        continue
                    candidate_id = second_hop["chunk_id"]
                    if candidate_id not in base_ids and candidate_id in chunk_by_id:
                        extra_chunks.append(
                            {
                                **chunk_by_id[candidate_id],
                                "score": None,
                                "rank": len(base_chunks) + len(extra_chunks) + 1,
                                "graph_reason": (
                                    f"Connected through entity "
                                    f"'{graph.nodes[neighbor_id].get('label', neighbor_id)}'"
                                ),
                            }
                        )

            seen_extra_ids = {chunk["id"] for chunk in extra_chunks}
            if len(seen_extra_ids) >= max_extra_chunks:
                break

        if len({chunk["id"] for chunk in extra_chunks}) >= max_extra_chunks:
            break

    deduped_extra_chunks = []
    seen_ids = set(base_ids)
    for chunk in extra_chunks:
        if chunk["id"] in seen_ids:
            continue
        seen_ids.add(chunk["id"])
        deduped_extra_chunks.append(chunk)
        if len(deduped_extra_chunks) >= max_extra_chunks:
            break

    augmented_chunks = base_chunks + deduped_extra_chunks
    return {
        "graph": graph,
        "base_chunks": base_chunks,
        "graph_chunks": deduped_extra_chunks,
        "augmented_chunks": augmented_chunks,
        "seed_entities": seed_entities,
        "relationships": graph_relationship_rows(graph, seed_entities),
    }


def build_graph_rag_prompt(
    query: str,
    augmented_chunks: list[dict[str, Any]],
    relationships: list[dict[str, str]],
) -> str:
    context = format_context(augmented_chunks)
    relationship_text = "\n".join(
        f"- {row['from']} --{row['relation']}--> {row['to']}" for row in relationships
    )
    if not relationship_text:
        relationship_text = "No graph relationships were available."

    return f"""You are answering a question using Graph RAG.

Synthesize a clear, direct, and well-structured answer using the retrieved document chunks and entity relationships provided below.
Do not invent details outside the context. Cite source filenames or chunk IDs where appropriate.

Question:
{query}

Entity relationships:
{relationship_text}

Retrieved context:
{context}

Answer:"""


def graph_rag_answer(
    query: str,
    chunks: list[dict[str, Any]],
    vector_store: Any,
    provider_name: str,
    top_k: int,
) -> dict[str, Any]:
    retrieval = graph_augmented_retrieval(
        query=query,
        chunks=chunks,
        vector_store=vector_store,
        top_k=top_k,
    )

    if not retrieval["augmented_chunks"]:
        return {
            "answer": "No relevant chunks were retrieved.",
            "prompt": "",
            **retrieval,
        }

    prompt = build_graph_rag_prompt(
        query=query,
        augmented_chunks=retrieval["augmented_chunks"],
        relationships=retrieval["relationships"],
    )
    answer = generate_answer(provider_name, prompt)
    return {
        "answer": answer,
        "prompt": prompt,
        **retrieval,
    }


def parse_subquestions(raw_text: str, original_query: str, max_questions: int = 4) -> list[str]:
    candidates = []
    for line in raw_text.splitlines():
        cleaned = re.sub(r"^\s*[-*]?\s*\d*[\).:-]?\s*", "", line).strip()
        if cleaned:
            candidates.append(cleaned)

    if not candidates:
        candidates = [part.strip() for part in re.split(r"[?;]\s+", raw_text) if part.strip()]

    subquestions = []
    seen = set()
    for candidate in candidates:
        candidate = candidate.strip(" -")
        if not candidate:
            continue
        if not candidate.endswith("?"):
            candidate = f"{candidate}?"
        key = candidate.lower()
        if key not in seen:
            seen.add(key)
            subquestions.append(candidate)
        if len(subquestions) >= max_questions:
            break

    return subquestions or [original_query]


def fallback_subquestions(query: str) -> list[str]:
    return [
        query,
        f"What evidence in the documents is most relevant to: {query}",
    ]


def decompose_query(query: str, provider_name: str, max_questions: int = 4) -> dict[str, Any]:
    prompt = f"""You are an advanced RAG query decomposition agent.
Break the following complex user question into 2 to {max_questions} distinct, highly specific, and non-overlapping retrieval subquestions.

Requirements:
- Each subquestion must target a completely different entity, metric, time period, or perspective mentioned in the original question.
- Subquestions must be self-contained and search-ready.
- Return ONLY a numbered list of subquestions. Do not answer them.

Question:
{query}"""

    raw_response = generate_answer(
        provider_name=provider_name,
        prompt=prompt,
        temperature=0.1,
        max_output_tokens=250,
    )
    subquestions = parse_subquestions(raw_response, query, max_questions=max_questions)
    return {
        "prompt": prompt,
        "raw_response": raw_response,
        "subquestions": subquestions,
        "used_fallback": False,
    }


def build_agentic_synthesis_prompt(
    query: str,
    subquestion_results: list[dict[str, Any]],
) -> str:
    trace_blocks = []
    
    for item in subquestion_results:
        trace_blocks.append(
            "\n".join(
                [
                    f"Subquestion {item['index']}: {item['subquestion']}",
                    "Retrieved context:",
                    format_context(item["retrieved_chunks"]),
                ]
            )
        )

    trace_text = "\n\n===\n\n".join(trace_blocks)
    return f"""You are synthesizing an Agentic RAG answer.

Synthesize a clear, comprehensive, and direct answer using the subquestion evidence below.
Do not invent details outside the context. Cite source filenames or chunk IDs where appropriate.

Original question:
{query}

Subquestion retrieval trace:
{trace_text}

Answer:"""


def agentic_rag_answer(
    query: str,
    vector_store: Any,
    provider_name: str,
    top_k: int,
) -> dict[str, Any]:
    decomposition = decompose_query(query, provider_name)
    subquestions = decomposition["subquestions"]

    subquestion_results = []
    
    for index, subquestion in enumerate(subquestions, start=1):
        retrieved_chunks = vector_store.search(subquestion, top_k=top_k)
        subquestion_results.append(
            {
                "index": index,
                "subquestion": subquestion,
                "retrieved_chunks": retrieved_chunks,
            }
        )

    synthesis_prompt = build_agentic_synthesis_prompt(query, subquestion_results)
    answer = generate_answer(
        provider_name=provider_name,
        prompt=synthesis_prompt,
        temperature=0.2,
        max_output_tokens=700,
    )

    return {
        "answer": answer,
        "decomposition": decomposition,
        "subquestion_results": subquestion_results,
        "synthesis_prompt": synthesis_prompt,
    }
