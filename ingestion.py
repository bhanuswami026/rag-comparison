"""Shared document loading, chunking, and entity extraction."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}
ENTITY_STOPWORDS = {
    "A",
    "An",
    "And",
    "As",
    "By",
    "For",
    "From",
    "In",
    "It",
    "Of",
    "On",
    "Or",
    "The",
    "This",
    "To",
    "Using",
    "With",
}


def normalize_text(text: str) -> str:
    """Collapse noisy whitespace while keeping paragraph boundaries readable."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_text_bytes(content: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="ignore")


def read_pdf_bytes(content: bytes) -> str:
    import fitz

    pages = []
    with fitz.open(stream=content, filetype="pdf") as document:
        for page in document:
            pages.append(page.get_text())
    return "\n\n".join(pages)


def read_uploaded_document(uploaded_file: Any) -> dict[str, str]:
    name = uploaded_file.name
    extension = Path(name).suffix.lower()
    content = uploaded_file.getvalue()
    text = read_document_bytes(content, name)
    return {"source": name, "text": text, "type": extension.lstrip(".")}


def read_document_bytes(content: bytes, source_name: str) -> str:
    extension = Path(source_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {extension or 'unknown'}")
    if extension == ".pdf":
        return normalize_text(read_pdf_bytes(content))
    return normalize_text(read_text_bytes(content))


def load_sample_documents(sample_dir: str | Path = "sample_docs") -> list[dict[str, str]]:
    sample_path = Path(sample_dir)
    documents = []
    for path in sorted(sample_path.glob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            text = read_document_bytes(path.read_bytes(), path.name)
            documents.append(
                {
                    "source": path.name,
                    "text": text,
                    "type": path.suffix.lower().lstrip("."),
                }
            )
    return documents


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = normalize_text(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        if end < text_length:
            boundary = text.rfind(" ", start, end)
            if boundary > start + chunk_size // 2:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = max(end - chunk_overlap, start + 1)
        while start < text_length and text[start].isspace():
            start += 1

    return chunks


def extract_semantic_triplets(text: str, max_triplets: int = 10) -> list[dict[str, str]]:
    """Extract semantic (subject, relation, object) triplets from text using clause and relation pattern matching."""
    rel_patterns = [
        r"(?P<sub>\b[A-Z][a-zA-Z0-9_\-\s]{1,30}\b)\s+(?P<rel>uses|utilizes|supports|indexes|builds|augments|decomposes|retrieves|synthesizes|compares|integrates|provides|contains|implements|requires|connects|generates|features)\s+(?P<obj>\b[A-Z][a-zA-Z0-9_\-\s]{1,30}\b)",
        r"(?P<sub>\b[A-Z0-9_\-]{2,}\b)\s+(?P<rel>uses|utilizes|supports|indexes|builds|augments|decomposes|retrieves|synthesizes|compares|integrates|provides|contains|implements|requires|connects|generates)\s+(?P<obj>\b[A-Z0-9_\-]{2,}\b)",
    ]
    
    triplets = []
    seen = set()
    
    for pattern in rel_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            sub = match.group("sub").strip()
            rel = match.group("rel").strip().upper()
            obj = match.group("obj").strip()
            
            if sub in ENTITY_STOPWORDS or obj in ENTITY_STOPWORDS:
                continue
            if len(sub) < 2 or len(obj) < 2 or sub.lower() == obj.lower():
                continue
                
            key = (sub.lower(), rel, obj.lower())
            if key not in seen:
                seen.add(key)
                triplets.append({"subject": sub, "relation": rel, "object": obj})
            if len(triplets) >= max_triplets:
                break
        if len(triplets) >= max_triplets:
            break
            
    # Fallback to semantic co-occurrence with RELATED_TO if few explicit relations found
    if len(triplets) < 2:
        entities = extract_entities(text)
        for i in range(len(entities) - 1):
            sub = entities[i]
            obj = entities[i + 1]
            key = (sub.lower(), "RELATED_TO", obj.lower())
            if key not in seen:
                seen.add(key)
                triplets.append({"subject": sub, "relation": "RELATED_TO", "object": obj})
            if len(triplets) >= max_triplets:
                break

    return triplets


def extract_entities(text: str, max_entities: int = 12) -> list[str]:
    acronym_pattern = r"\b[A-Z]{2,}(?:-[A-Z0-9]+)*\b"
    title_pattern = r"\b[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+){0,3}\b"
    candidates = re.findall(f"{acronym_pattern}|{title_pattern}", text)

    entities = []
    seen = set()
    for candidate in candidates:
        entity = candidate.strip()
        if entity in ENTITY_STOPWORDS or len(entity) < 2:
            continue
        key = entity.lower()
        if key not in seen:
            seen.add(key)
            entities.append(entity)
        if len(entities) >= max_entities:
            break

    return entities


def build_chunks(
    documents: list[dict[str, str]],
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict[str, Any]]:
    chunks = []
    for document in documents:
        for local_id, text in enumerate(chunk_text(document["text"], chunk_size, chunk_overlap)):
            chunk_id = len(chunks)
            triplets = extract_semantic_triplets(text)
            entities = extract_entities(text)
            
            # Combine entities from triplets and direct extraction
            for triplet in triplets:
                if triplet["subject"] not in entities:
                    entities.append(triplet["subject"])
                if triplet["object"] not in entities:
                    entities.append(triplet["object"])

            chunks.append(
                {
                    "id": chunk_id,
                    "source": document["source"],
                    "source_type": document["type"],
                    "local_id": local_id,
                    "text": text,
                    "entities": entities,
                    "triplets": triplets,
                }
            )
    return chunks


def ingest_documents(
    uploaded_files: list[Any] | None,
    use_sample_doc: bool,
    chunk_size: int,
    chunk_overlap: int,
    sample_dir: str | Path = "sample_docs",
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    documents = []

    if use_sample_doc:
        documents.extend(load_sample_documents(sample_dir))

    for uploaded_file in uploaded_files or []:
        documents.append(read_uploaded_document(uploaded_file))

    documents = [document for document in documents if document["text"]]
    chunks = build_chunks(documents, chunk_size, chunk_overlap)
    return documents, chunks
