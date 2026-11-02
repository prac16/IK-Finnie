"""Build a persistent Chroma vector index from local PDF files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterator

import chromadb
from pypdf import PdfReader

from rag.categories import CATEGORIES, infer_category

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge_base"
DEFAULT_PDF_DIR = DEFAULT_KNOWLEDGE_DIR / "documents"
DEFAULT_PERSIST_DIR = DEFAULT_KNOWLEDGE_DIR / "index"
DEFAULT_COLLECTION = "finnie_pdfs"
BATCH_SIZE = 128


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping chunks, preferring word boundaries."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")

    text = " ".join(text.split())
    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        next_start = max(end - chunk_overlap, start + 1)
        boundary = text.rfind(" ", start, next_start + 1)
        start = boundary + 1 if boundary >= start else next_start

    return chunks


def iter_pdf_chunks(
    pdf_path: Path, chunk_size: int, chunk_overlap: int, category: str | None = None
) -> Iterator[tuple[str, dict[str, str | int], str]]:
    """Yield stable IDs, metadata, and text for each page chunk in a PDF."""
    reader = PdfReader(str(pdf_path))
    source = pdf_path.name

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        if not page_text:
            continue
        for chunk_number, text in enumerate(
            split_text(page_text, chunk_size, chunk_overlap)
        ):
            key = f"{source}:{page_number}:{chunk_number}"
            chunk_id = hashlib.sha256(key.encode("utf-8")).hexdigest()
            metadata: dict[str, str | int] = {
                "source": source,
                "page": page_number,
                "chunk": chunk_number,
                "category": category or infer_category(source),
            }
            yield chunk_id, metadata, text


def index_pdfs(
    pdf_dir: Path,
    persist_dir: Path,
    collection_name: str = DEFAULT_COLLECTION,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    category_overrides: dict[str, str] | None = None,
) -> int:
    """Upsert all readable page chunks from a directory of PDFs."""
    if not pdf_dir.is_dir():
        raise FileNotFoundError(f"PDF directory does not exist: {pdf_dir}")
    if not collection_name.strip():
        raise ValueError("collection_name must not be empty")
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunking requires chunk_size > chunk_overlap >= 0")

    category_overrides = category_overrides or {}
    if any(value not in CATEGORIES for value in category_overrides.values()):
        raise ValueError(f"Categories must be one of {CATEGORIES}")

    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in: {pdf_dir}")

    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    count = 0
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, str | int]] = []

    def flush_batch() -> None:
        nonlocal count
        if not ids:
            return
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        count += len(ids)
        ids.clear()
        documents.clear()
        metadatas.clear()

    for pdf_path in pdf_paths:
        for chunk_id, metadata, text in iter_pdf_chunks(
            pdf_path, chunk_size, chunk_overlap, category_overrides.get(pdf_path.name)
        ):
            ids.append(chunk_id)
            metadatas.append(metadata)
            documents.append(text)
            if len(ids) >= BATCH_SIZE:
                flush_batch()
    flush_batch()
    if count == 0:
        raise ValueError(f"No extractable text found in PDFs under: {pdf_dir}")
    return count


def update_categories(
    persist_dir: Path,
    collection_name: str = DEFAULT_COLLECTION,
    category_overrides: dict[str, str] | None = None,
) -> int:
    """Backfill category metadata without changing text or recomputing embeddings."""
    overrides = category_overrides or {}
    if any(value not in CATEGORIES for value in overrides.values()):
        raise ValueError(f"Categories must be one of {CATEGORIES}")
    collection = chromadb.PersistentClient(path=str(persist_dir)).get_collection(collection_name)
    updated = 0
    for offset in range(0, collection.count(), BATCH_SIZE):
        batch = collection.get(limit=BATCH_SIZE, offset=offset, include=["metadatas"])
        ids, metadatas = [], []
        for chunk_id, metadata in zip(batch["ids"], batch["metadatas"]):
            if not metadata or not isinstance(metadata.get("source"), str):
                continue
            category = overrides.get(metadata["source"], metadata.get("category"))
            if category not in CATEGORIES:
                category = infer_category(metadata["source"])
            ids.append(chunk_id)
            metadatas.append({**metadata, "category": category})
        if ids:
            collection.update(ids=ids, metadatas=metadatas)
            updated += len(ids)
    return updated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Index PDFs into a persistent local Chroma vector database."
    )
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--persist-dir", type=Path, default=DEFAULT_PERSIST_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument("--category-map", type=Path, help="JSON object mapping PDF filenames to categories")
    parser.add_argument("--categories-only", action="store_true", help="Update existing category metadata without re-embedding")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    overrides = json.loads(args.category_map.read_text()) if args.category_map else None
    if overrides is not None and not isinstance(overrides, dict):
        raise ValueError("category-map must contain a JSON object")
    if args.categories_only:
        count = update_categories(args.persist_dir, args.collection, overrides)
        print(f"Updated categories for {count} chunks")
        return
    count = index_pdfs(
        pdf_dir=args.pdf_dir,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        category_overrides=overrides,
    )
    print(f"Indexed {count} PDF chunks into {args.persist_dir / args.collection}")


if __name__ == "__main__":
    main()
