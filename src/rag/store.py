"""Connect to an existing local vector collection without creating an empty index."""
from pathlib import Path
import chromadb
from rag.ingest import DEFAULT_COLLECTION, DEFAULT_PERSIST_DIR
from rag.retrieval import QueryCollection


def local_collection(
    persist_dir: Path = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION,
) -> QueryCollection:
    if not persist_dir.exists():
        raise FileNotFoundError(f"Vector index not found in {persist_dir}. Build it first with finnie-index-pdfs.")
    client = chromadb.PersistentClient(path=str(persist_dir))
    names = {item if isinstance(item, str) else item.name for item in client.list_collections()}
    if collection_name not in names:
        raise FileNotFoundError(f"Vector collection '{collection_name}' was not found in {persist_dir}. Build it first with finnie-index-pdfs.")
    return client.get_collection(name=collection_name)
