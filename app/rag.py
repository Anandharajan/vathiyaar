import argparse
import os
from typing import Iterable, List, Sequence, Tuple

try:  # pragma: no cover - optional heavy deps
    import chromadb
except Exception:  # pragma: no cover
    chromadb = None  # type: ignore[assignment]

try:  # pragma: no cover - optional heavy deps
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None  # type: ignore[assignment]

try:  # pragma: no cover - optional heavy deps
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore[assignment]

from app.config import CFG

_EMBED = None
_CLIENT = None


def _ensure_client():
    global _CLIENT
    if chromadb is None:
        raise RuntimeError("chromadb is required for retrieval. Install chromadb to enable RAG.")
    if _CLIENT is None:
        _CLIENT = chromadb.PersistentClient(path=CFG["chroma_dir"])
    return _CLIENT


def _ensure_embedder():
    global _EMBED
    if SentenceTransformer is None:
        raise RuntimeError(
            "sentence-transformers is required for embeddings. Install sentence-transformers to enable RAG."
        )
    if _EMBED is None:
        _EMBED = SentenceTransformer(CFG["embeddings"])
    return _EMBED


def _collection():
    return _ensure_client().get_or_create_collection("study_notes")


def _split_text(text: str, chunk: int = 800, overlap: int = 120) -> List[str]:
    clean = " ".join(text.split())
    output: List[str] = []
    cursor = 0
    step = max(chunk - overlap, 1)
    while cursor < len(clean):
        output.append(clean[cursor : cursor + chunk])
        cursor += step
    return output


def _embed(texts: Sequence[str]) -> List[List[float]]:
    if not texts:
        return []
    embedder = _ensure_embedder()
    return embedder.encode(texts, normalize_embeddings=True).tolist()


def ingest() -> None:
    ids: List[str] = []
    docs: List[str] = []
    metas: List[dict] = []

    os.makedirs(CFG["docs_dir"], exist_ok=True)
    if PdfReader is None:
        raise RuntimeError("pypdf is required to ingest documents. Install pypdf to enable ingestion.")
    for filename in os.listdir(CFG["docs_dir"]):
        if not filename.lower().endswith(".pdf"):
            continue
        path = os.path.join(CFG["docs_dir"], filename)
        pdf = PdfReader(path)
        for page_index, page in enumerate(pdf.pages):
            raw = page.extract_text() or ""
            for chunk_index, chunk in enumerate(_split_text(raw)):
                ids.append(f"{filename}-{page_index}-{chunk_index}")
                docs.append(chunk)
                metas.append({"source": filename, "page": page_index + 1})

    if ids:
        collection = _collection()
        collection.upsert(
            ids=ids,
            documents=docs,
            metadatas=metas,
            embeddings=_embed(docs),
        )
        print(f"Indexed {len(ids)} chunks.")
    else:
        print("No PDFs found to index.")


def query(question: str, k: int = 4) -> List[Tuple[str, dict]]:
    if not question.strip():
        return []
    collection = _collection()
    result = collection.query(
        query_texts=[question],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    documents: Iterable[str] = result.get("documents", [[]])[0]
    metadatas: Iterable[dict] = result.get("metadatas", [[]])[0]
    return list(zip(documents, metadatas))


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="drop existing collection before ingesting")
    args = parser.parse_args()

    if args.rebuild:
        try:
            _ensure_client().delete_collection("study_notes")
        except Exception:
            pass
    _collection()
    ingest()


if __name__ == "__main__":
    _main()
