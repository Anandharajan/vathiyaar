import argparse
import os
from typing import Iterable, List, Sequence, Tuple

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from app.config import CFG

_EMBED = SentenceTransformer(CFG["embeddings"])
_CLIENT = chromadb.PersistentClient(path=CFG["chroma_dir"])


def _collection():
    return _CLIENT.get_or_create_collection("study_notes")


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
    return _EMBED.encode(texts, normalize_embeddings=True).tolist()


def ingest() -> None:
    ids: List[str] = []
    docs: List[str] = []
    metas: List[dict] = []

    os.makedirs(CFG["docs_dir"], exist_ok=True)
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
            _CLIENT.delete_collection("study_notes")
        except Exception:
            pass
    _collection()
    ingest()


if __name__ == "__main__":
    _main()
