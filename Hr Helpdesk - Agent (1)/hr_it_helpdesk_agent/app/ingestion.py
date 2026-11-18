"""Idempotent PDF/Markdown ingestion with page/section metadata."""
import argparse
import hashlib
import re
from pathlib import Path
from typing import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import DATA_DIR, CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL

def _markdown_sections(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    sections, heading, buf = [], "Introduction", []
    for line in text.splitlines():
        if line.startswith("#"):
            if buf:
                sections.append({"text": "\n".join(buf).strip(), "section": heading})
            heading = line.lstrip("#").strip() or "Untitled"
            buf = [line]
        else:
            buf.append(line)
    if buf:
        sections.append({"text": "\n".join(buf).strip(), "section": heading})
    return [s for s in sections if s["text"]]

def _pdf_pages(path: Path) -> list[dict]:
    import fitz
    doc = fitz.open(path)
    return [{"text": page.get_text("text"), "page": i + 1, "section": None}
            for i, page in enumerate(doc) if page.get_text("text").strip()]

def load_documents(data_dir: Path = DATA_DIR) -> list[dict]:
    docs = []
    for path in sorted(data_dir.iterdir()):
        if path.suffix.lower() == ".md":
            records = _markdown_sections(path)
        elif path.suffix.lower() == ".pdf":
            records = _pdf_pages(path)
        else:
            continue
        for record in records:
            docs.append({
                **record,
                "source": path.name,
                "country": _country_from_name(path.name),
                "doc_type": _type_from_name(path.name),
                "effective_date": _effective_date(path.name),
            })
    return docs

def _country_from_name(name: str):
    n = name.lower()
    if "sri-lanka" in n: return "Sri Lanka"
    if "india" in n: return "India"
    return None

def _type_from_name(name: str):
    n = name.lower()
    if "leave" in n: return "leave"
    if "travel" in n: return "travel"
    if "laptop" in n: return "laptop"
    if "it-helpdesk" in n: return "it"
    return "general"

def _effective_date(name: str):
    match = re.search(r"(20\d{2})", name)
    return f"{match.group(1)}-01-01" if match else None

def chunk_records(records: Iterable[dict], chunk_size=700, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "]
    )
    chunks = []
    for rec in records:
        # Keep each PDF page and Markdown section as a hard boundary.
        for chunk in splitter.split_text(rec["text"]):
            if not chunk.strip():
                continue
            metadata = {k: rec.get(k) for k in
                        ("source", "page", "section", "country", "doc_type", "effective_date")
                        if rec.get(k) is not None}
            digest = hashlib.sha256((str(metadata) + chunk).encode()).hexdigest()
            metadata["chunk_id"] = digest
            chunks.append((chunk, metadata))
    return chunks

def ingest(data_dir: Path = DATA_DIR, reset=False):
    records = load_documents(data_dir)
    chunks = chunk_records(records)
    if not chunks:
        raise RuntimeError(f"No PDF or Markdown documents found in {data_dir}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    if reset and CHROMA_DIR.exists():
        import shutil
        shutil.rmtree(CHROMA_DIR)
    store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )
    texts = [t for t, _ in chunks]
    metas = [m for _, m in chunks]
    ids = [m["chunk_id"] for m in metas]
    store.add_texts(texts=texts, metadatas=metas, ids=ids)  # deterministic IDs prevent duplicates
    return {"documents": len(records), "chunks": len(chunks), "indexed": len(ids)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    print(ingest(reset=args.reset))
