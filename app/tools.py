"""Application tools. Tool outputs retain source metadata for reliable citations."""
import json
from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from app.config import (EMPLOYEES_FILE, TICKETS_FILE, CHROMA_DIR,
                        COLLECTION_NAME, EMBEDDING_MODEL, SIMILARITY_THRESHOLD)

_store = None

def get_store():
    global _store
    if _store is None:
        if not CHROMA_DIR.exists():
            raise RuntimeError("Vector index not found. Run: python -m app.ingestion")
        _store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(CHROMA_DIR),
            embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL),
        )
    return _store

def get_employee_profile(employee_id: str) -> dict:
    if not EMPLOYEES_FILE.exists():
        raise RuntimeError("employees.json is missing")
    data = json.loads(EMPLOYEES_FILE.read_text(encoding="utf-8"))
    employees = data.get("employees", data)
    if employee_id not in employees:
        raise ValueError(f"Unknown employee_id: {employee_id}")
    profile = employees[employee_id]
    return {"employee_id": employee_id, **profile}

def search_policies(query: str, country: Optional[str] = None,
                    doc_type: Optional[str] = None, k: int = 6) -> list[dict]:
    store = get_store()
    where = None
    filters = []
    if country:
        filters.append({"country": country})
    if doc_type:
        filters.append({"doc_type": doc_type})
    if len(filters) == 1:
        where = filters[0]
    elif len(filters) > 1:
        where = {"$and": filters}
    # Chroma returns distance; lower is closer. Convert to a bounded relevance score.
    results = store.similarity_search_with_score(query, k=k, filter=where)
    output = []
    for doc, distance in results:
        score = 1.0 / (1.0 + max(float(distance), 0.0))
        output.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source"),
            "page": doc.metadata.get("page"),
            "section": doc.metadata.get("section"),
            "country": doc.metadata.get("country"),
            "doc_type": doc.metadata.get("doc_type"),
            "effective_date": doc.metadata.get("effective_date"),
            "relevance": score,
        })
    return output

def create_helpdesk_ticket(category: str, summary: str) -> str:
    path = Path(TICKETS_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        tickets = json.loads(path.read_text(encoding="utf-8"))
    else:
        tickets = []
    next_num = len(tickets) + 1
    ticket_id = f"HD-{1000 + next_num}"
    tickets.append({"ticket_id": ticket_id, "category": category,
                    "summary": summary, "status": "Open"})
    path.write_text(json.dumps(tickets, indent=2), encoding="utf-8")
    return ticket_id
