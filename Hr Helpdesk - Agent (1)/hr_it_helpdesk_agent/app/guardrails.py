"""Deterministic guardrails applied before/after LLM generation."""
from app.config import FALLBACK, SIMILARITY_THRESHOLD

def relevant(results):
    return [r for r in results if r.get("relevance", 0) >= SIMILARITY_THRESHOLD]

def citations(results):
    out = []
    for r in results:
        location = f"page {r['page']}" if r.get("page") else (r.get("section") or "section unavailable")
        label = f"{r.get('source')} - {location}"
        if label not in out:
            out.append(label)
    return out

def latest_policy(results):
    """Drop superseded versions while retaining every chunk from the winning version."""
    latest_by_family = {}
    for r in results:
        src = (r.get("source") or "").lower()
        family = "leave" if "leave-policy" in src else src
        date = r.get("effective_date") or "0000-00-00"
        if family not in latest_by_family or date > latest_by_family[family]:
            latest_by_family[family] = date
    kept = []
    for r in results:
        src = (r.get("source") or "").lower()
        family = "leave" if "leave-policy" in src else src
        date = r.get("effective_date") or "0000-00-00"
        if date == latest_by_family.get(family):
            kept.append(r)
    return kept

def safe_fallback():
    return FALLBACK
