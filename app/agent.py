"""Controlled RAG agent. Uses OpenAI when configured; deterministic fallback for tests."""
import json
from typing import Optional

from app.config import FALLBACK, MAX_TOOL_CALLS
from app.tools import search_policies, get_employee_profile, create_helpdesk_ticket
from app.guardrails import relevant, citations, latest_policy
from app.memory import memory

def _is_explicit_ticket_request(message: str) -> bool:
    text = message.lower()
    phrases = ("raise a ticket", "create a ticket", "log a ticket",
               "open a ticket", "please ticket", "submit a ticket")
    return any(p in text for p in phrases)

def _country_from_history(history):
    for item in reversed(history):
        text = item["content"].lower()
        if "sri lanka" in text or "sri-lanka" in text:
            return "Sri Lanka"
        if "india" in text:
            return "India"
    return None

def _call_llm(question: str, context: list[dict]) -> Optional[str]:
    from app.config import OPENAI_API_KEY, OPENAI_MODEL
    if not OPENAI_API_KEY:
        return None
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    excerpts = "\n\n".join(
        f"SOURCE: {r['source']} | PAGE: {r.get('page')} | SECTION: {r.get('section')}\n{r['content']}"
        for r in context
    )
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content":
             "Answer ONLY using the supplied company-document excerpts. Do not use outside knowledge. "
             "Do not infer missing approvals, amounts, exceptions or benefits. If evidence is insufficient, "
             f"reply exactly: {FALLBACK}"},
            {"role": "user", "content": f"Question: {question}\n\nDocument excerpts:\n{excerpts}"}
        ]
    )
    return response.choices[0].message.content.strip()

def handle_chat(message: str, employee_id: Optional[str], session_id: str) -> dict:
    history = memory.get(session_id)
    tool_calls, evidence = [], []
    ticket_id = None
    original_question = message

    # Continue a pending country-dependent question after a clarification reply.
    if history and history[-1].get("role") == "assistant" and "confirm" in history[-1]["content"].lower():
        prior_user = next((x["content"] for x in reversed(history[:-1]) if x["role"] == "user"), None)
        if prior_user:
            original_question = prior_user + " " + message

    country = None
    if employee_id:
        try:
            profile = get_employee_profile(employee_id)
            country = profile.get("country")
            tool_calls.append({"tool": "get_employee_profile", "input": {"employee_id": employee_id}})
        except ValueError as exc:
            return {"answer": str(exc), "sources": [], "confidence_flag": "low",
                    "tool_calls": tool_calls, "ticket_id": None}
    if not country:
        country = _country_from_history(history) or _country_from_history([{"content": message}])

    country_dependent = any(w in original_question.lower() for w in
                            ("travel", "allowance", "leave", "approval", "approver", "colombo"))
    if country_dependent and not country:
        answer = "Could you please confirm whether you are based in India or Sri Lanka?"
        memory.add(session_id, "user", message)
        memory.add(session_id, "assistant", answer)
        return {"answer": answer, "sources": [], "confidence_flag": "partial",
                "tool_calls": tool_calls, "ticket_id": None}

    if len(tool_calls) >= MAX_TOOL_CALLS:
        return {"answer": FALLBACK, "sources": [], "confidence_flag": "low",
                "tool_calls": tool_calls, "ticket_id": None}

    q = original_question
    qlow = q.lower()
    # Split cross-policy questions into two distinct searches.
    queries = []
    if any(x in qlow for x in ("business trip", "business travel", "colombo")) and "leave" in qlow:
        queries = [
            ("leave approval", country, "leave"),
            ("Colombo business travel approval", "Sri Lanka", "travel"),
        ]
    else:
        doc_type = ("leave" if "leave" in qlow or "carry forward" in qlow else
                    "travel" if "travel" in qlow or "allowance" in qlow or "colombo" in qlow else
                    "laptop" if "laptop" in qlow else "it" if any(x in qlow for x in ("password", "vpn", "login")) else None)
        queries = [(q, country, doc_type)]

    for query, query_country, doc_type in queries:
        if len(tool_calls) >= MAX_TOOL_CALLS:
            break
        results = search_policies(query, country=query_country, doc_type=doc_type)
        tool_calls.append({"tool": "search_policies",
                           "input": {"query": query, "country": query_country, "doc_type": doc_type}})
        evidence.extend(relevant(results))

    # Only create ticket after an explicit user request.
    if _is_explicit_ticket_request(message):
        if len(tool_calls) < MAX_TOOL_CALLS:
            ticket_id = create_helpdesk_ticket("Hardware" if "laptop" in qlow else "IT", message)
            tool_calls.append({"tool": "create_helpdesk_ticket",
                               "input": {"category": "Hardware" if "laptop" in qlow else "IT", "summary": message}})

    evidence = latest_policy(evidence)
    source_list = citations(evidence)
    if not evidence:
        answer, flag = FALLBACK, "low"
    else:
        answer = _call_llm(original_question, evidence)
        if not answer:
            # Offline/dev mode: conservative extractive answer, no invented claims.
            answer = "\n\n".join(r["content"] for r in evidence[:3])
        if FALLBACK in answer:
            flag = "low"
            source_list = []
        else:
            flag = "high" if len(evidence) >= 1 else "partial"

    if ticket_id:
        answer += f"\n\nHelpdesk ticket created: {ticket_id}."
    memory.add(session_id, "user", message)
    memory.add(session_id, "assistant", answer)
    return {"answer": answer, "sources": source_list, "confidence_flag": flag,
            "tool_calls": tool_calls, "ticket_id": ticket_id}
