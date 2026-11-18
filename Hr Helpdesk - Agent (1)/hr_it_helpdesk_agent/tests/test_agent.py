import json
from app.agent import handle_chat
from app import agent
from app.memory import memory

def setup_function():
    memory.clear("t1"); memory.clear("t2"); memory.clear("t3"); memory.clear("t4"); memory.clear("t5")

def test_single_document_answer():
    result = handle_chat("How many leave days can I carry forward?", "E102", "t1")
    assert "5 days" in result["answer"]
    assert any("leave-policy-2026.md" in s for s in result["sources"])

def test_two_document_answer_two_searches():
    result = handle_chat("I am taking 3 days of leave for a business trip to Colombo. What approvals do I need?", "E102", "t2")
    searches = [x for x in result["tool_calls"] if x["tool"] == "search_policies"]
    assert len(searches) == 2
    assert len(result["sources"]) == 2

def test_clarification_when_country_missing():
    result = handle_chat("What travel allowance can I claim?", None, "t3")
    assert "India or Sri Lanka" in result["answer"]
    assert result["confidence_flag"] == "partial"

def test_fallback_for_out_of_scope():
    result = handle_chat("What is the capital of Australia?", None, "t4")
    assert result["answer"] == "I couldn't find this in the provided documents."
    assert result["confidence_flag"] == "low"

def test_no_ticket_without_explicit_request(monkeypatch, tmp_path):
    created = []
    monkeypatch.setattr(agent, "create_helpdesk_ticket", lambda category, summary: created.append(summary) or "HD-1001")
    result = handle_chat("What should I do if my laptop screen cracks?", "E102", "t5")
    assert result["ticket_id"] is None
    assert created == []
