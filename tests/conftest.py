import pytest
from app import agent

@pytest.fixture(autouse=True)
def fake_search(monkeypatch):
    def search(query, country=None, doc_type=None):
        if "password" in query.lower() or "login" in query.lower():
            return [{"content": "To reset your password, open the self-service portal and click Forgot Password.",
                     "source": "it-helpdesk-faq.md", "page": None, "section": "Password Reset",
                     "country": None, "doc_type": "it", "effective_date": None, "relevance": 0.95}]
        if "leave" in query.lower() or "carry" in query.lower():
            return [{"content": "Employees can carry forward a maximum of 5 days of annual leave.",
                     "source": "leave-policy-2026.md", "page": None, "section": "Carry Forward",
                     "country": country, "doc_type": "leave", "effective_date": "2026-01-01", "relevance": 0.95}]
        if "travel" in query.lower() or "colombo" in query.lower():
            return [{"content": "Sri Lanka business travel approval details are defined in this policy.",
                     "source": "travel-policy-sri-lanka.md", "page": None, "section": "Approvals",
                     "country": "Sri Lanka", "doc_type": "travel", "effective_date": None, "relevance": 0.95}]
        return []
    monkeypatch.setattr(agent, "search_policies", search)
    monkeypatch.setattr(agent, "_call_llm", lambda q, ctx: "\n".join(x["content"] for x in ctx))
    monkeypatch.setattr(agent, "get_employee_profile", lambda eid: {"employee_id": eid, "country": "India", "manager": "Manager", "grade": "G5"})
