from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from app.agent import handle_chat

app = FastAPI(title="HR & IT Helpdesk Agent", version="1.0.0")

class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    employee_id: Optional[str] = None
    session_id: str = Field(min_length=1)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        return handle_chat(request.message.strip(), request.employee_id, request.session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        # Log exception in a real deployment; do not expose stack traces to clients.
        raise HTTPException(status_code=503, detail="The helpdesk assistant is temporarily unavailable. Please try again.")
