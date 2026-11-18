"""In-memory session history. Replace with Redis/DB for multi-worker production use."""
from collections import defaultdict, deque
from threading import RLock

class SessionMemory:
    def __init__(self, max_messages=20):
        self._data = defaultdict(lambda: deque(maxlen=max_messages))
        self._lock = RLock()

    def get(self, session_id):
        with self._lock:
            return list(self._data[session_id])

    def add(self, session_id, role, content):
        with self._lock:
            self._data[session_id].append({"role": role, "content": content})

    def clear(self, session_id):
        with self._lock:
            self._data.pop(session_id, None)

memory = SessionMemory()
