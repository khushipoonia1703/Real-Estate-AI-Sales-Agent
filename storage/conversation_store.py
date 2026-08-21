"""Session memory: conversation history plus a small lead profile.

State lives in a dict, and is mirrored to a JSON file so a server restart does
not lose the conversations. The file is a convenience, not a database: writes
are whole-file and synchronous, which is fine at demo scale and would be the
first thing to replace with Redis or Postgres.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from config import get_settings

Role = Literal["user", "assistant", "system"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Message:
    role: Role
    content: str
    ts: str = field(default_factory=_now)


@dataclass
class Profile:
    """The small structured picture the agent builds up as it learns things."""

    name: Optional[str] = None
    phone: Optional[str] = None
    language: Optional[str] = None
    configuration_interest: Optional[str] = None
    budget_signal: Optional[str] = None
    purpose: Optional[str] = None
    timeline: Optional[str] = None
    site_visit_status: str = "none"
    site_visit_datetime: Optional[str] = None
    booking_reference: Optional[str] = None
    booking_attempts: int = 0
    do_not_contact: bool = False
    escalated_to_human: bool = False


@dataclass
class Session:
    session_id: str
    messages: List[Message] = field(default_factory=list)
    profile: Profile = field(default_factory=Profile)
    force_booking_failure: bool = False
    created_at: str = field(default_factory=_now)
    ended: bool = False

    def add(self, role: Role, content: str) -> None:
        self.messages.append(Message(role=role, content=content))

    def history(self) -> List[Dict[str, str]]:
        """History in OpenAI chat format, ready to replay to the model."""
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def transcript(self) -> str:
        """Readable transcript for the analytics call."""
        label = {"user": "Customer", "assistant": "Ava", "system": "System"}
        return "\n".join(f"{label[m.role]}: {m.content}" for m in self.messages)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "ended": self.ended,
            "force_booking_failure": self.force_booking_failure,
            "profile": asdict(self.profile),
            "messages": [asdict(m) for m in self.messages],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        """Rebuild a session from stored JSON, ignoring anything unrecognised."""
        profile_fields = Profile.__dataclass_fields__
        stored_profile = data.get("profile") or {}
        profile = Profile(
            **{k: v for k, v in stored_profile.items() if k in profile_fields}
        )
        messages = [
            Message(
                role=m.get("role", "user"),
                content=m.get("content", ""),
                ts=m.get("ts") or _now(),
            )
            for m in data.get("messages") or []
            if m.get("role") in ("user", "assistant", "system")
        ]
        return cls(
            session_id=data["session_id"],
            messages=messages,
            profile=profile,
            force_booking_failure=bool(data.get("force_booking_failure", False)),
            created_at=data.get("created_at") or _now(),
            ended=bool(data.get("ended", False)),
        )


class SessionStore:
    """Thread-safe dict of sessions, mirrored to a JSON file."""

    def __init__(self, path=None) -> None:
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.RLock()
        self._path = path or get_settings().conversations_path

    # -- memory ------------------------------------------------------------ #

    def get_or_create(self, session_id: Optional[str] = None) -> Session:
        sid = (session_id or "").strip() or f"sess_{uuid.uuid4().hex[:12]}"
        with self._lock:
            session = self._sessions.get(sid)
            if session is None:
                session = Session(session_id=sid)
                self._sessions[sid] = session
            return session

    def get(self, session_id: str) -> Optional[Session]:
        with self._lock:
            return self._sessions.get(session_id)

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def clear(self) -> None:
        """Forget everything in memory. Does not touch the file on disk."""
        with self._lock:
            self._sessions.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    # -- persistence ------------------------------------------------------- #

    def load(self) -> None:
        """Read sessions from disk. A missing, empty or corrupt file is not fatal."""
        try:
            raw = self._path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return
        if not raw.strip():
            return
        try:
            stored = json.loads(raw)
        except json.JSONDecodeError:
            # A corrupt file must never stop the server from starting.
            return
        if not isinstance(stored, dict):
            return
        with self._lock:
            for sid, data in stored.items():
                if not isinstance(data, dict):
                    continue
                try:
                    data.setdefault("session_id", sid)
                    self._sessions[sid] = Session.from_dict(data)
                except (KeyError, TypeError, ValueError):
                    continue  # skip one bad record, keep the rest

    def save(self) -> None:
        """Write every session out. Atomic, so a crash mid-write cannot corrupt it."""
        with self._lock:
            payload = {sid: s.to_dict() for sid, s in self._sessions.items()}
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            handle, tmp_name = tempfile.mkstemp(
                dir=str(self._path.parent), prefix=".conversations-", suffix=".tmp"
            )
            with os.fdopen(handle, "w", encoding="utf-8") as tmp:
                json.dump(payload, tmp, ensure_ascii=False, indent=2)
            os.replace(tmp_name, self._path)
        except OSError:
            # Persistence is a convenience; never take a request down for it.
            pass


store = SessionStore()
store.load()
