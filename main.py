"""Northstar Homes AI sales agent - the whole application in one file.

An AI sales consultant ("Ava") for Northstar One, Sector 79, Gurugram. She talks
in English, Hindi or Hinglish, answers only from a closed fact sheet, qualifies
the lead, books a site visit, and - through a second, separate LLM call - emits
structured analytics after the conversation.

The system prompt in ``system_prompt.md`` is the product. This file exists to
make that prompt observable, testable and demonstrable.

Sections, in order:
    1. Settings
    2. Prompts            (system prompt loaded from disk, analytics prompt inline)
    3. Language detection
    4. Session memory     (in-memory dict: history + a small lead profile)
    5. Booking simulation (with a real, deterministic failure path)
    6. LLM access         (Groq via the OpenAI-compatible API, plus an offline mock)
    7. Agent              (prompt assembly, turn orchestration, the booking tool call)
    8. Analytics          (separate call, strict JSON contract, coerced not trusted)
    9. FastAPI routes + the chat UI

Run:   uvicorn main:app --reload
Test:  LLM_MOCK=1 pytest -q
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

logger = logging.getLogger(__name__)

__version__ = "1.0.0"


# =========================================================================== #
# 1. Settings
# =========================================================================== #


def _flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Runtime configuration, read once."""

    groq_api_key: str
    groq_model: str
    groq_base_url: str
    llm_mock: bool
    temperature: float
    max_tokens: int
    booking_force_failure: bool

    @property
    def use_mock(self) -> bool:
        """Mock mode is on when forced, or whenever there is no API key."""
        return self.llm_mock or not self.groq_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings(
        groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
        groq_model=os.getenv("GROQ_MODEL", "").strip() or "llama-3.3-70b-versatile",
        groq_base_url=(
            os.getenv("GROQ_BASE_URL", "").strip() or "https://api.groq.com/openai/v1"
        ),
        llm_mock=_flag("LLM_MOCK"),
        temperature=_float("LLM_TEMPERATURE", 0.4),
        max_tokens=_int("LLM_MAX_TOKENS", 400),
        booking_force_failure=_flag("BOOKING_FORCE_FAILURE"),
    )


# =========================================================================== #
# 2. Prompts
# =========================================================================== #

# The conversation prompt is the deliverable, so it stays in its own file and is
# loaded at runtime - never hard-coded here.
SYSTEM_PROMPT = (BASE_DIR / "system_prompt.md").read_text(encoding="utf-8")

# The analytics prompt is machine plumbing, not product copy, so it lives inline
# next to the schema it has to satisfy.
ANALYTICS_PROMPT = """# ROLE

You are a lead-analysis engine for Northstar Homes. You read one finished
conversation between a sales agent (Ava) and a customer, and you output a single
JSON object describing the lead. Your output goes straight into a CRM, so it
must be machine-readable and it must be true.

You never talk to the customer. You never write prose outside the JSON.

# OUTPUT CONTRACT

Return exactly one JSON object and nothing else. No markdown fences, no
commentary, no trailing text. The object must have exactly these keys:

{
  "lead_summary": "string, 1-2 sentences",
  "language": "english | hindi | hinglish | mixed",
  "budget": "string range or null",
  "configuration_interest": "2BHK | 3BHK | both | undecided | null",
  "purpose": "end_use | investment | unknown",
  "interest_level": "high | medium | low | opted_out",
  "qualification_status": "qualified | partially_qualified | unqualified",
  "objections_raised": ["price", "location", "timing", "trust", "competition", "other"],
  "site_visit_status": "booked | booking_failed | proposed | declined | none",
  "site_visit_datetime": "string or null",
  "follow_up_required": true,
  "follow_up_notes": "string or null",
  "do_not_contact": false,
  "escalated_to_human": false,
  "contact": { "name": "string or null", "phone": "string or null" },
  "next_action": "string, one concrete recommendation for the sales team"
}

# RULES

1. Never invent. If the customer did not say it, it is null, "unknown",
   "undecided", "none", or false. An empty field is correct; a guessed field is
   a bug. Do not infer a budget from the price the agent quoted.
2. Use only the enum values listed. Never invent a new enum value, never change
   the casing.
3. lead_summary is one or two plain sentences in English, whatever language the
   conversation was in.
4. language is the language the customer used. Use "mixed" only if they
   genuinely used more than one across turns and no single one dominates.
5. budget is a short normalised quote of what they said, for example
   "1.2-1.4 crore" or "under 1.5 crore". If they never gave a number, null.
6. purpose is end_use if they will live in it, investment if they will rent or
   resell it, otherwise unknown.
7. interest_level: opted_out if they asked not to be contacted, and that wins
   over everything else; high if they booked or actively agreed to a site visit,
   or pushed for details and next steps; medium if they engaged but did not
   commit; low if they were dismissive, busy, or not interested.
8. qualification_status: qualified if at least three of configuration, budget,
   purpose, timeline and contact number are known; partially_qualified if one or
   two are; unqualified if none are.
9. objections_raised is drawn only from price, location, timing, trust,
   competition, other. Empty list if there were none. No duplicates.
10. site_visit_status: booked only if the system confirmed a booking and the
    agent confirmed it to the customer; booking_failed if a booking was
    attempted, failed, and was not rescued; proposed if a visit was offered or
    discussed but not booked; declined if the customer refused; none if it never
    came up.
11. site_visit_datetime is the slot as the customer said it, for example
    "Saturday 5 pm". Do not convert it to a calendar date and do not invent one.
12. follow_up_required is true if anything is owed to the customer: a callback,
    a confirmed detail, a rescheduled booking, a promised time.
13. do_not_contact is true only on an explicit opt-out request. When it is true,
    follow_up_required must be false and next_action must be about suppressing
    contact, not about selling.
14. escalated_to_human is true if the agent offered or arranged a human
    callback, or the customer asked for a person.
15. contact.phone is copied digit for digit from what the customer gave. Never
    reconstruct one you are unsure about, and never invent one.
16. next_action is one short, concrete instruction for a human. If
    do_not_contact is true it must be
    "Add to do-not-contact list. No further outreach."

Return the JSON object only.
"""


# =========================================================================== #
# 3. Language detection
# =========================================================================== #

# Romanised-Hindi markers. Used only to label the session profile and to drive
# the offline mock; the live agent mirrors language by instruction, not by a
# classifier, because classifiers handle Hinglish badly.
#
# STRONG markers are unambiguous Hindi. WEAK markers are Hindi words that are
# also ordinary English words ("me", "main", "par"), so one alone proves nothing.
_STRONG_MARKERS = {
    "aap", "aapka", "aapke", "aapko", "aapki", "hain", "haan", "nahi", "nhi",
    "kya", "kahan", "kitna", "kitne", "kitni", "kaise", "kaisa", "kaunsa",
    "kyun", "kyu", "mujhe", "mera", "meri", "mere", "hume", "humein", "acha",
    "accha", "theek", "thik", "batao", "bata", "bataiye", "batayiye",
    "chahiye", "chahta", "chahti", "karna", "karo", "karun", "kijiye", "karein",
    "raha", "rahi", "rahe", "liye", "mein", "bhai", "thoda", "zyada", "sirf",
    "matlab", "dekhna", "dekhne", "dekh", "lena", "dena", "jaanna", "janna",
    "bilkul", "shuru", "ghar", "makan", "paisa", "dobara", "phir", "baad",
    "pehle", "abhi", "jaldi", "kabhi", "sochta", "sochti", "rehne", "rehna",
    "hoon", "hun", "hu", "tha", "thi", "jayega", "jayegi", "milega",
    "milegi", "banda", "namaste", "namaskar", "shukriya", "maafi", "baare",
    "jaunga", "jaungi", "aaunga", "aaungi", "galat", "kaam", "waqt",
    "samay", "parso", "haa", "hai", "aur", "koi", "kuch", "wala", "wali",
}
_WEAK_MARKERS = {
    "ke", "ka", "ki", "ko", "se", "me", "bhi", "main", "mai", "hum", "mat",
    "ji", "ye", "wo", "kar", "sahi", "kal", "aaj",
}

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_WORD = re.compile(r"[a-z]+")


def detect_language(text: str) -> str:
    """Best-effort label for a customer message: hindi, hinglish or english.

    This never changes what the agent says (the prompt handles mirroring); it
    only annotates the session profile and steers the deterministic mock.
    """
    if _DEVANAGARI.search(text):
        return "hindi"
    words = _WORD.findall(text.lower())
    if not words:
        return "english"
    strong = sum(1 for w in words if w in _STRONG_MARKERS)
    weak = sum(1 for w in words if w in _WEAK_MARKERS)
    if strong == 0 and weak < 2:
        return "english"
    ratio = (strong + weak) / len(words)
    if ratio >= 0.8 and len(words) >= 3:
        return "hindi"  # romanised Hindi
    return "hinglish"


# =========================================================================== #
# 4. Session memory
# =========================================================================== #

Role = Literal["user", "assistant", "system"]


@dataclass
class Message:
    role: Role
    content: str
    ts: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )


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
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
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
            "profile": asdict(self.profile),
            "messages": [asdict(m) for m in self.messages],
        }


class SessionStore:
    """Thread-safe dict of sessions keyed by session_id.

    Deliberately a dict, not a database. Swapping in Redis or Postgres means
    replacing this one class; nothing else in the app knows where state lives.
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()

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
        with self._lock:
            self._sessions.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)


store = SessionStore()


# =========================================================================== #
# 5. Booking simulation
# =========================================================================== #
#
# Real booking would call a CRM or calendar API. Here it is simulated, but the
# failure path is real and deterministic so the agent's honesty about failures
# can be demonstrated and tested rather than described.
#
# Failure triggers, in priority order:
#   1. force_failure (per-session demo switch, or BOOKING_FORCE_FAILURE=1)
#   2. missing name, phone or slot
#   3. a same-day slot ("today", "tonight") the site team cannot confirm now
#   4. a slot already taken in this process (double booking)

_BOOKED_SLOTS: Dict[str, str] = {}
_BOOKING_LOCK = threading.Lock()

_SAME_DAY = ("today", "tonight", "right now", "abhi", "aaj")

FAILURE_MESSAGES = {
    "forced": "the booking system rejected the request",
    "missing_details": "required booking details were missing",
    "slot_unavailable": "that slot could not be confirmed by the site team",
    "slot_taken": "that slot is already taken",
}


@dataclass(frozen=True)
class BookingResult:
    """Outcome of a booking attempt. ``ok`` is the only thing the agent may trust."""

    ok: bool
    status: str  # "booked" | "booking_failed"
    name: Optional[str] = None
    phone: Optional[str] = None
    when: Optional[str] = None
    confirmation_id: Optional[str] = None
    reason_code: Optional[str] = None
    reason: Optional[str] = None

    def as_system_note(self) -> str:
        """The one message the agent is told about the outcome."""
        if self.ok:
            return (
                "BOOKING RESULT: SUCCESS. The site visit is confirmed for "
                f"{self.when}. Confirmation id {self.confirmation_id}. "
                "Confirm this to the customer in one or two short sentences, "
                "repeat the day and time, mention they will get a text, and close warmly."
            )
        return (
            f"BOOKING RESULT: FAILED. Reason: {self.reason}. "
            "You must NOT tell the customer it was booked. Say briefly that it did "
            "not go through, apologise once, and offer either a different slot or a "
            "callback from a colleague. Keep it short and ask one question."
        )


def _normalise_slot(when: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", when.lower()).strip()


def _digits(phone: str) -> str:
    return re.sub(r"\D", "", phone)


def _confirmation_id(name: str, phone: str, when: str) -> str:
    seed = f"{name}|{_digits(phone)}|{_normalise_slot(when)}".encode("utf-8")
    return "NSH-" + hashlib.sha256(seed).hexdigest()[:8].upper()


def simulate_booking(
    name: str,
    phone: str,
    when: str,
    *,
    force_failure: bool = False,
) -> BookingResult:
    """Attempt to book a site visit. Deterministic: same inputs, same outcome."""
    name = (name or "").strip()
    phone = (phone or "").strip()
    when = (when or "").strip()

    def fail(code: str) -> BookingResult:
        return BookingResult(
            ok=False,
            status="booking_failed",
            name=name or None,
            phone=phone or None,
            when=when or None,
            reason_code=code,
            reason=FAILURE_MESSAGES[code],
        )

    if force_failure or get_settings().booking_force_failure:
        return fail("forced")

    if not name or not when or len(_digits(phone)) < 10:
        return fail("missing_details")

    slot = _normalise_slot(when)
    if any(token in slot for token in _SAME_DAY):
        return fail("slot_unavailable")

    with _BOOKING_LOCK:
        if slot in _BOOKED_SLOTS:
            return fail("slot_taken")
        confirmation_id = _confirmation_id(name, phone, when)
        _BOOKED_SLOTS[slot] = confirmation_id

    return BookingResult(
        ok=True,
        status="booked",
        name=name,
        phone=phone,
        when=when,
        confirmation_id=confirmation_id,
    )


def reset_calendar() -> None:
    """Clear the in-process calendar (used by tests and by /reset)."""
    with _BOOKING_LOCK:
        _BOOKED_SLOTS.clear()


# =========================================================================== #
# 6. LLM access: Groq (OpenAI-compatible) + deterministic offline mock
# =========================================================================== #
#
# The mock exists so the app, the demo and the whole test suite run with no API
# key and no network. It is a small rule-based stand-in for the model, not a
# second agent: the behaviour it imitates is exactly the behaviour
# system_prompt.md specifies, which is what the tests assert on. Run the same
# tests with LIVE_TESTS=1 and a key to check the real prompt against them.

BOOK_TOKEN_RE = re.compile(r"\[\[\s*BOOK\b(?P<body>[^\]]*)\]\]", re.IGNORECASE)

# Providers retire models. If the configured one is gone, fall back rather than
# taking the whole agent down - and say so in the log, never silently.
MODEL_FALLBACKS = ("openai/gpt-oss-120b", "llama-3.1-8b-instant")
_MODEL_GONE = ("model_not_found", "does not exist", "decommissioned", "deprecated")

_active_model: Optional[str] = None
_client = None


class LLMError(RuntimeError):
    """Raised when the live provider cannot be reached or returns nothing."""


def is_mock() -> bool:
    """True when replies come from the deterministic offline mock."""
    return get_settings().use_mock


def describe_backend() -> Dict[str, Any]:
    settings = get_settings()
    return {
        "mode": "mock" if settings.use_mock else "groq",
        "model": None if settings.use_mock else (_active_model or settings.groq_model),
        "base_url": None if settings.use_mock else settings.groq_base_url,
    }


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI  # imported lazily so mock mode needs no SDK

        settings = get_settings()
        _client = OpenAI(api_key=settings.groq_api_key, base_url=settings.groq_base_url)
    return _client


def llm_chat(
    messages: List[Dict[str, str]],
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    json_mode: bool = False,
) -> str:
    """Send a chat completion and return the reply text.

    Falls back to the deterministic mock whenever mock mode is on.
    """
    global _active_model

    settings = get_settings()
    if settings.use_mock:
        return mock_reply(messages, json_mode=json_mode)

    kwargs: Dict[str, Any] = {
        "messages": messages,
        "temperature": settings.temperature if temperature is None else temperature,
        "max_tokens": settings.max_tokens if max_tokens is None else max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    candidates = [_active_model or settings.groq_model]
    candidates += [m for m in MODEL_FALLBACKS if m not in candidates]

    last_error: Optional[Exception] = None
    for model in candidates:
        try:
            response = _get_client().chat.completions.create(model=model, **kwargs)
        except Exception as exc:  # noqa: BLE001 - surfaced as one typed error
            last_error = exc
            if any(marker in str(exc).lower() for marker in _MODEL_GONE):
                logger.warning("Model %s unavailable, trying the next one.", model)
                continue
            raise LLMError(str(exc)) from exc

        if model != _active_model:
            if model != settings.groq_model:
                logger.warning(
                    "Configured model %s is unavailable; using %s instead.",
                    settings.groq_model,
                    model,
                )
            _active_model = model

        content = (response.choices[0].message.content or "").strip()
        if not content:
            raise LLMError("empty completion")
        return content

    raise LLMError(str(last_error))


# --------------------------------------------------------------------------- #
# Phrase lists, shared by the mock, the profile learner and the analytics
# --------------------------------------------------------------------------- #

OPT_OUT_PHRASES = (
    "stop contacting", "stop calling", "stop messaging", "don't contact",
    "do not contact", "dont contact", "remove my number", "remove me",
    "unsubscribe", "never call", "don't call me", "do not call", "dont call me",
    "call mat", "contact mat", "mat karna", "band karo", "band kar",
    "परेशान मत", "मत करना", "बंद कर",
)
HUMAN_REQUEST_PHRASES = (
    "real person", "actual person", "human", "speak to someone",
    "talk to someone", "sales manager", "your manager", "agent se baat",
    "insaan", "kisi se baat", "बात कराओ", "व्यक्ति से",
)
BUSY_PHRASES = (
    "in a meeting", "i'm busy", "im busy", "i am busy", "busy hoon", "busy hu",
    "abhi busy", "driving", "not a good time", "bad time", "can't talk",
    "cant talk", "abhi baat nahi",
)
NOT_INTERESTED_PHRASES = (
    "not interested", "no interest", "interest nahi", "nahi chahiye",
    "not looking", "already bought", "मुझे नहीं चाहिए", "दिलचस्पी नहीं",
)
LATER_PHRASES = (
    "call me later", "call later", "next week", "call me tomorrow", "baad mein",
    "baad me", "later", "kal call", "phir call", "some other time",
)
PRICE_PHRASES = (
    "price", "cost", "rate", "how much", "kitne ka", "keemat", "daam",
    "कीमत", "दाम", "कितने",
)
UNKNOWN_PHRASES = (
    "possession", "handover", "carpet", "built up", "built-up", "square feet",
    "sq ft", "sqft", "floor plan", "layout", "amenit", "clubhouse", "swimming",
    "pool", "gym", "parking", "rera", "approval", "emi", "loan", "bank",
    "payment plan", "maintenance", "booking amount", "availability",
    "units left", "inventory",
    "brochure", "area of", "size of", "how big", "kitna bada", "kab milega",
    "kab tak", "possession kab", "kitna area", "area kitna", "कब मिलेगा", "क्षेत्रफल",
)
# Anything that asks the neighbourhood to be quantified. Section 1.1 of the
# prompt lets Ava describe Sector 79, but never with a number or a place name.
DISTANCE_PHRASES = (
    "how far", "distance", "how long does it take", "how much time",
    "minutes from", "km from", "metro", "highway", "expressway", "airport",
    "cyber city", "commute time", "kitni door", "kitna door", "kitne door",
    "कितनी दूर", "मेट्रो",
)
# Open questions about the area, which Ava may answer in general terms.
LOCATION_PHRASES = (
    "what's around", "whats around", "what is around", "nearby", "near by",
    "around there", "around it", "neighbourhood", "neighborhood", "locality",
    "location", "school", "hospital", "clinic", "mall", "market", "office",
    "connectivity", "aas paas", "aaspaas", "paas mein", "nazdeek",
    "आसपास", "पास में", "स्कूल", "अस्पताल",
)
OBJECTION_PRICE_PHRASES = (
    "too expensive", "expensive", "costly", "too much", "over budget",
    "out of budget", "mehnga", "mehanga", "zyada hai", "bahut zyada",
    "discount", "negotiat", "kam ho", "kam kar", "महंगा", "छूट",
)
OBJECTION_LOCATION_PHRASES = (
    "too far", "far from", "location is bad", "sector 79 is far", "door hai",
    "dur hai", "bahut door", "बहुत दूर",
)
THINKING_PHRASES = (
    "think about it", "let me think", "sochta", "sochti", "soch kar",
    "get back to you", "सोच", "need time",
)
COMPETITION_PHRASES = (
    "comparing", "other project", "another project", "competitor",
    "dusre project", "dusra project", "also looking at",
)
VISIT_PHRASES = (
    "site visit", "visit", "come see", "see the", "dekhne", "dekhna hai",
    "aana chahta", "aa jaunga", "schedule", "book", "appointment", "monday",
    "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "weekend", "milte hain", "aa sakta", "aa sakte",
)
PUSHBACK_PHRASES = (
    "rough idea", "roughly", "approx", "estimate", "ballpark", "just tell me",
    "andaza", "andaaza", "lagbhag", "koi idea", "any idea", "guess",
)
AFFIRM_PHRASES = (
    "yes", "yeah", "yep", "sure", "ok", "okay", "please", "confirm",
    "confirmed", "go ahead", "lock", "haan", "haa", "ji", "bilkul",
    "theek hai", "thik hai", "kar do", "kar dijiye", "हाँ", "ठीक",
)
GREETING_PHRASES = (
    "hi", "hello", "hey", "namaste", "namaskar", "good morning", "good evening",
)

# Words that look like a bare name reply but are not one.
_NON_NAME_WORDS = {
    "yes", "no", "not", "ok", "okay", "sure", "please", "thanks", "thank", "hi",
    "hello", "hey", "haan", "haa", "ji", "nahi", "theek", "thik", "hai", "bhi",
    "visit", "book", "price", "call", "later", "busy", "stop", "just", "the",
    "a", "an", "in", "on", "at", "im", "i", "am", "is", "are", "was", "here",
    "there", "from", "looking", "interested", "sorry", "fine", "good", "free",
    "available", "coming", "going", "meeting", "driving", "sector", "gurugram",
    "morning", "evening", "afternoon", "tomorrow", "weekend", "today",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
}

PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?([6-9]\d{9})(?!\d)")
NAME_RE = re.compile(
    r"(?:my name is|name is|mera naam|naam hai|this is|i am|i'm|im)\s+"
    r"([A-Za-z]{2,20}(?:\s+[A-Za-z]{2,20})?)",
    re.IGNORECASE,
)
_SLOT_RE = re.compile(
    r"((?:this |next |coming )?(?:monday|tuesday|wednesday|thursday|friday|"
    r"saturday|sunday|tomorrow|weekend|today|tonight)"
    r"(?:\s+(?:morning|afternoon|evening|night))?"
    r"(?:\s+at)?(?:\s+\d{1,2}(?::\d{2})?\s*(?:am|pm|baje|o'clock)?)?)",
    re.IGNORECASE,
)
_TIME_RE = re.compile(r"\b(\d{1,2}(?::\d{2})?\s*(?:am|pm|baje))\b", re.IGNORECASE)
_BUDGET_MENTION_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:crore|cr\b|lakh|lac|करोड़|लाख)", re.IGNORECASE
)

# Markers that identify the mock's own booking questions in the history.
_ASK_NAME_MARKERS = ("may i have your name", "aapka naam", "आपका नाम")
_ASK_SLOT_MARKERS = ("saturday work", "saturday theek", "शनिवार ठीक")
_ASK_PHONE_MARKERS = ("send the confirmation", "confirmation kis number", "पुष्टि किस नंबर")
_CONFIRM_MARKERS = ("lock that in", "confirm karun", "पुष्टि करूँ")


def has_phrase(text: str, needles) -> bool:
    return any(n in text for n in needles)


def _tokens(text: str):
    return set(re.findall(r"[\w']+", text.lower()))


def _affirms(text: str) -> bool:
    """Whole-word yes-detection, so 'what' does not count as 'ha'."""
    low = text.lower()
    if any(" " in a and a in low for a in AFFIRM_PHRASES):
        return True
    return bool(_tokens(low) & {a for a in AFFIRM_PHRASES if " " not in a})


def name_from(text: str) -> Optional[str]:
    """Pull a customer name out of a message, or None. Conservative on purpose."""
    match = NAME_RE.search(text)
    candidate = None
    if match:
        candidate = match.group(1).strip()
    elif len(text.split()) <= 3 and re.fullmatch(r"[A-Za-z ]{3,30}", text.strip()):
        candidate = text.strip()
    if not candidate:
        return None
    words = candidate.split()
    if not words or _tokens(candidate) & _NON_NAME_WORDS:
        return None
    return " ".join(w.capitalize() for w in words)


def _lang(text: str) -> str:
    lang = detect_language(text)
    return "hinglish" if lang == "hindi" and not _DEVANAGARI.search(text) else lang


def _pick(variants: Dict[str, str], lang: str) -> str:
    return variants.get(lang) or variants["english"]


def _last_user(messages: List[Dict[str, str]]) -> str:
    for msg in reversed(messages):
        if msg["role"] == "user":
            return msg["content"]
    return ""


def _user_messages(messages: List[Dict[str, str]]) -> List[str]:
    return [m["content"] for m in messages if m["role"] == "user"]


def _assistant_text(messages: List[Dict[str, str]]) -> str:
    return " ".join(m["content"].lower() for m in messages if m["role"] == "assistant")


def _collected(messages: List[Dict[str, str]]) -> Dict[str, Optional[str]]:
    """Pull name, phone and slot out of everything the customer has said."""
    name = phone = slot = None
    for text in _user_messages(messages):
        found_phone = PHONE_RE.search(text.replace(" ", ""))
        if found_phone:
            phone = found_phone.group(1)
        found_name = name_from(text)
        if found_name:
            name = found_name
        found_slot = _SLOT_RE.search(text)
        if found_slot and found_slot.group(1).strip():
            slot = re.sub(r"\s+", " ", found_slot.group(1)).strip(" ,.")
            time_part = _TIME_RE.search(text)
            if time_part and time_part.group(1).lower() not in slot.lower():
                slot = f"{slot} {time_part.group(1)}"
    return {"name": name, "phone": phone, "when": slot}


def _qualification(messages: List[Dict[str, str]]) -> Dict[str, Optional[str]]:
    """What the mock has learned that it could qualify on."""
    low = " ".join(_user_messages(messages)).lower()
    two = bool(re.search(r"\b2\s*bhk|two\s*bhk", low))
    three = bool(re.search(r"\b3\s*bhk|three\s*bhk", low))
    config = "both" if two and three else "2BHK" if two else "3BHK" if three else None
    purpose = None
    if any(w in low for w in ("invest", "rent", "resale", "nivesh", "निवेश")):
        purpose = "investment"
    elif any(
        w in low
        for w in ("live in", "living", "myself", "my family", "rehne", "rehna", "रहने")
    ):
        purpose = "end_use"
    budget_match = _BUDGET_MENTION_RE.search(low)
    return {
        "config": config,
        "purpose": purpose,
        "budget": budget_match.group(0) if budget_match else None,
    }


_ASK_CONFIG = {
    "english": "Are you looking at a two BHK or a three BHK?",
    "hinglish": "Aap 2 BHK dekh rahe hain ya 3 BHK?",
    "hindi": "आप 2 BHK देख रहे हैं या 3 BHK?",
}
_ASK_PURPOSE = {
    "english": "Is this for you to live in, or an investment?",
    "hinglish": "Aap khud rehne ke liye dekh rahe hain ya investment ke liye?",
    "hindi": "यह खुद रहने के लिए है या निवेश के लिए?",
}
_ASK_BUDGET = {
    "english": "What budget range are you working with?",
    "hinglish": "Aapka comfortable budget range kya hai?",
    "hindi": "आपका बजट रेंज क्या है?",
}
_INVITE_VISIT = {
    "english": "Would this Saturday work for a site visit, or is Sunday easier?",
    "hinglish": "Site visit ke liye Saturday theek rahega ya Sunday better hai?",
    "hindi": "साइट विज़िट के लिए शनिवार ठीक रहेगा या रविवार बेहतर है?",
}


def _next_question(messages: List[Dict[str, str]], lang: str) -> str:
    """The one question this turn should end on: the first gap in the profile."""
    known = _qualification(messages)
    if not known["config"]:
        return _pick(_ASK_CONFIG, lang)
    if not known["purpose"]:
        return _pick(_ASK_PURPOSE, lang)
    if not known["budget"]:
        return _pick(_ASK_BUDGET, lang)
    return _pick(_INVITE_VISIT, lang)


def mock_reply(messages: List[Dict[str, str]], *, json_mode: bool = False) -> str:
    """Deterministic stand-in for the model. Same input, same output, always."""
    if json_mode:
        return '{"note": "analytics runs rule-based in mock mode"}'

    last = messages[-1] if messages else {"role": "user", "content": ""}

    # A booking outcome reported by the system outranks everything else.
    if last["role"] == "system" and "BOOKING RESULT" in last["content"]:
        return _booking_outcome_reply(last["content"], messages)

    user_text = _last_user(messages)
    text = user_text.lower().strip()
    lang = _lang(user_text)
    assistant_text = _assistant_text(messages)

    # 1. Opt-out beats every other goal.
    if has_phrase(text, OPT_OUT_PHRASES):
        return _pick(
            {
                "english": "Understood, I have noted that and you will not be contacted again. Sorry for the trouble, and have a good day.",
                "hinglish": "Bilkul, main note kar deti hoon, aage se aapko contact nahi kiya jayega. Pareshani ke liye maafi chahti hoon.",
                "hindi": "ज़रूर, मैंने नोट कर लिया है, आपको दोबारा संपर्क नहीं किया जाएगा। परेशानी के लिए क्षमा चाहती हूँ।",
            },
            lang,
        )
    if any(has_phrase(m.lower(), OPT_OUT_PHRASES) for m in _user_messages(messages)):
        return _pick(
            {
                "english": "Thank you, take care.",
                "hinglish": "Shukriya, apna dhyaan rakhiye.",
                "hindi": "धन्यवाद, अपना ध्यान रखिए।",
            },
            lang,
        )

    # 2. Human escalation.
    if has_phrase(text, HUMAN_REQUEST_PHRASES):
        return _pick(
            {
                "english": "Absolutely, I will have one of our sales managers call you. What time suits you best?",
                "hinglish": "Bilkul, main sales manager se call karwa deti hoon. Aapko kis time call karein?",
                "hindi": "बिलकुल, मैं हमारे सेल्स मैनेजर से कॉल करवा देती हूँ। आपको किस समय कॉल करें?",
            },
            lang,
        )

    # 3. The neighbourhood. Describable in general terms, never quantified.
    if has_phrase(text, DISTANCE_PHRASES):
        return _pick(
            {
                "english": "I do not want to give you a wrong distance on that, and traffic changes it anyway. Our team can give you the exact answer. Would a site visit work, so you can judge the drive yourself?",
                "hinglish": "Us par main aapko galat distance nahi batana chahti, aur traffic se waise bhi badal jata hai. Team aapko exact bata degi. Aap site visit pe aa jaiye, khud andaza ho jayega, theek hai?",
                "hindi": "उस पर मैं आपको गलत दूरी नहीं बताना चाहती, और ट्रैफ़िक से वैसे भी बदल जाती है। टीम आपको सही जानकारी दे देगी। आप साइट विज़िट पर आ जाइए, खुद अंदाज़ा हो जाएगा, ठीक है?",
            },
            lang,
        )
    if has_phrase(text, LOCATION_PHRASES) and not has_phrase(
        text, OBJECTION_LOCATION_PHRASES
    ):
        return _pick(
            {
                "english": "Sector 79 is one of Gurugram's newer residential sectors, so there are schools, hospitals, markets and malls around it. I would rather not name one and get it wrong, so shall I have the team send you the exact list?",
                "hinglish": "Sector 79 Gurugram ke naye residential sectors mein se hai, to aas paas schools, hospitals, markets aur malls sab hain. Main kisi ka naam galat nahi batana chahti, to team se exact list bhijwa dun?",
                "hindi": "सेक्टर 79 गुड़गांव के नए रिहायशी सेक्टरों में से है, तो आसपास स्कूल, अस्पताल, मार्केट और मॉल सब हैं। मैं किसी का नाम गलत नहीं बताना चाहती, तो टीम से सही सूची भिजवा दूँ?",
            },
            lang,
        )

    # 4. Unknown questions, and pushing for a guess about one.
    asked_unknown_before = (
        "wrong figure" in assistant_text or "galat number" in assistant_text
    )
    if has_phrase(text, UNKNOWN_PHRASES) and not has_phrase(text, OBJECTION_PRICE_PHRASES):
        return _pick(
            {
                "english": "I do not want to give you a wrong figure on that, so let me get it confirmed by our team. Would you like them to call you with the exact details?",
                "hinglish": "Us baare mein main aapko galat number nahi batana chahti, isliye team se confirm karwa deti hoon. Wo aapko exact detail ke saath call kar lein?",
                "hindi": "उस बारे में मैं आपको गलत आँकड़ा नहीं बताना चाहती, इसलिए टीम से पुष्टि करवा देती हूँ। क्या वे आपको सही जानकारी के साथ कॉल कर लें?",
            },
            lang,
        )
    if asked_unknown_before and has_phrase(text, PUSHBACK_PHRASES):
        return _pick(
            {
                "english": "I understand, but a rough number from me could be badly off and I do not want to mislead you. I will get you the exact figure from the team. Can I take your number for that?",
                "hinglish": "Samajh sakti hoon, lekin mera andaza galat ho sakta hai aur main aapko mislead nahi karna chahti. Main team se exact figure nikalwa deti hoon. Aapka number le lun?",
                "hindi": "समझती हूँ, पर मेरा अंदाज़ा गलत हो सकता है और मैं आपको गुमराह नहीं करना चाहती। मैं टीम से सही आँकड़ा निकलवा देती हूँ। आपका नंबर ले लूँ?",
            },
            lang,
        )

    # 5. Objections.
    if has_phrase(text, OBJECTION_PRICE_PHRASES):
        return _pick(
            {
                "english": "I hear you, it is a serious amount. That is the starting price in Sector 79, Gurugram, and I would not want to promise anything I cannot. Would seeing it this weekend help you judge it?",
                "hinglish": "Samajh sakti hoon, amount bada hai. Ye Sector 79, Gurugram mein starting price hai, aur main koi aisa vaada nahi karna chahti jo pura na ho. Weekend pe dekh lenge to andaza aa jayega, chalein?",
                "hindi": "समझ सकती हूँ, रकम बड़ी है। यह सेक्टर 79, गुड़गांव में शुरुआती कीमत है, और मैं ऐसा कोई वादा नहीं करना चाहती जो पूरा न हो। इस weekend देख लें तो अंदाज़ा आ जाएगा, चलें?",
            },
            lang,
        )
    if has_phrase(text, OBJECTION_LOCATION_PHRASES):
        return _pick(
            {
                "english": "That is fair, it depends a lot on where you are based. Where would you be commuting from?",
                "hinglish": "Sahi baat hai, ye is pe depend karta hai ki aap kahan rehte hain. Aapko kahan se aana jaana hoga?",
                "hindi": "सही बात है, यह इस पर निर्भर करता है कि आप कहाँ रहते हैं। आपको कहाँ से आना-जाना होगा?",
            },
            lang,
        )
    if has_phrase(text, COMPETITION_PHRASES):
        return _pick(
            {
                "english": "Makes sense, you should compare. I would just say see Northstar One in person before you decide. Can I fit a visit around your schedule?",
                "hinglish": "Bilkul sahi, compare karna chahiye. Bas ek baar Northstar One khud dekh lijiye decide karne se pehle. Aapke schedule ke hisaab se visit rakh dun?",
                "hindi": "सही है, तुलना करनी चाहिए। बस एक बार नॉर्थस्टार वन खुद देख लीजिए। आपके समय के अनुसार विज़िट रख दूँ?",
            },
            lang,
        )
    if has_phrase(text, THINKING_PHRASES):
        return _pick(
            {
                "english": "Of course, it is a big decision. What is the one thing that would help you decide?",
                "hinglish": "Bilkul, bada decision hai. Ek cheez batayiye jo decide karne mein sabse zyada help karegi?",
                "hindi": "बिलकुल, बड़ा फैसला है। एक चीज़ बताइए जो निर्णय लेने में सबसे ज़्यादा मदद करेगी?",
            },
            lang,
        )

    # 6. Not interested, busy, later.
    if has_phrase(text, NOT_INTERESTED_PHRASES):
        return _pick(
            {
                "english": "Understood, thanks for telling me. If you ever want to look at Sector 79 again, we are here. Have a good day.",
                "hinglish": "Samajh gayi, batane ke liye shukriya. Kabhi bhi Sector 79 dekhna ho to hum yahin hain. Aapka din achha rahe.",
                "hindi": "समझ गई, बताने के लिए धन्यवाद। कभी सेक्टर 79 देखना हो तो हम यहीं हैं। आपका दिन अच्छा रहे।",
            },
            lang,
        )
    if has_phrase(text, BUSY_PHRASES):
        return _pick(
            {
                "english": "No problem at all, I will keep this for later. What time works better for you?",
                "hinglish": "Koi baat nahi, main baad mein baat kar leti hoon. Aapko kis time convenient rahega?",
                "hindi": "कोई बात नहीं, मैं बाद में बात कर लेती हूँ। आपको किस समय सुविधा रहेगी?",
            },
            lang,
        )
    if has_phrase(text, LATER_PHRASES):
        return _pick(
            {
                "english": "Sure, I will note that down. Which day and time should I call you on?",
                "hinglish": "Theek hai, main note kar leti hoon. Kis din aur kis time call karun?",
                "hindi": "ठीक है, मैं नोट कर लेती हूँ। किस दिन और किस समय कॉल करूँ?",
            },
            lang,
        )

    # 7. Booking flow: active once a visit is on the table, and it stays active
    #    while the details are being collected.
    booking_started = any(
        has_phrase(m.lower(), VISIT_PHRASES) for m in _user_messages(messages)
    ) or has_phrase(
        assistant_text, _ASK_NAME_MARKERS + _ASK_SLOT_MARKERS + _ASK_PHONE_MARKERS
    )
    already_booked = any(
        "BOOKING RESULT" in m["content"] for m in messages if m["role"] == "system"
    )
    if booking_started and not already_booked:
        return _booking_turn(messages, lang, assistant_text)

    # 8. Price: the one thing we can always answer.
    if has_phrase(text, PRICE_PHRASES) or re.search(r"\b[23]\s*bhk|two bhk|three bhk", text):
        config = _qualification(messages)["config"]
        if config == "3BHK":
            fact = {
                "english": "Three BHKs at Northstar One start at one point seven five crore.",
                "hinglish": "Northstar One mein 3 BHK one point seven five crore se shuru hote hain.",
                "hindi": "नॉर्थस्टार वन में 3 BHK एक पॉइंट सात पाँच करोड़ से शुरू होते हैं।",
            }
        elif config == "2BHK":
            fact = {
                "english": "Two BHKs start at one point three five crore.",
                "hinglish": "2 BHK one point three five crore se shuru hote hain.",
                "hindi": "2 BHK एक पॉइंट तीन पाँच करोड़ से शुरू होते हैं।",
            }
        else:
            fact = {
                "english": "Two BHKs start at one point three five crore, and three BHKs at one point seven five crore.",
                "hinglish": "2 BHK one point three five crore se aur 3 BHK one point seven five crore se shuru hote hain.",
                "hindi": "2 BHK एक पॉइंट तीन पाँच करोड़ से और 3 BHK एक पॉइंट सात पाँच करोड़ से शुरू होते हैं।",
            }
        return f"{_pick(fact, lang)} {_next_question(messages, lang)}"

    # 9. Greeting or first turn.
    if has_phrase(text, GREETING_PHRASES) or len(messages) <= 2:
        opener = {
            "english": "Hi, this is Ava from Northstar Homes, about Northstar One in Sector 79, Gurugram.",
            "hinglish": "Hi, main Ava, Northstar Homes se. Northstar One Sector 79, Gurugram mein hai.",
            "hindi": "नमस्ते, मैं आवा, नॉर्थस्टार होम्स से। नॉर्थस्टार वन सेक्टर 79, गुड़गांव में है।",
        }
        return f"{_pick(opener, lang)} {_next_question(messages, lang)}"

    # 10. Otherwise: acknowledge, then keep qualifying.
    ack = {
        "english": "Got it, thank you.",
        "hinglish": "Theek hai, shukriya.",
        "hindi": "ठीक है, धन्यवाद।",
    }
    return f"{_pick(ack, lang)} {_next_question(messages, lang)}"


def _booking_turn(messages: List[Dict[str, str]], lang: str, assistant_text: str) -> str:
    """Collect name, then slot, then phone, then read back, then book."""
    collected = _collected(messages)
    asked_to_confirm = has_phrase(assistant_text, _CONFIRM_MARKERS)

    if asked_to_confirm and _affirms(_last_user(messages)) and all(collected.values()):
        spoken = _pick(
            {
                "english": "Booking that for you now.",
                "hinglish": "Main abhi book kar deti hoon.",
                "hindi": "मैं अभी बुक कर देती हूँ।",
            },
            lang,
        )
        return (
            f'{spoken}\n[[BOOK name="{collected["name"]}"; '
            f'phone="{collected["phone"]}"; when="{collected["when"]}"]]'
        )

    if not collected["name"]:
        return _pick(
            {
                "english": "Happy to set that up. May I have your name, please?",
                "hinglish": "Bilkul set kar deti hoon. Aapka naam bata dijiye?",
                "hindi": "ज़रूर, मैं व्यवस्था कर देती हूँ। आपका नाम बता दीजिए?",
            },
            lang,
        )
    if not collected["when"]:
        return _pick(
            {
                "english": "Great. Would this Saturday work, or is Sunday easier?",
                "hinglish": "Badhiya. Saturday theek rahega ya Sunday better hai?",
                "hindi": "बढ़िया। शनिवार ठीक रहेगा या रविवार बेहतर है?",
            },
            lang,
        )
    if not collected["phone"]:
        return _pick(
            {
                "english": "Noted. What number should I send the confirmation to?",
                "hinglish": "Note kar liya. Confirmation kis number pe bhejun?",
                "hindi": "नोट कर लिया। पुष्टि किस नंबर पर भेजूँ?",
            },
            lang,
        )
    return _pick(
        {
            "english": (
                f"So that is {collected['name']}, {collected['when']}, and I will text "
                f"the confirmation to {collected['phone']}. Shall I lock that in?"
            ),
            "hinglish": (
                f"To {collected['name']}, {collected['when']}, aur confirmation "
                f"{collected['phone']} pe bhej dungi. Main confirm karun?"
            ),
            "hindi": (
                f"तो {collected['name']}, {collected['when']}, और पुष्टि "
                f"{collected['phone']} पर भेज दूँगी। मैं पुष्टि करूँ?"
            ),
        },
        lang,
    )


def _booking_outcome_reply(note: str, messages: List[Dict[str, str]]) -> str:
    lang = _lang(_last_user(messages))
    when = _collected(messages)["when"] or "that slot"
    if "SUCCESS" in note:
        return _pick(
            {
                "english": f"You are confirmed for {when}, and you will get a text with the details. See you there.",
                "hinglish": f"Aapki visit {when} ke liye confirm ho gayi hai, details text pe aa jayengi. Milte hain.",
                "hindi": f"आपकी विज़िट {when} के लिए पक्की हो गई है, विवरण मैसेज पर आ जाएगा। मिलते हैं।",
            },
            lang,
        )
    return _pick(
        {
            "english": "I am sorry, that slot did not go through at our end. I can try another time, or have a colleague call you and lock it in. Which would you prefer?",
            "hinglish": "Maafi chahti hoon, wo slot confirm nahi ho payi. Main dusra time try kar sakti hoon, ya team se call karwa deti hoon. Aap kya prefer karenge?",
            "hindi": "क्षमा चाहती हूँ, वह स्लॉट कन्फर्म नहीं हो पाया। मैं दूसरा समय देख सकती हूँ, या टीम से कॉल करवा देती हूँ। आप क्या पसंद करेंगे?",
        },
        lang,
    )


# =========================================================================== #
# 7. The agent: prompt assembly and turn orchestration
# =========================================================================== #
#
# One turn is:
#     user message -> [system prompt + session context + history] -> model reply
# and, if the model emitted the booking control line, a second pass:
#     booking result -> model -> the reply the customer actually sees.
#
# The control line [[BOOK name="..."; phone="..."; when="..."]] is the agent's
# only tool call. It is stripped here and never reaches the customer.

_KV_RE = re.compile(r'(\w+)\s*=\s*"([^"]*)"')
_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF←-⇿⬀-⯿]",
    flags=re.UNICODE,
)

# Used only when the provider itself fails. Honest, and escalates.
FALLBACK_REPLY = (
    "Sorry, I am having trouble on my end right now. Can I have a colleague "
    "call you back instead?"
)

_BUDGET_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(crore|cr|karod|करोड़|lakh|lac|lakhs|लाख)", re.IGNORECASE
)
_INVEST_WORDS = ("invest", "rental", "rent out", "resale", "nivesh", "निवेश")
_END_USE_WORDS = (
    "live in", "living", "for myself", "my family", "end use", "shift",
    "rehne", "rehna", "khud rehne", "रहने",
)
_TIMELINE_WORDS = (
    "immediately", "right away", "this month", "next month", "in a month",
    "few months", "this year", "next year", "just exploring", "just looking",
    "abhi", "jaldi", "agle mahine", "do teen mahine",
)


@dataclass
class TurnResult:
    """What one turn produced: the reply, plus anything the system did."""

    reply: str
    session_id: str
    booking: Optional[BookingResult] = None
    profile: Optional[Dict[str, object]] = None


def build_system_prompt(session: Session, channel: str = "chat") -> str:
    """System prompt plus a short, factual note on what we already know."""
    profile = session.profile
    known: List[str] = []
    if profile.name:
        known.append(f"customer name: {profile.name}")
    if profile.language:
        known.append(f"language they have been using: {profile.language}")
    if profile.configuration_interest:
        known.append(f"configuration of interest: {profile.configuration_interest}")
    if profile.budget_signal:
        known.append(f"budget signal: {profile.budget_signal}")
    if profile.purpose:
        known.append(f"purpose: {profile.purpose}")
    if profile.timeline:
        known.append(f"timeline: {profile.timeline}")
    if profile.phone:
        known.append(f"phone on file: {profile.phone}")
    if profile.site_visit_status != "none":
        known.append(f"site visit status: {profile.site_visit_status}")
    if profile.site_visit_datetime:
        known.append(f"slot discussed: {profile.site_visit_datetime}")
    if profile.do_not_contact:
        known.append("THIS CUSTOMER HAS OPTED OUT: do not sell, do not ask for a visit")
    if profile.booking_attempts >= 2 and profile.site_visit_status == "booking_failed":
        known.append("booking has failed twice: offer a human callback now")

    context = "\n".join(f"- {item}" for item in known) or "- nothing yet"
    return (
        f"{SYSTEM_PROMPT}\n\n---\n\n"
        "# SESSION CONTEXT (internal notes, never read this aloud)\n\n"
        f"Channel: {channel}\n"
        "What you already know about this customer:\n"
        f"{context}\n\n"
        "Do not ask again for anything listed above. Use it naturally."
    )


def _messages_for(session: Session, channel: str = "chat") -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": build_system_prompt(session, channel)}
    ] + session.history()


def clean_reply(text: str) -> str:
    """Strip anything that is not plain speech: markdown, emojis, control lines."""
    text = BOOK_TOKEN_RE.sub("", text)
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = _EMOJI_RE.sub("", text)
    lines = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•·>]+|#{1,6}|\d+[.)])\s+", "", line)
        line = line.replace("**", "").replace("__", "").replace("`", "")
        line = re.sub(r"(?<!\w)\*(?!\w)", "", line)
        lines.append(line.strip())
    text = " ".join(line for line in lines if line)
    return re.sub(r"\s{2,}", " ", text).strip()


def extract_booking_request(text: str) -> Optional[Dict[str, str]]:
    """Parse the ``[[BOOK ...]]`` control line, if the model emitted one."""
    match = BOOK_TOKEN_RE.search(text)
    if not match:
        return None
    fields = {k.lower(): v.strip() for k, v in _KV_RE.findall(match.group("body"))}
    return {
        "name": fields.get("name", ""),
        "phone": fields.get("phone", ""),
        "when": fields.get("when", ""),
    }


def update_profile_from_user(session: Session, message: str) -> None:
    """Learn the cheap, unambiguous things from what the customer just said."""
    profile = session.profile
    low = message.lower()
    profile.language = detect_language(message)

    if has_phrase(low, OPT_OUT_PHRASES):
        profile.do_not_contact = True
    if has_phrase(low, HUMAN_REQUEST_PHRASES):
        profile.escalated_to_human = True

    has_two = bool(re.search(r"\b2\s*bhk\b|two\s*bhk", low))
    has_three = bool(re.search(r"\b3\s*bhk\b|three\s*bhk", low))
    if has_two and has_three:
        profile.configuration_interest = "both"
    elif has_two:
        profile.configuration_interest = "2BHK"
    elif has_three:
        profile.configuration_interest = "3BHK"

    budget = _BUDGET_RE.search(message)
    if budget:
        unit = budget.group(2).lower()
        unit = "crore" if unit in {"crore", "cr", "karod", "करोड़"} else "lakh"
        # Only treat it as their budget if they framed it as theirs, not as our price.
        if any(
            w in low
            for w in ("budget", "afford", "range", "under", "upto", "up to", "around", "tak", "tight")
        ):
            profile.budget_signal = f"{budget.group(1)} {unit}"

    if any(w in low for w in _INVEST_WORDS):
        profile.purpose = "investment"
    elif any(w in low for w in _END_USE_WORDS):
        profile.purpose = "end_use"

    for word in _TIMELINE_WORDS:
        if word in low:
            profile.timeline = word
            break

    phone = PHONE_RE.search(message.replace(" ", ""))
    if phone:
        profile.phone = phone.group(1)
    name = name_from(message)
    if name:
        profile.name = name


def _apply_booking_to_profile(session: Session, result: BookingResult) -> None:
    profile = session.profile
    profile.booking_attempts += 1
    profile.site_visit_status = result.status
    if result.name:
        profile.name = result.name
    if result.phone:
        profile.phone = result.phone
    if result.when:
        profile.site_visit_datetime = result.when
    profile.booking_reference = result.confirmation_id
    if not result.ok and profile.booking_attempts >= 2:
        profile.escalated_to_human = True


def handle_turn(session: Session, message: str, channel: str = "chat") -> TurnResult:
    """Run one customer turn end to end and return the reply the customer sees."""
    session.add("user", message)
    update_profile_from_user(session, message)

    try:
        raw = llm_chat(_messages_for(session, channel))
    except LLMError:
        session.add("assistant", FALLBACK_REPLY)
        session.profile.escalated_to_human = True
        return TurnResult(reply=FALLBACK_REPLY, session_id=session.session_id)

    booking_request = extract_booking_request(raw)
    if booking_request is None:
        reply = clean_reply(raw)
        session.add("assistant", reply)
        return TurnResult(
            reply=reply,
            session_id=session.session_id,
            profile=vars(session.profile).copy(),
        )

    # The model asked to book. Record what it said, run the booking, then let it
    # speak again with the real outcome. It never announces the result itself.
    pre_booking = clean_reply(raw)
    if pre_booking:
        session.add("assistant", pre_booking)

    result = simulate_booking(
        booking_request["name"],
        booking_request["phone"],
        booking_request["when"],
        force_failure=session.force_booking_failure,
    )
    _apply_booking_to_profile(session, result)
    session.add("system", result.as_system_note())

    try:
        reply = clean_reply(llm_chat(_messages_for(session, channel)))
    except LLMError:
        reply = FALLBACK_REPLY

    if not reply:
        reply = FALLBACK_REPLY
    session.add("assistant", reply)
    return TurnResult(
        reply=reply,
        session_id=session.session_id,
        booking=result,
        profile=vars(session.profile).copy(),
    )


# =========================================================================== #
# 8. Post-conversation analytics
# =========================================================================== #
#
# A separate LLM call with its own prompt and its own output contract, so a
# spoken reply can never be contaminated with machine formatting.
#
# Two things keep the output trustworthy:
#   1. the schema is coerced, not trusted - unknown values fall back to
#      null/false rather than to a guess;
#   2. facts the system actually knows (booking outcome, opt-out, phone captured
#      during booking) override whatever the model inferred.

LANGUAGES = ("english", "hindi", "hinglish", "mixed")
CONFIGURATIONS = ("2BHK", "3BHK", "both", "undecided")
PURPOSES = ("end_use", "investment", "unknown")
INTEREST_LEVELS = ("high", "medium", "low", "opted_out")
QUALIFICATION = ("qualified", "partially_qualified", "unqualified")
OBJECTIONS = ("price", "location", "timing", "trust", "competition", "other")
VISIT_STATUSES = ("booked", "booking_failed", "proposed", "declined", "none")

SCHEMA_KEYS = (
    "lead_summary", "language", "budget", "configuration_interest", "purpose",
    "interest_level", "qualification_status", "objections_raised",
    "site_visit_status", "site_visit_datetime", "follow_up_required",
    "follow_up_notes", "do_not_contact", "escalated_to_human", "contact",
    "next_action",
)

DNC_NEXT_ACTION = "Add to do-not-contact list. No further outreach."

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _empty_analytics() -> Dict[str, Any]:
    return {
        "lead_summary": "No conversation recorded.",
        "language": "english",
        "budget": None,
        "configuration_interest": None,
        "purpose": "unknown",
        "interest_level": "low",
        "qualification_status": "unqualified",
        "objections_raised": [],
        "site_visit_status": "none",
        "site_visit_datetime": None,
        "follow_up_required": False,
        "follow_up_notes": None,
        "do_not_contact": False,
        "escalated_to_human": False,
        "contact": {"name": None, "phone": None},
        "next_action": "No action needed.",
    }


def _enum(value: Any, allowed: tuple, default: Any) -> Any:
    return value if isinstance(value, str) and value in allowed else default


def _text(value: Any) -> Optional[str]:
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned and cleaned.lower() not in {"null", "none", "n/a", "unknown", ""}:
            return cleaned
    return None


def _bool(value: Any, default: bool = False) -> bool:
    return value if isinstance(value, bool) else default


def coerce(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Force any model output into the schema. Unknown never becomes invented."""
    out = _empty_analytics()
    if not isinstance(raw, dict):
        return out

    out["lead_summary"] = _text(raw.get("lead_summary")) or out["lead_summary"]
    out["language"] = _enum(raw.get("language"), LANGUAGES, "english")
    out["budget"] = _text(raw.get("budget"))
    out["configuration_interest"] = _enum(
        raw.get("configuration_interest"), CONFIGURATIONS, None
    )
    out["purpose"] = _enum(raw.get("purpose"), PURPOSES, "unknown")
    out["interest_level"] = _enum(raw.get("interest_level"), INTEREST_LEVELS, "low")
    out["qualification_status"] = _enum(
        raw.get("qualification_status"), QUALIFICATION, "unqualified"
    )

    objections = raw.get("objections_raised")
    if isinstance(objections, list):
        seen: List[str] = []
        for item in objections:
            if isinstance(item, str) and item in OBJECTIONS and item not in seen:
                seen.append(item)
            elif isinstance(item, str) and item and "other" not in seen:
                seen.append("other")
        out["objections_raised"] = seen

    out["site_visit_status"] = _enum(raw.get("site_visit_status"), VISIT_STATUSES, "none")
    out["site_visit_datetime"] = _text(raw.get("site_visit_datetime"))
    out["follow_up_required"] = _bool(raw.get("follow_up_required"))
    out["follow_up_notes"] = _text(raw.get("follow_up_notes"))
    out["do_not_contact"] = _bool(raw.get("do_not_contact"))
    out["escalated_to_human"] = _bool(raw.get("escalated_to_human"))

    contact = raw.get("contact")
    if isinstance(contact, dict):
        phone = _text(contact.get("phone"))
        out["contact"] = {
            "name": _text(contact.get("name")),
            "phone": re.sub(r"[^\d+]", "", phone) if phone else None,
        }

    out["next_action"] = _text(raw.get("next_action")) or out["next_action"]
    return out


def _apply_ground_truth(data: Dict[str, Any], session: Session) -> Dict[str, Any]:
    """System facts beat model inference. The booking system cannot be argued with."""
    profile = session.profile

    if profile.site_visit_status != "none":
        data["site_visit_status"] = profile.site_visit_status
        if profile.site_visit_datetime:
            data["site_visit_datetime"] = profile.site_visit_datetime
    if profile.name and not data["contact"]["name"]:
        data["contact"]["name"] = profile.name
    if profile.phone and not data["contact"]["phone"]:
        data["contact"]["phone"] = profile.phone
    if profile.configuration_interest and not data["configuration_interest"]:
        data["configuration_interest"] = profile.configuration_interest
    if profile.escalated_to_human:
        data["escalated_to_human"] = True
    if profile.do_not_contact:
        data["do_not_contact"] = True

    if data["site_visit_status"] == "booking_failed":
        data["follow_up_required"] = True

    if data["do_not_contact"]:
        data["interest_level"] = "opted_out"
        data["follow_up_required"] = False
        data["follow_up_notes"] = None
        data["next_action"] = DNC_NEXT_ACTION
    return data


def _rule_based_analytics(session: Session) -> Dict[str, Any]:
    """Deterministic extraction used when there is no live model."""
    data = _empty_analytics()
    profile = session.profile
    user_messages = [m.content for m in session.messages if m.role == "user"]
    low = " ".join(user_messages).lower()

    languages = {detect_language(m) for m in user_messages} or {"english"}
    data["language"] = languages.pop() if len(languages) == 1 else "mixed"

    data["budget"] = profile.budget_signal
    data["configuration_interest"] = profile.configuration_interest or (
        "undecided" if user_messages else None
    )
    data["purpose"] = profile.purpose or "unknown"

    objections: List[str] = []
    if has_phrase(low, OBJECTION_PRICE_PHRASES):
        objections.append("price")
    if has_phrase(low, OBJECTION_LOCATION_PHRASES):
        objections.append("location")
    if has_phrase(low, BUSY_PHRASES) or has_phrase(low, LATER_PHRASES):
        objections.append("timing")
    if has_phrase(low, COMPETITION_PHRASES):
        objections.append("competition")
    data["objections_raised"] = objections

    if profile.site_visit_status == "booked":
        data["interest_level"] = "high"
    elif has_phrase(low, NOT_INTERESTED_PHRASES):
        data["interest_level"] = "low"
    elif len(user_messages) >= 3:
        data["interest_level"] = "medium"

    known = sum(
        1
        for value in (
            profile.configuration_interest,
            profile.budget_signal,
            profile.purpose,
            profile.timeline,
            profile.phone,
        )
        if value
    )
    data["qualification_status"] = (
        "qualified" if known >= 3 else "partially_qualified" if known else "unqualified"
    )

    data["site_visit_status"] = profile.site_visit_status
    data["site_visit_datetime"] = profile.site_visit_datetime
    data["contact"] = {"name": profile.name, "phone": profile.phone}
    data["escalated_to_human"] = profile.escalated_to_human
    data["do_not_contact"] = profile.do_not_contact

    if profile.site_visit_status == "booked":
        data["follow_up_required"] = False
        data["next_action"] = (
            f"Site visit confirmed for {profile.site_visit_datetime}. Send reminder."
        )
    elif profile.site_visit_status == "booking_failed":
        data["follow_up_required"] = True
        data["follow_up_notes"] = "Booking failed; slot needs to be rearranged."
        data["next_action"] = "Call back to rebook the site visit manually."
    elif profile.escalated_to_human:
        data["follow_up_required"] = True
        data["follow_up_notes"] = "Customer asked for a human."
        data["next_action"] = "Sales manager to call the customer back."
    elif has_phrase(low, LATER_PHRASES) or has_phrase(low, BUSY_PHRASES):
        data["follow_up_required"] = True
        data["follow_up_notes"] = "Customer asked to be contacted at a better time."
        data["next_action"] = "Call back at the time the customer suggested."
    elif user_messages and not profile.do_not_contact:
        data["follow_up_required"] = True
        data["next_action"] = "Follow up to arrange a site visit."

    config = data["configuration_interest"] or "no stated configuration"
    data["lead_summary"] = (
        f"Customer discussed {config} at Northstar One over {len(user_messages)} messages; "
        f"site visit status is {data['site_visit_status']}."
        if user_messages
        else data["lead_summary"]
    )
    return data


def analyse(session: Session) -> Dict[str, Any]:
    """Extract structured lead data for one session. Always schema-valid."""
    if not any(m.role == "user" for m in session.messages):
        return _empty_analytics()

    if is_mock():
        return _apply_ground_truth(coerce(_rule_based_analytics(session)), session)

    messages = [
        {"role": "system", "content": ANALYTICS_PROMPT},
        {
            "role": "user",
            "content": (
                "Analyse this conversation and return the JSON object.\n\n"
                "--- TRANSCRIPT START ---\n"
                f"{session.transcript()}\n"
                "--- TRANSCRIPT END ---"
            ),
        },
    ]

    # Two attempts: strict JSON mode first, then plain text with the JSON block
    # pulled out (some models spend their budget reasoning before the object).
    for json_mode in (True, False):
        try:
            raw_text = llm_chat(
                messages, temperature=0.0, max_tokens=1500, json_mode=json_mode
            )
            match = _JSON_BLOCK_RE.search(raw_text)
            parsed = json.loads(match.group(0) if match else raw_text)
        except (LLMError, json.JSONDecodeError, AttributeError, TypeError):
            continue
        return _apply_ground_truth(coerce(parsed), session)

    # Never fail the endpoint and never invent: fall back to what we know.
    return _apply_ground_truth(coerce(_rule_based_analytics(session)), session)


# =========================================================================== #
# 9. FastAPI app
# =========================================================================== #

app = FastAPI(
    title="Northstar Homes AI Sales Agent",
    description="An AI sales consultant for Northstar One, Sector 79, Gurugram.",
    version=__version__,
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    # Demo switch: makes the next booking attempt fail, so the failure path can
    # be shown on demand instead of described.
    force_booking_failure: Optional[bool] = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    profile: Dict[str, Any]
    booking: Optional[Dict[str, Any]] = None


class SessionRequest(BaseModel):
    session_id: str


@app.get("/health")
def health() -> Dict[str, Any]:
    """Liveness plus which LLM backend is in use."""
    return {"status": "ok", "llm": describe_backend(), "sessions": len(store)}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """One customer turn in, one agent reply out."""
    session = store.get_or_create(request.session_id)
    if request.force_booking_failure is not None:
        session.force_booking_failure = request.force_booking_failure

    result = handle_turn(session, request.message.strip())

    booking_payload = None
    if result.booking is not None:
        booking_payload = {
            "ok": result.booking.ok,
            "status": result.booking.status,
            "confirmation_id": result.booking.confirmation_id,
            "when": result.booking.when,
            "reason": result.booking.reason,
        }

    return ChatResponse(
        session_id=session.session_id,
        reply=result.reply,
        profile=vars(session.profile).copy(),
        booking=booking_payload,
    )


@app.post("/end")
def end_chat(request: SessionRequest) -> Dict[str, Any]:
    """End a conversation and return the analytics JSON for it."""
    session = store.get(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    session.ended = True
    return {"session_id": session.session_id, "analytics": analyse(session)}


@app.get("/analytics/{session_id}")
def get_analytics(session_id: str) -> Dict[str, Any]:
    """Analytics for a session, on demand, without ending it."""
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    return {"session_id": session.session_id, "analytics": analyse(session)}


@app.get("/session/{session_id}")
def get_session(session_id: str) -> Dict[str, Any]:
    """Full session state: history plus the profile built up so far."""
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    return session.to_dict()


@app.post("/reset")
def reset(request: SessionRequest) -> Dict[str, str]:
    """Forget a session and free its slots on the simulated calendar."""
    store.reset(request.session_id)
    reset_calendar()
    return {"status": "reset", "session_id": request.session_id}


@app.get("/")
def index() -> FileResponse:
    """The chat UI: one self-contained HTML file, no build step."""
    return FileResponse(BASE_DIR / "index.html")
