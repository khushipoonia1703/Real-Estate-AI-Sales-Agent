"""The conversation: language handling, message parsing, and one turn of the agent.

This module owns everything that reads a customer message or assembles what the
model sees. It knows nothing about which model is being called - ``handle_turn``
receives the LLM function as an argument, so main.py can inject the live Groq
client or the offline mock without this module ever importing either.

The phrase lists and parsing helpers here are also the vocabulary used by the
deterministic mock in main.py and by the analytics fallback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from models.schemas import LLMError
from prompts.system_prompt import SYSTEM_PROMPT
from services.booking_service import BookingResult, simulate_booking
from storage.conversation_store import Session

LLMFunc = Callable[..., str]


# =========================================================================== #
# Language detection
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

DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_WORD = re.compile(r"[a-z]+")


def detect_language(text: str) -> str:
    """Best-effort label for a customer message: hindi, hinglish or english.

    This never changes what the agent says (the prompt handles mirroring); it
    only annotates the session profile and steers the deterministic mock.
    """
    if DEVANAGARI_RE.search(text):
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


def lang_of(text: str) -> str:
    """The reply language to use: romanised Hindi is answered as Hinglish."""
    lang = detect_language(text)
    return "hinglish" if lang == "hindi" and not DEVANAGARI_RE.search(text) else lang


# =========================================================================== #
# Intent vocabulary
# =========================================================================== #

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


# =========================================================================== #
# Message parsing
# =========================================================================== #

BOOK_TOKEN_RE = re.compile(r"\[\[\s*BOOK\b(?P<body>[^\]]*)\]\]", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?([6-9]\d{9})(?!\d)")
NAME_RE = re.compile(
    r"(?:my name is|name is|mera naam|naam hai|this is|i am|i'm|im)\s+"
    r"([A-Za-z]{2,20}(?:\s+[A-Za-z]{2,20})?)",
    re.IGNORECASE,
)
SLOT_RE = re.compile(
    r"((?:this |next |coming )?(?:monday|tuesday|wednesday|thursday|friday|"
    r"saturday|sunday|tomorrow|weekend|today|tonight)"
    r"(?:\s+(?:morning|afternoon|evening|night))?"
    r"(?:\s+at)?(?:\s+\d{1,2}(?::\d{2})?\s*(?:am|pm|baje|o'clock)?)?)",
    re.IGNORECASE,
)
TIME_RE = re.compile(r"\b(\d{1,2}(?::\d{2})?\s*(?:am|pm|baje))\b", re.IGNORECASE)
BUDGET_MENTION_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:crore|cr\b|lakh|lac|करोड़|लाख)", re.IGNORECASE
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


def has_phrase(text: str, needles) -> bool:
    return any(n in text for n in needles)


def tokens(text: str):
    return set(re.findall(r"[\w']+", text.lower()))


def affirms(text: str) -> bool:
    """Whole-word yes-detection, so 'what' does not count as 'ha'."""
    low = text.lower()
    if any(" " in a and a in low for a in AFFIRM_PHRASES):
        return True
    return bool(tokens(low) & {a for a in AFFIRM_PHRASES if " " not in a})


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
    if not words or tokens(candidate) & _NON_NAME_WORDS:
        return None
    return " ".join(w.capitalize() for w in words)


def last_user(messages: List[Dict[str, str]]) -> str:
    for msg in reversed(messages):
        if msg["role"] == "user":
            return msg["content"]
    return ""


def user_messages(messages: List[Dict[str, str]]) -> List[str]:
    return [m["content"] for m in messages if m["role"] == "user"]


def assistant_text(messages: List[Dict[str, str]]) -> str:
    return " ".join(m["content"].lower() for m in messages if m["role"] == "assistant")


def collected(messages: List[Dict[str, str]]) -> Dict[str, Optional[str]]:
    """Pull name, phone and slot out of everything the customer has said."""
    name = phone = slot = None
    for text in user_messages(messages):
        found_phone = PHONE_RE.search(text.replace(" ", ""))
        if found_phone:
            phone = found_phone.group(1)
        found_name = name_from(text)
        if found_name:
            name = found_name
        found_slot = SLOT_RE.search(text)
        if found_slot and found_slot.group(1).strip():
            slot = re.sub(r"\s+", " ", found_slot.group(1)).strip(" ,.")
            time_part = TIME_RE.search(text)
            if time_part and time_part.group(1).lower() not in slot.lower():
                slot = f"{slot} {time_part.group(1)}"
    return {"name": name, "phone": phone, "when": slot}


def qualification(messages: List[Dict[str, str]]) -> Dict[str, Optional[str]]:
    """Configuration, purpose and budget as stated by the customer so far."""
    low = " ".join(user_messages(messages)).lower()
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
    budget_match = BUDGET_MENTION_RE.search(low)
    return {
        "config": config,
        "purpose": purpose,
        "budget": budget_match.group(0) if budget_match else None,
    }


# =========================================================================== #
# Prompt assembly, reply hygiene, the booking control line
# =========================================================================== #

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


def messages_for(session: Session, channel: str = "chat") -> List[Dict[str, str]]:
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


# =========================================================================== #
# Profile learning
# =========================================================================== #


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


# =========================================================================== #
# One turn
# =========================================================================== #
#
# A turn is:
#     user message -> [system prompt + session context + history] -> model reply
# and, if the model emitted the booking control line, a second pass:
#     booking result -> model -> the reply the customer actually sees.
#
# The control line is the agent's only tool call. It is stripped here and never
# reaches the customer.


def handle_turn(
    session: Session,
    message: str,
    llm: LLMFunc,
    channel: str = "chat",
) -> TurnResult:
    """Run one customer turn end to end and return the reply the customer sees.

    ``llm`` is injected so this module never has to know whether it is talking
    to Groq or to the offline mock.
    """
    session.add("user", message)
    update_profile_from_user(session, message)

    try:
        raw = llm(messages_for(session, channel))
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
        reply = clean_reply(llm(messages_for(session, channel)))
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
