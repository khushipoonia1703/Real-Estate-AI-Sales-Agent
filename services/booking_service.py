"""Simulated site-visit booking.

Real booking would call a CRM or calendar API. Here it is simulated, but the
failure path is real and deterministic so the agent's honesty about failures can
be demonstrated and tested rather than described.

Failure triggers, in priority order:
1. ``force_failure`` (per-session demo switch, or BOOKING_FORCE_FAILURE=1)
2. missing name, phone or slot
3. a same-day slot ("today", "tonight") the site team cannot confirm now
4. a slot already taken in this process (double booking)
"""

from __future__ import annotations

import hashlib
import re
import threading
from dataclasses import dataclass
from typing import Dict, Optional

from config import get_settings

# slot key -> confirmation id, for the in-process calendar.
_BOOKED_SLOTS: Dict[str, str] = {}
_LOCK = threading.Lock()

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

    with _LOCK:
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
    with _LOCK:
        _BOOKED_SLOTS.clear()
