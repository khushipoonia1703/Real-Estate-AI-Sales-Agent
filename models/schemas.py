"""Shared contracts: the HTTP request/response models and the analytics schema.

Both the services and main.py import from here, and this module imports nothing
from the app, so nothing in here can create an import cycle.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class LLMError(RuntimeError):
    """Raised when the live provider cannot be reached or returns nothing.

    The LLM layer itself lives in main.py, but the services have to catch this
    without importing main - so the type is defined here, where both can see it.
    """


# --------------------------------------------------------------------------- #
# HTTP models
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Analytics schema
# --------------------------------------------------------------------------- #

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


def empty_analytics() -> Dict[str, Any]:
    """A schema-valid record that claims nothing. Every unknown degrades to this."""
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
