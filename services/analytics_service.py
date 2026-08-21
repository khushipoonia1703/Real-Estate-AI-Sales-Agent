"""Post-conversation analytics: transcript in, strict JSON out.

A separate LLM call with its own prompt and its own output contract, so a spoken
reply can never be contaminated with machine formatting. Like the chat service,
this module receives the LLM function as an argument rather than importing one.

Two things keep the output trustworthy:
  1. the schema is coerced, not trusted - unknown values fall back to null/false
     rather than to a guess;
  2. facts the system actually knows (booking outcome, opt-out, phone captured
     during booking) override whatever the model inferred.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional

from models.schemas import (
    CONFIGURATIONS,
    DNC_NEXT_ACTION,
    INTEREST_LEVELS,
    LANGUAGES,
    OBJECTIONS,
    PURPOSES,
    QUALIFICATION,
    VISIT_STATUSES,
    LLMError,
    empty_analytics,
)
from services.chat_service import (
    BUSY_PHRASES,
    COMPETITION_PHRASES,
    LATER_PHRASES,
    NOT_INTERESTED_PHRASES,
    OBJECTION_LOCATION_PHRASES,
    OBJECTION_PRICE_PHRASES,
    detect_language,
    has_phrase,
)
from storage.conversation_store import Session

LLMFunc = Callable[..., str]

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

# Machine plumbing, not product copy, so it lives next to the schema it has to
# satisfy rather than in the prompts package with the conversational prompt.
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
    out = empty_analytics()
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


def _rule_based(session: Session) -> Dict[str, Any]:
    """Deterministic extraction used when there is no live model."""
    data = empty_analytics()
    profile = session.profile
    user_msgs = [m.content for m in session.messages if m.role == "user"]
    low = " ".join(user_msgs).lower()

    languages = {detect_language(m) for m in user_msgs} or {"english"}
    data["language"] = languages.pop() if len(languages) == 1 else "mixed"

    data["budget"] = profile.budget_signal
    data["configuration_interest"] = profile.configuration_interest or (
        "undecided" if user_msgs else None
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
    elif len(user_msgs) >= 3:
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
    elif user_msgs and not profile.do_not_contact:
        data["follow_up_required"] = True
        data["next_action"] = "Follow up to arrange a site visit."

    config = data["configuration_interest"] or "no stated configuration"
    data["lead_summary"] = (
        f"Customer discussed {config} at Northstar One over {len(user_msgs)} messages; "
        f"site visit status is {data['site_visit_status']}."
        if user_msgs
        else data["lead_summary"]
    )
    return data


def analyse(session: Session, llm: Optional[LLMFunc] = None) -> Dict[str, Any]:
    """Extract structured lead data for one session. Always schema-valid.

    ``llm`` is the injected model call. Pass None (or a mock-mode function) and
    the deterministic rule-based extraction is used instead.
    """
    if not any(m.role == "user" for m in session.messages):
        return empty_analytics()

    if llm is None:
        return _apply_ground_truth(coerce(_rule_based(session)), session)

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
            raw_text = llm(
                messages, temperature=0.0, max_tokens=1500, json_mode=json_mode
            )
            match = _JSON_BLOCK_RE.search(raw_text)
            parsed = json.loads(match.group(0) if match else raw_text)
        except (LLMError, json.JSONDecodeError, AttributeError, TypeError):
            continue
        return _apply_ground_truth(coerce(parsed), session)

    # Never fail the endpoint and never invent: fall back to what we know.
    return _apply_ground_truth(coerce(_rule_based(session)), session)
