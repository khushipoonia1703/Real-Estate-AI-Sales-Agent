"""Northstar Homes AI sales agent - the FastAPI app and the LLM layer.

This is the composition root. It owns two things:

  1. **LLM access** - the Groq client (OpenAI-compatible), the model fallback
     chain, and the entire deterministic offline mock that stands in for the
     model when there is no API key.
  2. **The HTTP surface** - the routes, and wiring the services together.

The services (chat, booking, analytics) never import this module. They receive
``llm_chat`` as an argument, which keeps the import graph one-way and lets the
whole agent be exercised offline.

Run:   uvicorn main:app --reload
Test:  LLM_MOCK=1 pytest -q
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config import FRONTEND_DIR, get_settings
from models.schemas import (
    ChatRequest,
    ChatResponse,
    LLMError,
    SessionRequest,
)
from services import analytics_service, booking_service, chat_service
from services.chat_service import (
    BUSY_PHRASES,
    COMPETITION_PHRASES,
    DISTANCE_PHRASES,
    GREETING_PHRASES,
    HUMAN_REQUEST_PHRASES,
    LATER_PHRASES,
    LOCATION_PHRASES,
    NOT_INTERESTED_PHRASES,
    OBJECTION_LOCATION_PHRASES,
    OBJECTION_PRICE_PHRASES,
    OPT_OUT_PHRASES,
    PRICE_PHRASES,
    PUSHBACK_PHRASES,
    THINKING_PHRASES,
    UNKNOWN_PHRASES,
    VISIT_PHRASES,
    affirms,
    assistant_text,
    collected,
    has_phrase,
    lang_of,
    last_user,
    qualification,
    user_messages,
)
from storage.conversation_store import store

logger = logging.getLogger(__name__)

__version__ = "1.0.0"


# =========================================================================== #
# LLM access
# =========================================================================== #

# Providers retire models. If the configured one is gone, fall back rather than
# taking the whole agent down - and say so in the log, never silently.
MODEL_FALLBACKS = ("openai/gpt-oss-120b", "llama-3.1-8b-instant")
_MODEL_GONE = ("model_not_found", "does not exist", "decommissioned", "deprecated")

_active_model: Optional[str] = None
_client = None


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


# =========================================================================== #
# Deterministic offline mock
# =========================================================================== #
#
# The mock exists so the app, the demo and the whole test suite run with no API
# key and no network. It is a small rule-based stand-in for the model, not a
# second agent: the behaviour it imitates is exactly the behaviour the system
# prompt specifies, which is what the tests assert on. Run the same tests with
# LIVE_TESTS=1 and a key to check the real prompt against them.

# Markers that identify the mock's own booking questions in the history.
_ASK_NAME_MARKERS = ("may i have your name", "aapka naam", "आपका नाम")
_ASK_SLOT_MARKERS = ("saturday work", "saturday theek", "शनिवार ठीक")
_ASK_PHONE_MARKERS = ("send the confirmation", "confirmation kis number", "पुष्टि किस नंबर")
_CONFIRM_MARKERS = ("lock that in", "confirm karun", "पुष्टि करूँ")

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


def _pick(variants: Dict[str, str], lang: str) -> str:
    return variants.get(lang) or variants["english"]


# Prices follow the channel, exactly as section 4 rule 4 of the prompt requires:
# compact and readable in chat, speakable words on a call.
_PRICES = {
    "2BHK": {
        "chat": "₹1.35 Cr",
        "voice": "one point three five crore",
        "voice_hindi": "एक पॉइंट तीन पाँच करोड़",
    },
    "3BHK": {
        "chat": "₹1.75 Cr",
        "voice": "one point seven five crore",
        "voice_hindi": "एक पॉइंट सात पाँच करोड़",
    },
}


_CHANNEL_LINE_RE = re.compile(r"^Channel:[ \t]*(chat|voice)[ \t]*$", re.MULTILINE)


def _channel_of(messages: List[Dict[str, str]]) -> str:
    """Read the channel out of the SESSION CONTEXT block in the system prompt.

    Matches only a whole line, because the prompt body itself discusses both
    `Channel: chat` and `Channel: voice` when it explains the price rule. The
    last match wins: the session context is appended after the prompt.
    """
    for msg in messages:
        if msg["role"] != "system":
            continue
        found = _CHANNEL_LINE_RE.findall(msg["content"])
        if found:
            return found[-1]
    return "chat"


def _price(config: str, channel: str, lang: str) -> str:
    entry = _PRICES[config]
    if channel == "voice":
        return entry["voice_hindi"] if lang == "hindi" else entry["voice"]
    return entry["chat"]


def _next_question(messages: List[Dict[str, str]], lang: str) -> str:
    """The one question this turn should end on: the first gap in the profile."""
    known = qualification(messages)
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

    user_text = last_user(messages)
    text = user_text.lower().strip()
    lang = lang_of(user_text)
    channel = _channel_of(messages)
    said_before = assistant_text(messages)

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
    if any(has_phrase(m.lower(), OPT_OUT_PHRASES) for m in user_messages(messages)):
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
    asked_unknown_before = "wrong figure" in said_before or "galat number" in said_before
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
        has_phrase(m.lower(), VISIT_PHRASES) for m in user_messages(messages)
    ) or has_phrase(
        said_before, _ASK_NAME_MARKERS + _ASK_SLOT_MARKERS + _ASK_PHONE_MARKERS
    )
    already_booked = any(
        "BOOKING RESULT" in m["content"] for m in messages if m["role"] == "system"
    )
    if booking_started and not already_booked:
        return _booking_turn(messages, lang, said_before)

    # 8. Price: the one thing we can always answer.
    if has_phrase(text, PRICE_PHRASES) or re.search(r"\b[23]\s*bhk|two bhk|three bhk", text):
        config = qualification(messages)["config"]
        two = _price("2BHK", channel, lang)
        three = _price("3BHK", channel, lang)
        if config == "3BHK":
            fact = {
                "english": f"Three BHKs at Northstar One start at {three}.",
                "hinglish": f"Northstar One mein 3 BHK {three} se shuru hote hain.",
                "hindi": f"नॉर्थस्टार वन में 3 BHK {three} से शुरू होते हैं।",
            }
        elif config == "2BHK":
            fact = {
                "english": f"Two BHKs start at {two}.",
                "hinglish": f"2 BHK {two} se shuru hote hain.",
                "hindi": f"2 BHK {two} से शुरू होते हैं।",
            }
        else:
            fact = {
                "english": f"Two BHKs start at {two}, and three BHKs at {three}.",
                "hinglish": f"2 BHK {two} se aur 3 BHK {three} se shuru hote hain.",
                "hindi": f"2 BHK {two} से और 3 BHK {three} से शुरू होते हैं।",
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


def _booking_turn(messages: List[Dict[str, str]], lang: str, said_before: str) -> str:
    """Collect name, then slot, then phone, then read back, then book."""
    details = collected(messages)
    asked_to_confirm = has_phrase(said_before, _CONFIRM_MARKERS)

    if asked_to_confirm and affirms(last_user(messages)) and all(details.values()):
        spoken = _pick(
            {
                "english": "Booking that for you now.",
                "hinglish": "Main abhi book kar deti hoon.",
                "hindi": "मैं अभी बुक कर देती हूँ।",
            },
            lang,
        )
        return (
            f'{spoken}\n[[BOOK name="{details["name"]}"; '
            f'phone="{details["phone"]}"; when="{details["when"]}"]]'
        )

    if not details["name"]:
        return _pick(
            {
                "english": "Happy to set that up. May I have your name, please?",
                "hinglish": "Bilkul set kar deti hoon. Aapka naam bata dijiye?",
                "hindi": "ज़रूर, मैं व्यवस्था कर देती हूँ। आपका नाम बता दीजिए?",
            },
            lang,
        )
    if not details["when"]:
        return _pick(
            {
                "english": "Great. Would this Saturday work, or is Sunday easier?",
                "hinglish": "Badhiya. Saturday theek rahega ya Sunday better hai?",
                "hindi": "बढ़िया। शनिवार ठीक रहेगा या रविवार बेहतर है?",
            },
            lang,
        )
    if not details["phone"]:
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
                f"So that is {details['name']}, {details['when']}, and I will text "
                f"the confirmation to {details['phone']}. Shall I lock that in?"
            ),
            "hinglish": (
                f"To {details['name']}, {details['when']}, aur confirmation "
                f"{details['phone']} pe bhej dungi. Main confirm karun?"
            ),
            "hindi": (
                f"तो {details['name']}, {details['when']}, और पुष्टि "
                f"{details['phone']} पर भेज दूँगी। मैं पुष्टि करूँ?"
            ),
        },
        lang,
    )


def _booking_outcome_reply(note: str, messages: List[Dict[str, str]]) -> str:
    lang = lang_of(last_user(messages))
    when = collected(messages)["when"] or "that slot"
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
# Composition: the services, with this app's LLM injected
# =========================================================================== #


def agent_turn(session, message: str, channel: str = "chat"):
    """One customer turn, with this app's LLM wired into the chat service."""
    return chat_service.handle_turn(session, message, llm_chat, channel)


def analyse_session(session) -> Dict[str, Any]:
    """Analytics for a session.

    In mock mode there is no model to ask, so the deterministic rule-based
    extraction is used - that is what passing None means to the service.
    """
    return analytics_service.analyse(session, None if is_mock() else llm_chat)


# =========================================================================== #
# FastAPI app
# =========================================================================== #

app = FastAPI(
    title="Northstar Homes AI Sales Agent",
    description="An AI sales consultant for Northstar One, Sector 79, Gurugram.",
    version=__version__,
)


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

    result = agent_turn(session, request.message.strip())
    store.save()

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
    analytics = analyse_session(session)
    store.save()
    return {"session_id": session.session_id, "analytics": analytics}


@app.get("/analytics/{session_id}")
def get_analytics(session_id: str) -> Dict[str, Any]:
    """Analytics for a session, on demand, without ending it."""
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    return {"session_id": session.session_id, "analytics": analyse_session(session)}


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
    booking_service.reset_calendar()
    store.save()
    return {"status": "reset", "session_id": request.session_id}


@app.get("/")
def index() -> FileResponse:
    """The chat UI."""
    return FileResponse(FRONTEND_DIR / "index.html")


# style.css and script.js are served from here.
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
