"""Environment and settings.

Everything configurable is read here, once, so no other module touches
os.environ. This module imports nothing from the app, which is what keeps the
import graph acyclic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
DATA_DIR = BASE_DIR / "data"

load_dotenv(BASE_DIR / ".env")


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
    groq_strict_model: bool
    llm_mock: bool
    temperature: float
    max_tokens: int
    booking_force_failure: bool
    conversations_path: Path

    @property
    def use_mock(self) -> bool:
        """Mock mode is on when forced, or whenever there is no API key."""
        return self.llm_mock or not self.groq_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings(
        groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
        groq_model=os.getenv("GROQ_MODEL", "").strip() or "openai/gpt-oss-safeguard-20b",
        groq_base_url=(
            os.getenv("GROQ_BASE_URL", "").strip() or "https://api.groq.com/openai/v1"
        ),
        # On: never let a fallback model answer. A request the configured model
        # cannot serve fails loudly instead of being billed to another model.
        groq_strict_model=_flag("GROQ_STRICT_MODEL"),
        llm_mock=_flag("LLM_MOCK"),
        temperature=_float("LLM_TEMPERATURE", 0.4),
        # 400 truncates these models mid-sentence: the cap covers their
        # internal reasoning tokens, not just the visible reply.
        max_tokens=_int("LLM_MAX_TOKENS", 900),
        booking_force_failure=_flag("BOOKING_FORCE_FAILURE"),
        conversations_path=Path(
            os.getenv("CONVERSATIONS_PATH", "").strip()
            or (DATA_DIR / "conversations.json")
        ),
    )
