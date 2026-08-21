# Northstar Homes — AI Sales Agent

An AI sales consultant ("**Ava**") for a fictional real-estate project, **Northstar One** in Sector 79, Gurugram. Ava chats with prospective buyers in **English, Hindi, and Hinglish**, answers only from a fixed fact sheet, qualifies the lead, books a site visit, and — after the conversation — produces structured analytics for the sales team.

The focus of this project is **prompt engineering and agent behaviour**. The system prompt (`prompts/system_prompt.py`) is the heart of the solution; the code around it exists to make that prompt observable, testable, and demonstrable.

Backend: **FastAPI (Python)**. LLM: **Groq** (Llama 3.3 / GPT-OSS) via its OpenAI-compatible API.

---

## How to run the bot

**Prerequisites:** Python 3.11+ and a free Groq API key from https://console.groq.com/keys

```bash
# 1. From the project folder, create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up your environment file
copy .env.example .env      # Windows   (macOS/Linux: cp .env.example .env)
# then open .env and paste your GROQ_API_KEY

# 4. Start the server
uvicorn main:app --reload
```

Then open **http://localhost:8000** in your browser.

**Windows note:** if you see `WinError 10013` (socket access forbidden), port 8000 is reserved on your machine. Just pick another port:

```bash
uvicorn main:app --reload --port 8010
```

and open the matching URL (http://localhost:8010).

**Check the backend:** open `http://localhost:8000/health` — it shows whether you're on live Groq (`"mode": "groq"`) or the offline mock, and which model is active.

### Running the tests

The whole scenario suite runs offline, with no API key needed:

```bash
# Windows:
set LLM_MOCK=1
pytest -q
# macOS / Linux:
LLM_MOCK=1 pytest -q
```

This also refreshes `scenario_report.md` with the input / expected behaviour / actual output for each scenario.

---

## Project structure

```
main.py                        FastAPI app + routes + LLM access (Groq) + offline mock
config.py                      Settings / environment variables (read once)
prompts/system_prompt.py       THE PROMPT — the agent's brain (the deliverable)
services/
  chat_service.py              Turn orchestration: prompt assembly, memory update, reply cleaning
  booking_service.py           Simulated site-visit booking (with a real failure path)
  analytics_service.py         Post-conversation analytics (separate LLM call, strict JSON)
models/schemas.py              API request/response models + analytics schema
storage/conversation_store.py  Conversation state (in-memory + JSON persistence)
data/conversations.json        Persisted conversations
frontend/                      index.html, style.css, script.js — minimal chat UI
tests/test_scenarios.py        Scenario tests (input / expected / actual)
.env.example                   Environment template
```

## How it works (in brief)

Each customer message hits `POST /chat`. The app appends it to that session's history, updates a small structured **profile** (name, language, configuration, budget, purpose, opt-out, etc.), assembles the system prompt plus the full conversation history, and calls Groq. Booking uses a text control line (`[[BOOK ...]]`) the model emits and the server executes, so the agent can never fake a booking. When the chat ends (`POST /end`), a **separate** LLM call extracts structured analytics as JSON — kept apart so conversational replies never contain machine formatting.

---

## Key assumptions

- **The fact sheet is the whole truth.** The only property facts Ava may state are the ones in the prompt: the project name, Sector 79 Gurugram, 2 BHK and 3 BHK, and the two starting prices (₹1.35 crore / ₹1.75 crore). Everything else (carpet area, possession date, amenities, EMI, discounts, availability) is treated as unknown and deferred to the human team.
- **Neighbourhood landmarks are a team-verified list.** Ava may name a few nearby schools, malls, markets, and hospitals from an explicit verified list in the prompt, but never invents distances, travel times, or places outside that list.
- **Voice + chat share one prompt.** The prompt is written to be voice-safe (short, speakable, one question per turn); actual telephony/TTS is not wired up in this take-home — voice support is addressed at the prompt level.
- **Booking is simulated.** `booking_service.py` mimics a booking system, including a deterministic failure path, rather than integrating a real calendar/CRM.
- **Groq is the LLM provider.** Used through its OpenAI-compatible API, so the provider is swappable. The default model `llama-3.3-70b-versatile` returns `model_not_found` on some Groq keys; the app automatically falls back to `openai/gpt-oss-120b`.

## Known limitations

- **Storage is lightweight.** Conversations live in memory and are persisted to a JSON file (`data/conversations.json`), not a real database. Fine for a demo; swapping in Redis/Postgres means replacing one module.
- **The offline mock is a separate imitation.** For zero-key running and tests, a deterministic rule-based mock stands in for the model. It mirrors the prompt's intended behaviour but is hand-written, so it won't reproduce every nuance of the live LLM (for example, the verified-landmark answers appear on the live Groq bot, not in mock mode).
- **Language mirroring is instruction-based.** English/Hindi/Hinglish detection for the live agent is handled by the prompt, not a hard classifier — the right call for Hinglish, but edge cases are possible.
- **Analytics quality depends on the model.** Extracted fields are coerced to a fixed schema and overridden by known system facts (booking outcome, opt-out) so they never fabricate, but the free-text summary quality tracks the model.
- **No auth or rate-limiting.** Out of scope for a demo; do not expose the endpoints publicly as-is.

## AI tools used

- **Groq** — LLM inference (Llama 3.3 70B, with `openai/gpt-oss-120b` as the working fallback) via the OpenAI-compatible API.
- **OpenAI Python SDK** — as the client library pointed at Groq's endpoint.
- **Claude (Anthropic)** — used as an AI pair-programmer to help design the prompt, structure the code, and write the tests and documentation.
