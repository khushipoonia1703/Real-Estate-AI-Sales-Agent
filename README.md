# Northstar Homes — AI Sales Agent

An AI sales consultant, **Ava**, for a fictional real-estate developer **Northstar Homes**, selling one project: **Northstar One**, Sector 79, Gurugram.

She holds a natural conversation in **English, Hindi and Hinglish**, answers only from a closed fact sheet, qualifies the lead, books a site visit, handles a failed booking honestly, and — after the chat — emits **structured analytics** for the sales team.

FastAPI backend, Groq for inference, a single-page chat UI, no build step, no database.

> **This is a prompt-engineering project.** `system_prompt.md` is the product. The code exists to make that prompt observable, testable and demonstrable — not the other way around.

---

## 1. Quick start

```bash
# 1. install
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt

# 2. configure (optional — the app runs without a key)
cp .env.example .env            # then paste your GROQ_API_KEY

# 3. run
uvicorn main:app --reload       # http://localhost:8000

# 4. test (offline, deterministic, no key needed)
LLM_MOCK=1 pytest -q            # Windows: set LLM_MOCK=1 && pytest -q
```

Open <http://localhost:8000>, chat in any of the three languages, then click **End chat & see analytics**.

**No API key?** The app automatically falls back to a deterministic offline mock, and the header shows `offline mock mode`. Everything works — chat, memory, booking, booking failure, analytics.

### Files

```
main.py            the entire application: routes, Groq client, session memory,
                   booking simulation, agent orchestration, analytics extraction
system_prompt.md   the agent's brain — the actual deliverable
index.html         the chat UI (CSS + JS inline, no framework, no build)
test_bot.py        scenario tests: input / expected behaviour / actual output
requirements.txt
.env.example       placeholders only; .env is git-ignored and never committed
README.md          this file
```

Running the tests also writes `scenario_report.md` — a readable input / expected / actual table for every scenario.

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | — | Groq key. Absent ⇒ automatic offline mock mode. |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Model override. |
| `GROQ_BASE_URL` | `https://api.groq.com/openai/v1` | Swap this to change provider. |
| `LLM_MOCK` | `0` | `1` forces the deterministic offline mock. |
| `LLM_TEMPERATURE` | `0.4` | Sampling temperature. |
| `LLM_MAX_TOKENS` | `400` | Keeps turns short — a voice-safety measure. |
| `BOOKING_FORCE_FAILURE` | `0` | `1` makes every booking fail, to demo the failure path. |

### API

| Endpoint | Purpose |
|---|---|
| `GET /` | the chat UI |
| `POST /chat` | `{message, session_id?, force_booking_failure?}` → reply + profile + booking result |
| `POST /end` | `{session_id}` → end the conversation, return analytics JSON |
| `GET /analytics/{session_id}` | analytics on demand, without ending |
| `GET /session/{session_id}` | full history + learned profile (useful for demos) |
| `POST /reset` | forget a session, clear the simulated calendar |
| `GET /health` | liveness + which LLM backend is live |

---

## 2. The fact sheet — the only facts the agent may state

| Field | Value |
|---|---|
| Developer | Northstar Homes |
| Project | Northstar One |
| Location | Sector 79, Gurugram |
| Configurations | 2 BHK and 3 BHK |
| 2 BHK price | ₹1.35 crore onwards |
| 3 BHK price | ₹1.75 crore onwards |
| Languages | English, Hindi, Hinglish |
| Agent goal | understand needs → answer → qualify → arrange a site visit |

Everything else — carpet area, floor plans, possession date, amenities, payment plans, discounts, EMI, RERA, availability — is **unknown** and must be deferred to a human. This is the single most important rule in the project, and it is enforced in the prompt, in the analytics contract, and in the tests.

### The neighbourhood (prompt §1.1)

We know the project is in Sector 79, so the agent is allowed to talk about the area — but only in the terms that are actually true without research:

**She may say:** it is in Sector 79, Gurugram; it is one of Gurugram's newer planned residential sectors; there are schools, hospitals, markets, malls and offices in the sector and the ones around it; it is connected by road to the rest of Gurugram and towards Delhi; the area is still developing.

**She may never say:** the *name* of any specific mall, school, hospital, road, highway or metro station; any *distance* ("two km from", "walking distance"); any *travel time* ("fifteen minutes to Cyber City"); anything about metro lines or stations; anything about future infrastructure or appreciation.

Distance and connectivity claims are the most over-claimed thing in Indian real estate and a wrong one is a misleading advertisement, so a specific question ("how far is the metro?") is handled exactly like any other unknown: no number, offer to have the team confirm, and offer the site visit — where the customer judges the drive themselves.

To let the agent name real places, add them to the **VERIFIED LANDMARKS** list in `system_prompt.md` §1.1. It ships empty, and anything not on it stays unknown. Add names only, no distances — traffic makes those wrong.

### Non-negotiable guardrails

1. **No fabrication.** Never invent prices, discounts, availability, sizes, dates or specs — not in the prompt, the code, the tests or the demo data.
2. **One prompt, two channels.** Chat and voice share one brain: short turns, one question at a time, speakable numbers, no markdown / emojis / bullets in agent replies.
3. **Mirror the customer's language.** Never force one.
4. **Respect opt-out and "later" immediately.**
5. **Booking can fail.** Never fake a successful booking.
6. **Analytics is a separate call** with a strict JSON contract; JSON never appears in a conversational reply.
7. **No secrets in git.** Only `.env.example` is committed.

---

## 3. Design

### 3.1 Principles

1. **Grounded, not generative, on facts.** The model may be fluent about *how* it talks, never about *what* is true. Facts come from an explicit, closed list; everything else is deferred to a human.
2. **One prompt, two channels.** We write for voice — short, speakable, no markup, one question per turn — because voice is the stricter constraint, and that style reads well in chat too.
3. **Mirror the customer's language.** We do not pick a language; we match theirs and switch when they switch.
4. **Goal-directed, not pushy.** Every turn nudges toward qualification and a site visit, while respecting disinterest, "later" and opt-out immediately.
5. **Behaviour is specified, not hoped for.** Each required situation has an explicit rule *and* an example, so behaviour is reproducible and testable rather than emergent.
6. **Fail safe.** When unsure, out of scope, or facing an upset customer: slow down, tell the truth ("let me have a colleague confirm that"), escalate rather than fabricate.

### 3.2 System-prompt architecture

`system_prompt.md` is composed of labelled sections so it can be reasoned about, diffed and tested:

1. **Role** — who Ava is, channel-agnostic, warm and brief.
2. **Prime directive** — the closed fact sheet, plus an explicit list of everything she does *not* know and must never estimate.
3. **Goal ladder** — understand → answer → qualify → book, and what "qualified" means.
4. **Language policy** — detect and mirror English / Hindi (Devanagari *or* Roman) / Hinglish; switch silently mid-conversation.
5. **Style contract** — 1–3 sentences, one question per turn, plain speech only, speakable numbers ("one point three five crore"), confirm details back.
6. **Qualification framework** — what to collect and how to collect it conversationally, never as an interrogation.
7. **Behaviour playbook** — one rule plus one worked example for every required situation.
8. **Objection playbook** — acknowledge → reframe (grounded) → redirect.
9. **Booking protocol** — what to collect, read-back, the control line, and the failure posture.
10. **Escalation and opt-out rules.**
11. **Hard guardrails** — no advice, no pressure tactics, no competitor disparagement, stay on topic.
12. **Worked examples + a pre-reply checklist.**

**Techniques used:** role + closed-world grounding to suppress hallucination; few-shot behavioural anchors for the tricky cases (mixed-language, objection, opt-out, booking failure) so the model imitates a pattern rather than interpreting a rule; concrete, few negative constraints ("never invent…", "do not stack questions"); a channel-aware style contract so output is voice-safe by construction; separation of conversation and analytics into two prompts with two output contracts; and language mirroring by instruction plus example rather than by a classifier, because classifiers handle Hinglish badly.

### 3.3 Conversation model

Goal-driven, not a decision tree:

```
GREET → DISCOVER → QUALIFY → ADDRESS (answer / objection)
                                  ↓
                          INVITE SITE VISIT → BOOK → CONFIRM
                                  ↓                    ↓
                            (later / no)        BOOKING FAILURE → alternative slot
                                  ↓                                or human callback
                            CLOSE (polite ending / opt-out / escalate)
```

Phases overlap and loop — a customer may object mid-booking or ask a new question after qualifying. The prompt describes the *intent* of each phase rather than forcing linear progress.

### 3.4 Memory

An in-memory dict keyed by `session_id` holds the full message history plus a small `profile` (name, phone, language, configuration, budget signal, purpose, timeline, site-visit status, opt-out and escalation flags). History is replayed to the model every turn, so the agent genuinely remembers what was said. The profile is also injected as a short "what you already know" block so the agent never re-asks something it has been told.

Deliberately not a database: the brief values simplicity, and a dict shows the requirement working. Swapping in Redis or Postgres means replacing one class.

### 3.5 Behaviour playbook

| # | Situation | Designed behaviour |
|---|---|---|
| 1 | Natural conversation | Warm, concise, human. One question at a time. No "How may I assist you." |
| 2 | Customer qualification | BANT-style discovery woven into the chat: budget, configuration, purpose, timeline, financing. Never an interrogation. |
| 3 | English / Hindi / Hinglish | Detect and mirror language *and* script; switch mid-conversation if they do. |
| 4 | Common objections | Acknowledge → reframe with grounded value → redirect to a site visit. Never argue. |
| 5 | Busy / uninterested | Respected immediately. One short value hook at most, then ask for a better time or let go. |
| 6 | Contact later | Confirm and capture preferred time and channel; `follow_up_required=true`; end warmly. |
| 7 | Stop contacting me | Acknowledge once, confirm suppression, stop selling entirely. `do_not_contact=true`. No guilt-tripping. |
| 7b | Neighbourhood questions | May describe Sector 79 as a newer planned sector with schools, hospitals, markets and malls around it. Never a landmark name, a distance or a travel time unless it is on the verified list. |
| 8 | Unknown questions | Never guess. "I don't want to give you the wrong figure — let me get that confirmed." Offer follow-up or escalation, and hold the line if pushed. |
| 9 | Site-visit booking | Collect name, slot, phone — one at a time — read back, confirm, then book. |
| 10 | Booking failure | Detect it, apologise once, **never** claim success, offer an alternative slot or a human callback. |
| 11 | Human escalation | On request, frustration, price negotiation, legal/loan questions, or repeated failure. Capture a callback. |
| 12 | Proper ending | Restate the commitment, thank them, close cleanly. |
| — | No fabrication (global) | Overrides everything. |

### 3.6 Objection handling

Pattern: **acknowledge → reframe (grounded) → redirect (to a low-commitment visit)**.

| Objection | Reframe | Redirect |
|---|---|---|
| "Too expensive" | It *is* a serious amount; that is the starting price for Sector 79, Gurugram. No invented discount. | "The honest way to judge it is to see it — this weekend?" |
| "Location is far" | Honest about where it is; no fabricated connectivity or metro claims. | "Where would you be commuting from?" |
| "I'll think about it" | Respect it; find the blocker. | "What's the one thing that would help you decide?" |
| "Comparing other projects" | No disparagement of anyone. | "See ours in person before you decide." |
| "Loan / EMI / payment plan?" | Unknown → no numbers invented. | Human callback from the finance team. |
| "Give me a discount" | Not her call, and she won't make one up. | Sales manager callback. |

### 3.7 Site-visit booking and the failure path

Booking is simulated so both outcomes can be demonstrated on demand.

The agent's **only tool call** is a control line it appends to a reply once the customer has confirmed the details:

```
[[BOOK name="Rahul Verma"; phone="9876543210"; when="Saturday 5 pm"]]
```

`main.py` strips that line (it never reaches the customer), runs `simulate_booking()`, feeds the real outcome back as a system message, and lets the agent speak again. **The agent never announces a booking result it has not been told.** That single design choice is what makes the failure path honest rather than hopeful.

Failure is deterministic and triggers on: the per-session demo switch (the UI checkbox or `BOOKING_FORCE_FAILURE=1`), missing details, a same-day slot the site team can't confirm, or a slot already taken in this process. On failure the agent apologises once, offers an alternative slot or a human callback, and `site_visit_status` becomes `booking_failed`.

### 3.8 Analytics

After the chat, a **separate** LLM call with its own prompt and a strict JSON contract reads the transcript:

```json
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
  "follow_up_notes": "preferred time/channel, callback reason",
  "do_not_contact": false,
  "escalated_to_human": false,
  "contact": { "name": "string or null", "phone": "string or null" },
  "next_action": "string recommendation for the sales team"
}
```

Enums keep it CRM-ready; every field degrades to `null`/`false` when unknown, because there is no fabrication in analytics either. Two safeguards sit on top of the model:

- **Coercion, not trust.** Any out-of-enum, mistyped or missing value is forced back to the schema default. A garbage response still yields valid analytics (there is a test for exactly this).
- **Ground truth wins.** The booking system, the opt-out flag and the captured phone number override whatever the model inferred. If someone opted out, `interest_level` is `opted_out`, `follow_up_required` is `false`, and `next_action` is the suppression instruction — regardless of what the model said.

### 3.9 Architecture

```
Browser (index.html, inline CSS+JS)
   │  POST /chat {session_id, message}
   ▼
FastAPI (main.py)
   ├─ SessionStore      in-memory history + profile
   ├─ handle_turn()     assembles system prompt + context + history, calls the LLM
   ├─ llm_chat()        Groq via the OpenAI SDK, or the deterministic mock
   ├─ simulate_booking() success / failure
   └─ analyse()         separate extraction call → strict JSON
```

**Why these choices**

- **FastAPI** — required by the brief, and the cleanest small typed Python API.
- **Groq via the OpenAI-compatible endpoint** — fast (which matters for the voice story) and free-tier friendly; keeping the OpenAI interface means changing provider is a `GROQ_BASE_URL` edit, not a rewrite.
- **One file** — the whole app is ~1,300 lines including comments; for a project this size, one readable file beats seven modules and an import graph. The prompt stays in its own file because it is the deliverable.
- **In-memory sessions** — simplest thing that demonstrates the requirement.
- **Two-prompt design** — conversation and analytics have different output contracts, so JSON never leaks into speech.
- **Mock LLM mode** — the app, the demo and the full test suite work with zero API key, which de-risks a reviewer's first run and keeps CI free.

---

## 4. Tests

```bash
LLM_MOCK=1 pytest -q     # 26 tests, offline, deterministic
LIVE_TESTS=1 pytest -q   # same assertions against the real prompt on Groq
```

Tests assert on **behavioural properties**, not exact strings, so they survive rewording and are meaningful against a live model:

| Scenario | Asserted |
|---|---|
| Hinglish greeting | reply is Hinglish, one question, no markdown |
| Hindi in Devanagari | reply is in Devanagari |
| Language switch | English → Hinglish mid-conversation |
| Price question | fact-sheet price, spoken form, no `₹`, no digit grouping |
| Neighbourhood question | describes Sector 79 generally; no landmark name, no km, no travel time |
| Unknown (possession date) | no date invented, defers to the team |
| Pushed for a guess | still no number |
| Discount request | no invented offer or percentage |
| Price objection | acknowledged + redirected, no discount |
| Competitor objection | no disparagement |
| Busy / not interested | backs off, stops pitching |
| "Call me later" | `follow_up_required=true` |
| Opt-out | selling stops, `do_not_contact=true`, `interest_level=opted_out`, `follow_up_required=false` |
| Full booking flow | details collected and read back, `site_visit_status="booked"`, confirmation id |
| Forced booking failure | never claims success, apologises, offers a way forward, `booking_failed` |
| Human escalation | offers a manager, `escalated_to_human=true` |
| Memory | recalls name and 3 BHK, quotes the right price, doesn't re-ask |
| Analytics schema | every key present, all enums valid |
| Analytics with nothing shared | nulls, not guesses |
| Hostile model output | coerced back into a valid schema |
| Booking simulator | all four failure paths + success |
| Control token | parsed correctly and never leaks to the customer |
| Every reply, globally | no fabricated fact, voice-safe, ≤1 question |
| HTTP API | `/chat`, `/end`, `/analytics/{id}`, 404s, UI served, failure switch |

The run writes `scenario_report.md` with the input, the expected behaviour and the actual output for every scenario.

**About mock mode:** the mock is a small rule-based stand-in that reproduces the behaviour `system_prompt.md` specifies. Offline, the suite proves the *system* (orchestration, booking, honesty on failure, analytics contract, API) is correct; run it with `LIVE_TESTS=1` to prove the *prompt* satisfies the same assertions on a real model.

---

## 5. Verification performed

- `pytest -q` → **26 passed** in mock mode.
- `uvicorn main:app` → UI, `/health` and `/chat` all serve correctly.
- **Live Groq run** with a real key: Hinglish mirrored, possession date refused ("main aapko galat nahi batana chahti… team se confirm karwa deti hoon"), discount request deferred to a sales manager with no invented offer, full booking flow with read-back and a real confirmation, opt-out handled cleanly, and schema-valid analytics returned.

---

## 6. Assumptions

- The values in the fact sheet are **all** the product facts that exist. Anything else is out of scope and deferred to a human — including things a real agent might know (amenities, possession, RERA).
- **The neighbourhood is the one deliberate widening of that rule.** Because the sector is known, the agent may describe Sector 79's surroundings in general terms — schools, hospitals, markets, malls, offices, road connectivity. No specific landmark, distance or travel time has been verified for this project, so `VERIFIED LANDMARKS` ships empty and the agent names nothing; fill that list in and naming turns on immediately. The assumption behind the general statement is simply that a planned Gurugram residential sector has ordinary social infrastructure around it, which is true of the sector belt as a class rather than a researched claim about any one address.
- **Voice is addressed at the prompt level.** The style contract makes every reply speakable; no telephony or TTS is wired up in this take-home.
- **Booking is simulated.** No calendar or CRM integration; the failure path is deterministic so it can be demonstrated.
- The agent is female-presenting ("Ava") and introduces herself as Northstar Homes' assistant; if asked directly whether she is a bot, she says so honestly and offers a human.
- One conversation equals one session; analytics are computed per session.
- **Model availability:** Groq no longer serves `llama-3.3-70b-versatile` on every account. The documented default is kept, and if it 404s the app logs a warning and falls back to `openai/gpt-oss-120b` (then `llama-3.1-8b-instant`). Set `GROQ_MODEL` explicitly to skip the fallback. The live verification above ran on `openai/gpt-oss-120b`.
- The `[[BOOK ...]]` control line is an internal tool-call protocol, not conversational output; it is always stripped before the reply is returned, and a test asserts it can never leak.

## 7. Known limitations

- **Sessions are in memory** — they reset when the server restarts, and they do not scale across processes.
- **Analytics quality depends on the model.** Coercion and ground-truth overrides bound the damage, but a weak model can still produce a mediocre `lead_summary`.
- **Language mirroring is instruction-based**, not classifier-based. That is the right call for Hinglish, but edge cases exist. The heuristic in `detect_language()` only labels the profile and drives the mock; it never changes what a live agent says.
- **No verified local-area data ships with this repo.** The agent can describe Sector 79 categorically but cannot name a single mall, school or hospital until the sales team fills in `VERIFIED LANDMARKS`. That is a deliberate trade: a categorical answer that is always true beats a specific one that might not be.
- **No auth, no rate limiting, no persistence, no telephony.** Out of scope for a demo.
- **The mock is not the model.** Offline test passes prove the plumbing and contracts, not the prompt's fluency; that is what `LIVE_TESTS=1` is for.
- **Booking slots are free text** ("Saturday 5 pm"), not parsed dates, so there is no timezone or calendar validation.

## 8. AI tools used

- **Groq** (OpenAI-compatible API) for inference — `llama-3.3-70b-versatile` as the documented default, `openai/gpt-oss-120b` in the verified live run.
- **Claude (Anthropic), via Claude Code** — used as a pair programmer to draft and iterate on `system_prompt.md`, the FastAPI application, the deterministic mock and the test suite; every behaviour was then verified by running the suite and by live conversations against Groq.
- No other AI services, and no third-party data sources: all product facts come from the brief.
