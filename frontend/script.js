(() => {
  "use strict";

  const chat = document.getElementById("chat");
  const form = document.getElementById("composer");
  const input = document.getElementById("input");
  const sendBtn = document.getElementById("send");
  const endBtn = document.getElementById("endBtn");
  const resetBtn = document.getElementById("resetBtn");
  const failToggle = document.getElementById("failToggle");
  const banner = document.getElementById("banner");
  const analyticsBox = document.getElementById("analytics");
  const analyticsJson = document.getElementById("analyticsJson");
  const analyticsGrid = document.getElementById("analyticsGrid");
  const leadSummary = document.getElementById("leadSummary");
  const nextAction = document.getElementById("nextAction");
  const rawToggle = document.getElementById("rawToggle");
  const backendPill = document.getElementById("backend");
  const fab = document.getElementById("fab");
  const panel = document.getElementById("panel");
  const nudge = document.getElementById("nudge");

  const OPENING_LINE =
    "Hi, this is Ava from Northstar Homes, about Northstar One in Sector 79, Gurugram. Are you looking at a two BHK or a three BHK?";

  let sessionId = null;

  /* ------------------------------------------------------------------ */
  /* Launcher                                                            */
  /* ------------------------------------------------------------------ */

  function openPanel() {
    panel.classList.add("open");
    panel.setAttribute("aria-hidden", "false");
    fab.classList.add("open", "seen");
    fab.setAttribute("aria-expanded", "true");
    nudge.classList.add("hidden");
    input.focus();
  }

  function closePanel() {
    panel.classList.remove("open");
    panel.setAttribute("aria-hidden", "true");
    fab.classList.remove("open");
    fab.setAttribute("aria-expanded", "false");
    fab.focus();
  }

  fab.addEventListener("click", () => {
    if (panel.classList.contains("open")) closePanel();
    else openPanel();
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && panel.classList.contains("open")) closePanel();
  });

  /* ------------------------------------------------------------------ */
  /* Chat                                                                */
  /* ------------------------------------------------------------------ */

  function bubble(text, who, extraClass) {
    const wrap = document.createElement("div");
    wrap.className = `msg ${who}${extraClass ? " " + extraClass : ""}`;
    const p = document.createElement("p");
    p.textContent = text;
    wrap.appendChild(p);
    chat.appendChild(wrap);
    chat.scrollTop = chat.scrollHeight;
    return wrap;
  }

  function showBanner(text, kind) {
    banner.textContent = text;
    banner.className = `banner ${kind}`;
  }

  function clearBanner() {
    banner.className = "banner hidden";
    banner.textContent = "";
  }

  /* ------------------------------------------------------------------ */
  /* Analytics card                                                      */
  /* ------------------------------------------------------------------ */

  const EMPTY = "—"; // em dash

  // enum -> readable: "opted_out" -> "Opted out", "2BHK" stays "2BHK"
  function humanize(value) {
    if (value === null || value === undefined || value === "") return null;
    const text = String(value).replace(/_/g, " ").trim();
    if (!text) return null;
    return text.charAt(0).toUpperCase() + text.slice(1);
  }

  // Which colour a status value gets. Anything unlisted stays neutral.
  const BADGE_TONE = {
    high: "good", qualified: "good", booked: "good",
    medium: "warn", partially_qualified: "warn", proposed: "warn",
    booking_failed: "bad", opted_out: "bad",
    low: "flat", unqualified: "flat", declined: "flat", none: "flat",
  };

  function badge(value) {
    const el = document.createElement("span");
    el.className = `badge badge-${BADGE_TONE[value] || "flat"}`;
    el.textContent = humanize(value) || EMPTY;
    return el;
  }

  // Booleans read as Yes/No, coloured by whether "yes" is something to act on.
  function chip(value, yesTone) {
    const el = document.createElement("span");
    el.className = `badge badge-${value ? yesTone : "flat"}`;
    el.textContent = value ? "Yes" : "No";
    return el;
  }

  function tags(list) {
    if (!Array.isArray(list) || list.length === 0) return text("None", true);
    const wrap = document.createElement("div");
    wrap.className = "tags";
    list.forEach((item) => {
      const tag = document.createElement("span");
      tag.className = "tag";
      tag.textContent = humanize(item) || item;
      wrap.appendChild(tag);
    });
    return wrap;
  }

  function text(value, muted) {
    const el = document.createElement("span");
    const readable = muted ? value : humanize(value);
    el.textContent = readable === null ? EMPTY : readable;
    if (readable === null) el.dataset.empty = "true";
    return el;
  }

  function row(label, node) {
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    if (node.dataset && node.dataset.empty === "true") dd.className = "empty";
    dd.appendChild(node);
    analyticsGrid.appendChild(dt);
    analyticsGrid.appendChild(dd);
  }

  function renderAnalytics(data) {
    leadSummary.textContent = data.lead_summary || EMPTY;
    analyticsGrid.innerHTML = "";

    row("Interest level", badge(data.interest_level));
    row("Qualification", badge(data.qualification_status));
    row("Configuration", text(data.configuration_interest));
    row("Budget", text(data.budget));
    row("Purpose", text(data.purpose));
    row("Language", text(data.language));
    row("Objections", tags(data.objections_raised));
    row("Site visit", badge(data.site_visit_status));
    row("Site visit time", text(data.site_visit_datetime));
    row("Follow-up required", chip(data.follow_up_required, "warn"));
    row("Follow-up notes", text(data.follow_up_notes));
    row("Do not contact", chip(data.do_not_contact, "bad"));
    row("Escalated to human", chip(data.escalated_to_human, "warn"));
    row("Contact name", text((data.contact || {}).name));
    row("Contact phone", text((data.contact || {}).phone));

    nextAction.textContent = data.next_action || EMPTY;

    analyticsJson.textContent = JSON.stringify(data, null, 2);
    analyticsJson.classList.add("hidden");
    rawToggle.textContent = "Show raw JSON";
    rawToggle.setAttribute("aria-expanded", "false");

    analyticsBox.classList.remove("hidden");
    analyticsBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  rawToggle.addEventListener("click", () => {
    const hidden = analyticsJson.classList.toggle("hidden");
    rawToggle.textContent = hidden ? "Show raw JSON" : "Hide raw JSON";
    rawToggle.setAttribute("aria-expanded", hidden ? "false" : "true");
  });

  async function postJson(url, body) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res.json();
  }

  async function send(message) {
    bubble(message, "user");
    const typing = bubble("Ava is typing…", "agent", "typing");
    sendBtn.disabled = true;

    try {
      const data = await postJson("/chat", {
        message,
        session_id: sessionId,
        force_booking_failure: failToggle.checked,
      });
      sessionId = data.session_id;
      typing.remove();
      bubble(data.reply, "agent");

      if (data.booking) {
        if (data.booking.ok) {
          showBanner(
            `Site visit booked — ${data.booking.when} · ref ${data.booking.confirmation_id}`,
            "ok"
          );
        } else {
          showBanner(`Booking failed — ${data.booking.reason}`, "bad");
        }
      }
    } catch (err) {
      typing.remove();
      bubble(`Could not reach the server (${err.message}).`, "agent");
    } finally {
      sendBtn.disabled = false;
      input.focus();
    }
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const message = input.value.trim();
    if (!message) return;
    input.value = "";
    send(message);
  });

  endBtn.addEventListener("click", async () => {
    if (!sessionId) {
      showBanner("Say something first, then end the chat.", "bad");
      return;
    }
    endBtn.disabled = true;
    try {
      const data = await postJson("/end", { session_id: sessionId });
      renderAnalytics(data.analytics);
    } catch (err) {
      showBanner(`Could not load analytics (${err.message}).`, "bad");
    } finally {
      endBtn.disabled = false;
    }
  });

  resetBtn.addEventListener("click", async () => {
    if (sessionId) {
      try {
        await postJson("/reset", { session_id: sessionId });
      } catch (err) {
        /* a fresh session id is enough */
      }
    }
    sessionId = null;
    chat.innerHTML = "";
    bubble(OPENING_LINE, "agent");
    analyticsBox.classList.add("hidden");
    analyticsJson.textContent = "";
    analyticsJson.classList.add("hidden");
    analyticsGrid.innerHTML = "";
    leadSummary.textContent = "";
    nextAction.textContent = "";
    rawToggle.textContent = "Show raw JSON";
    rawToggle.setAttribute("aria-expanded", "false");
    clearBanner();
    input.focus();
  });

  fetch("/health")
    .then((res) => res.json())
    .then((data) => {
      backendPill.textContent =
        data.llm.mode === "mock" ? "offline mock mode" : `groq · ${data.llm.model}`;
    })
    .catch(() => {
      backendPill.textContent = "server unreachable";
    });
})();
