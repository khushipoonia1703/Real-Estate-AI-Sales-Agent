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
  const backendPill = document.getElementById("backend");

  const OPENING_LINE =
    "Hi, this is Ava from Northstar Homes, about Northstar One in Sector 79, Gurugram. Are you looking at a two BHK or a three BHK?";

  let sessionId = null;

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
      analyticsJson.textContent = JSON.stringify(data.analytics, null, 2);
      analyticsBox.classList.remove("hidden");
      analyticsBox.scrollIntoView({ behavior: "smooth", block: "end" });
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

  input.focus();
})();
