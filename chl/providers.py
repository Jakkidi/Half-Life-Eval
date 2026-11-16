"""Model providers. Standard library only.

A provider takes a chat history and returns a `Reply`. Adding a provider
(OpenAI, Anthropic, vLLM...) means writing one class with a `chat` method.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field


@dataclass
class Reply:
    text: str
    latency_s: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    meta: dict = field(default_factory=dict)


class Provider:
    name = "base"

    def chat(self, model: str, messages: list[dict], *, seed: int, temperature: float,
             hint: dict | None = None) -> Reply:
        """`hint` carries episode context. Real providers ignore it; the mock uses it."""
        raise NotImplementedError

    def context_window(self, model: str) -> int | None:
        return None


# --------------------------------------------------------------------------
# Ollama
# --------------------------------------------------------------------------


class OllamaProvider(Provider):
    """Talks to a local Ollama server via /api/chat.

    Ollama's default context window is small (2k-4k tokens depending on
    version) and it truncates the *start* of the prompt silently when it
    overflows -- which deletes the system prompt. That would make every model
    look like it "forgets" its rule. We therefore set `num_ctx` explicitly and
    log prompt token counts so the report can flag at-risk turns.

    Replies are *streamed*. With `stream: false` Ollama sends nothing until the
    whole reply is generated, so on a slow CPU a long reply late in a long
    conversation can exceed any fixed timeout, and each retry starts the
    generation over. Streaming means `timeout` only measures silence: the call
    fails if no new text arrives for `timeout` seconds, however long the reply.
    """

    name = "ollama"

    def __init__(self, host: str = "http://localhost:11434", num_ctx: int = 16384,
                 timeout: float = 300.0, retries: int = 3):
        self.host = host.rstrip("/")
        self.num_ctx = num_ctx
        self.timeout = timeout  # seconds of silence (no new tokens) before giving up
        self.retries = retries

    def context_window(self, model: str) -> int | None:
        return self.num_ctx

    def list_models(self) -> list[str]:
        req = urllib.request.Request(self.host + "/api/tags")
        with urllib.request.urlopen(req, timeout=10) as r:
            return [m["name"] for m in json.loads(r.read()).get("models", [])]

    def _stream_chat(self, payload: dict) -> tuple[str, dict]:
        """POST /api/chat with stream=true; return (full text, final chunk with token counts)."""
        req = urllib.request.Request(
            self.host + "/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        parts: list[str] = []
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            for raw in r:  # one JSON object per line
                line = raw.strip()
                if not line:
                    continue
                chunk = json.loads(line)
                if "error" in chunk:
                    raise _OllamaError(chunk["error"])
                parts.append(chunk.get("message", {}).get("content", ""))
                if chunk.get("done"):
                    return "".join(parts), chunk
        raise ConnectionError("stream ended before the reply was complete")

    def chat(self, model, messages, *, seed, temperature, hint=None) -> Reply:
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature, "seed": seed, "num_ctx": self.num_ctx},
        }
        last_err: Exception | None = None
        for attempt in range(self.retries):
            t0 = time.perf_counter()
            try:
                text, final = self._stream_chat(payload)
                return Reply(
                    text=text,
                    latency_s=time.perf_counter() - t0,
                    prompt_tokens=final.get("prompt_eval_count"),
                    completion_tokens=final.get("eval_count"),
                )
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace")[:300]
                if 400 <= e.code < 500:  # bad request / model not found: retrying won't help
                    raise RuntimeError(f"Ollama rejected the request (HTTP {e.code}): {body}") from e
                last_err = e
            except _OllamaError as e:
                raise RuntimeError(f"Ollama reported an error: {e}") from e
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
                last_err = e
            if attempt < self.retries - 1:
                time.sleep(2 ** attempt)
        raise RuntimeError(
            f"Ollama call failed after {self.retries} attempts ({last_err}). "
            f"Is `ollama serve` running at {self.host} and is '{model}' pulled? "
            f"If the model went silent for over {self.timeout:.0f}s, raise --timeout. "
            "Finished conversations are saved; re-run the same command to resume."
        )


class _OllamaError(Exception):
    """An error message Ollama sent inside the stream."""


# --------------------------------------------------------------------------
# Mock: a simulated model with a known ground-truth hazard.
# --------------------------------------------------------------------------

# Per-constraint baseline difficulty (log-odds offset). Higher = breaks sooner.
_MOCK_DIFFICULTY = {
    "no_emoji": -0.6, "no_digits": 0.5, "all_lowercase": 0.3, "word_limit": 0.6,
    "json_only": 0.1, "no_lists": 0.2, "signoff": 0.4, "forbidden_word": -0.4,
}

MOCK_MODELS = {
    # name: (base log-odds, per-turn drift, tempt bonus, adversarial bonus, contagion, reminder effect)
    "mock-steady": (-5.2, 0.05, 1.2, 2.2, 1.2, -1.0),
    "mock-drifty": (-4.4, 0.16, 1.6, 2.8, 2.4, -1.8),
}

_COMPLIANT = "here is a short answer in plain words, kept simple and clear for you."
_VIOLATIONS = {
    "no_emoji": "here is a short answer in plain words 🎉 kept simple and clear.",
    "no_digits": "here is a short answer: roughly 42 of them, kept simple and clear.",
    "all_lowercase": "Sure! Here is a short answer in plain words, kept simple and clear.",
    "word_limit": " ".join(["this answer keeps going with more and more detail"] * 10),
    "json_only": "Sure, here is the answer in plain text rather than JSON.",
    "no_lists": "here is a short answer:\n- first point\n- second point",
    "signoff": "here is a short answer in plain words, kept simple and clear.\nCheers!",
    "forbidden_word": "sorry about that, here is a short answer in plain words.",
}


class MockProvider(Provider):
    """Deterministic simulated model for testing the pipeline offline.

    Violation probability is a logistic function of turn index, pressure,
    whether the model already violated (contagion), and reminders. Because the
    true hazard is known, the mock doubles as a check that the statistics
    recover the effects they claim to measure. Its numbers say nothing about
    real models.
    """

    name = "mock"

    def chat(self, model, messages, *, seed, temperature, hint=None) -> Reply:
        if model not in MOCK_MODELS:
            raise KeyError(f"mock model must be one of {list(MOCK_MODELS)}")
        base, drift, tempt_b, adv_b, contagion, reminder = MOCK_MODELS[model]
        meta = hint or {}
        constraint = meta.get("constraint", "no_emoji")
        kind = meta.get("kind", "neutral")
        turn = sum(1 for m in messages if m["role"] == "user")
        prev_violated = meta.get("prev_violated", False)
        reminded = meta.get("reminded", False)

        z = base + _MOCK_DIFFICULTY.get(constraint, 0.0) + drift * turn
        z += tempt_b if kind == "tempt" else adv_b if kind == "adversarial" else 0.0
        z += contagion if prev_violated else 0.0
        z += reminder if reminded else 0.0
        p = 1 / (1 + math.exp(-z))

        h = hashlib.sha256(f"{seed}|{model}|{constraint}|{turn}|{len(messages)}".encode()).hexdigest()
        violate = random.Random(int(h[:12], 16)).random() < p
        text = _VIOLATIONS[constraint] if violate else _COMPLIANT
        if constraint == "json_only" and not violate:
            text = json.dumps({"reply": text})
        if constraint == "signoff" and not violate:
            text += "\n— Ava"
        approx_prompt = sum(len(m["content"].split()) for m in messages) * 4 // 3
        return Reply(text=text, latency_s=0.0, prompt_tokens=approx_prompt,
                     completion_tokens=len(text.split()), meta={"p_violate": round(p, 4)})


def make(name: str, **kw) -> Provider:
    if name == "ollama":
        return OllamaProvider(**{k: v for k, v in kw.items() if k in ("host", "num_ctx", "timeout")})
    if name == "mock":
        return MockProvider()
    raise KeyError(f"unknown provider {name!r} (expected 'ollama' or 'mock')")
