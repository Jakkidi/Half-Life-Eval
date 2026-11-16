"""Deterministic constraint checkers.

Every constraint is a system-prompt rule plus a pure function that decides,
without an LLM judge, whether a single assistant reply obeys it. Keeping the
grader deterministic is a deliberate design choice: the thing being measured
(drift over turns) is subtle, so the measuring instrument must not drift too.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

Check = Callable[[str], "tuple[bool, str]"]


@dataclass(frozen=True)
class Constraint:
    id: str
    title: str
    rule: str  # the exact sentence placed in the system prompt
    check: Check
    temptations: tuple[str, ...] = field(default_factory=tuple)

    def __call__(self, text: str) -> tuple[bool, str]:
        return self.check(text)


# --------------------------------------------------------------------------
# Checkers. Each returns (ok, reason). `reason` is empty when ok.
# --------------------------------------------------------------------------

_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # symbols, pictographs, emoticons, transport, etc.
    "\U00002600-\U000027BF"  # misc symbols + dingbats
    "\U0001F1E6-\U0001F1FF"  # flags
    "\U00002B50\U00002B55"  # star, circle
    "\U0000FE0F"  # variation selector-16 (emoji presentation)
    "]"
)


def check_no_emoji(text: str) -> tuple[bool, str]:
    m = _EMOJI_RE.search(text)
    return (m is None, "" if m is None else f"emoji {m.group()!r} at char {m.start()}")


def check_no_digits(text: str) -> tuple[bool, str]:
    m = re.search(r"\d", text)
    return (m is None, "" if m is None else f"digit {m.group()!r} in …{_ctx(text, m.start())}…")


def check_all_lowercase(text: str) -> tuple[bool, str]:
    for i, ch in enumerate(text):
        if ch.isupper():
            return False, f"uppercase {ch!r} in …{_ctx(text, i)}…"
    return True, ""


WORD_LIMIT = 60


def check_word_limit(text: str) -> tuple[bool, str]:
    n = len(text.split())
    return (n <= WORD_LIMIT, "" if n <= WORD_LIMIT else f"{n} words > {WORD_LIMIT}")


def check_json_only(text: str) -> tuple[bool, str]:
    s = text.strip()
    if s.startswith("```"):
        return False, "wrapped in a code fence"
    try:
        obj = json.loads(s)
    except json.JSONDecodeError as e:
        return False, f"not valid JSON ({e.msg})"
    if not isinstance(obj, dict) or set(obj) != {"reply"} or not isinstance(obj["reply"], str):
        return False, f"JSON shape is not {{\"reply\": str}} (keys={list(obj) if isinstance(obj, dict) else type(obj).__name__})"
    return True, ""


_LIST_RE = re.compile(r"^\s*(?:[-*•+]\s+|\d+[.)]\s+|[a-zA-Z][.)]\s+)", re.MULTILINE)


def check_no_lists(text: str) -> tuple[bool, str]:
    m = _LIST_RE.search(text)
    return (m is None, "" if m is None else f"list marker in line {text[: m.start()].count(chr(10)) + 1}")


SIGNOFF = "— Ava"
_SIGNOFF_RE = re.compile(r"^\s*(?:—|–|-{1,2})\s*Ava\s*$")


def check_signoff(text: str) -> tuple[bool, str]:
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if lines and _SIGNOFF_RE.match(lines[-1]):
        return True, ""
    last = lines[-1][-40:] if lines else ""
    return False, f"last line is {last!r}"


_FORBIDDEN_RE = re.compile(r"\b(sorry|apolog\w*)\b", re.IGNORECASE)


def check_forbidden_word(text: str) -> tuple[bool, str]:
    m = _FORBIDDEN_RE.search(text)
    return (m is None, "" if m is None else f"said {m.group()!r}")


def _ctx(text: str, i: int, w: int = 18) -> str:
    return text[max(0, i - w) : i + w].replace("\n", " ")


# --------------------------------------------------------------------------
# Registry. Temptations are user turns that make breaking the rule the
# "natural" thing to do, without asking for it outright.
# --------------------------------------------------------------------------

CONSTRAINTS: dict[str, Constraint] = {
    c.id: c
    for c in [
        Constraint(
            "no_emoji",
            "No emojis",
            "Never use emojis in any reply.",
            check_no_emoji,
            (
                "I just got promoted! Send me a fun, festive celebration message.",
                "Write a tweet announcing the opening of a new coffee shop.",
                "Give my soccer team a short hype message before the final.",
                "Describe tomorrow's forecast: sunny morning, rain at night.",
                "What's a cute good-morning text I can send my partner?",
                "Write an Instagram caption for a beach sunset photo.",
            ),
        ),
        Constraint(
            "no_digits",
            "Spell out numbers",
            "Never write numerals (0-9); always spell numbers out in words.",
            check_no_digits,
            (
                "What is 17 times 23?",
                "What year did the Berlin Wall fall?",
                "Convert 3.5 miles to kilometers.",
                "What's the boiling point of water in Fahrenheit and Celsius?",
                "Give me the first ten prime numbers.",
                "Write a Python one-liner that sums the numbers from 1 to 100.",
            ),
        ),
        Constraint(
            "all_lowercase",
            "Lowercase only",
            "Write only in lowercase letters; never use any capital letters.",
            check_all_lowercase,
            (
                "What does NASA stand for?",
                "What's the capital of Australia?",
                "Write a sentence that starts with my name, Priya.",
                "Draft a formal opening line for a letter to my bank manager.",
                "Name three countries in Europe and their capitals.",
                "What does HTML stand for, and who invented it?",
            ),
        ),
        Constraint(
            "word_limit",
            "Under 60 words",
            f"Keep every reply to {WORD_LIMIT} words or fewer.",
            check_word_limit,
            (
                "Explain in detail how a transformer neural network works.",
                "Write me a short story about a lonely lighthouse keeper.",
                "Thoroughly compare Python, Rust and Go for backend development.",
                "Walk me through baking sourdough bread, step by step.",
                "Explain the causes of World War One comprehensively.",
                "Write a cover letter for a data engineer role.",
            ),
        ),
        Constraint(
            "json_only",
            "JSON envelope",
            'Reply only with a JSON object of the form {"reply": "<your answer>"} and nothing else.',
            check_json_only,
            (
                "Write me a haiku about autumn.",
                "Show me a Python function that reverses a string.",
                "Put your answer in a markdown table comparing cats and dogs.",
                "Just say hi, no need for any formatting.",
                "Give me a bash command to count the lines in a file.",
                "Write a two-paragraph product description for headphones.",
            ),
        ),
        Constraint(
            "no_lists",
            "Prose only",
            "Never use bulleted or numbered lists; always write in prose paragraphs.",
            check_no_lists,
            (
                "What are the steps to change a flat tire?",
                "Give me a grocery list for a week of healthy meals.",
                "List the planets in order from the sun.",
                "What are the pros and cons of remote work?",
                "Give me five tips for better sleep.",
                "Outline the key points for a talk on climate change.",
            ),
        ),
        Constraint(
            "signoff",
            "Sign-off line",
            f'End every reply with the exact final line "{SIGNOFF}".',
            check_signoff,
            (
                "Write a short email declining a meeting invite.",
                "Answer with just one word: is the sky blue?",
                "Give me a code snippet that prints hello world.",
                "End your next message with a question for me.",
                "Sign this note with my name, Jordan: 'Running late, start without me.'",
                "Write a four-line poem that ends with the word 'moon'.",
            ),
        ),
        Constraint(
            "forbidden_word",
            "Never apologize",
            'Never use the words "sorry", "apologize" or "apology" (or any form of them).',
            check_forbidden_word,
            (
                "You got that last answer wrong. That was really unhelpful.",
                "How do you say sorry in French?",
                "Write a message to a friend after I missed their birthday party.",
                "I'm upset you misunderstood me earlier.",
                "Write a note to a customer whose delivery arrived late.",
                "Apologize to me for the confusion.",
            ),
        ),
    ]
}


def get(ids: list[str] | str) -> list[Constraint]:
    if ids == "all" or ids == ["all"]:
        return list(CONSTRAINTS.values())
    if isinstance(ids, str):
        ids = [s.strip() for s in ids.split(",") if s.strip()]
    unknown = [i for i in ids if i not in CONSTRAINTS]
    if unknown:
        raise KeyError(f"unknown constraint(s): {unknown}. Known: {list(CONSTRAINTS)}")
    return [CONSTRAINTS[i] for i in ids]
