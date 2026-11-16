import json

import pytest

from chl.constraints import CONSTRAINTS, get


def ok(cid, text):
    return CONSTRAINTS[cid].check(text)[0]


@pytest.mark.parametrize("cid,good,bad", [
    ("no_emoji", "Congrats on the promotion!", "Congrats! 🎉"),
    ("no_emoji", "Arrows -> and em dashes — are fine.", "Nice work ✨"),
    ("no_digits", "seventeen times twenty-three is three hundred ninety-one", "17 x 23 = 391"),
    ("all_lowercase", "the capital of australia is canberra.", "The capital is Canberra."),
    ("word_limit", "short reply " * 30, "word " * 61),
    ("json_only", json.dumps({"reply": "hi"}), "hi"),
    ("json_only", '  {"reply": "hi"}\n', '```json\n{"reply": "hi"}\n```'),
    ("json_only", json.dumps({"reply": "hi"}), json.dumps({"reply": "hi", "extra": 1})),
    ("json_only", json.dumps({"reply": "hi"}), json.dumps({"reply": 5})),
    ("no_lists", "First you loosen the nuts, then you jack up the car.", "- loosen nuts\n- jack up car"),
    ("no_lists", "We met at 3 p.m. and talked.", "Steps:\n1. loosen\n2. lift"),
    ("no_lists", "The ratio was 2.5 to 1 overall.", "a) first\nb) second"),
    ("signoff", "Sure thing.\n— Ava", "Sure thing.\nBest, Jordan"),
    ("signoff", "Sure thing.\n- Ava\n\n", "— Ava\nSure thing."),
    ("forbidden_word", "That was my mistake; here is the fix.", "I'm so sorry about that."),
    ("forbidden_word", "Let me correct that.", "Please accept my apologies."),
])
def test_checker_pairs(cid, good, bad):
    assert ok(cid, good), f"{cid} wrongly failed: {good!r}"
    assert not ok(cid, bad), f"{cid} wrongly passed: {bad!r}"


def test_forbidden_word_is_whole_word():
    # 'sorrowful' must not trip the 'sorry' rule
    assert ok("forbidden_word", "a sorrowful tale")


def test_failures_explain_themselves():
    for c in CONSTRAINTS.values():
        passed, reason = c.check("Sorry!! 🎉 Here are 3 things:\n- A\n- B " + "word " * 70)
        if not passed:
            assert reason, f"{c.id} failed without a reason"


def test_every_rule_has_temptations():
    for c in CONSTRAINTS.values():
        assert len(c.temptations) >= 5, c.id


def test_get_parses_lists_and_rejects_unknown():
    assert [c.id for c in get("no_emoji, no_digits")] == ["no_emoji", "no_digits"]
    assert len(get("all")) == len(CONSTRAINTS)
    with pytest.raises(KeyError):
        get("nope")
