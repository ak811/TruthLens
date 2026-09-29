import json
from types import SimpleNamespace

import pytest

from truthlens.judge import (SYSTEM_PROMPT_V1, USER_TEMPLATE_V1, MockJudge, OpenAIJudge, build_messages,
                             parse_judge_response)


def released_fallback(text):
    """Fallback branch of the released safe_parse_response (for regression comparison)."""
    return "FAKE" if "fake" in text.lower() else "REAL"


@pytest.mark.parametrize("text,verdict,status", [
    ('{"verdict": "FAKE", "justification": "dilated pupils"}', "FAKE", "json"),
    ('```json\n{"verdict": "REAL", "justification": "ok"}\n```', "REAL", "json"),
    ("{'verdict': 'FAKE', 'justification': 'x'}", "FAKE", "literal"),
    ('Here is my analysis:\n{"verdict": "real", "justification": "ok"}\nThanks', "REAL", "json"),
    ('Verdict: FAKE\nJustification: skin too smooth', "FAKE", "regex"),
    ('{"Verdict": "Fake", "Confidence": "high", "justification": "x"}', "FAKE", "json"),
])
def test_parse_valid_replies(text, verdict, status):
    parsed = parse_judge_response(text)
    assert (parsed.verdict, parsed.parse_status) == (verdict, status)


def test_parse_confidence():
    p = parse_judge_response('{"verdict": "FAKE", "confidence": "MEDIUM", "justification": "x"}')
    assert p.confidence == "MEDIUM"


@pytest.mark.parametrize("text", [
    "There are no signs that this image is fake; it looks like a genuine photograph.",
    "I cannot determine whether it is real or fake.",
    '{"verdict": "UNSURE"}',
    "",
])
def test_unparseable_replies_are_not_guessed(text):
    assert parse_judge_response(text).verdict is None


def test_regression_released_fallback_mislabelled_real_as_fake():
    text = "There are no signs that this image is fake; it looks like a genuine photograph."
    assert released_fallback(text) == "FAKE"          # the bug
    assert parse_judge_response(text).parse_status == "unparseable"  # fixed behaviour


def test_messages_use_released_prompt_and_template():
    msgs = build_messages("SUMMARY")
    assert msgs[0]["content"] == SYSTEM_PROMPT_V1
    assert msgs[1]["content"] == USER_TEMPLATE_V1.format(description="SUMMARY")
    assert msgs[1]["content"] == ("Analyze this image description and determine if it's real or fake: SUMMARY")
    assert SYSTEM_PROMPT_V1.startswith("\n" + " " * 24 + "You are a forensic image analyst")
    assert '"verdict": "FAKE"' in SYSTEM_PROMPT_V1 and "5. Eye and pupil details" in SYSTEM_PROMPT_V1


class FakeClient:
    def __init__(self):
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        msg = SimpleNamespace(content='{"verdict": "REAL", "justification": "ok"}')
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], model="gpt-4-0613", system_fingerprint=None)


def test_openai_judge_omits_unset_parameters():
    client = FakeClient()
    judge = OpenAIJudge(model="gpt-4", temperature=None, client=client)
    assert parse_judge_response(judge.complete(build_messages("s"))).verdict == "REAL"
    assert set(client.calls[0]) == {"model", "messages"}
    OpenAIJudge(model="gpt-4", temperature=0.0, seed=1, client=client).complete(build_messages("s"))
    assert client.calls[1]["temperature"] == 0.0 and client.calls[1]["seed"] == 1


def test_openai_judge_requires_env_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(EnvironmentError):
        OpenAIJudge()


def test_mock_judge_is_deterministic_json():
    out = json.loads(MockJudge().complete(build_messages("skin looks unnatural")))
    assert out["verdict"] == "FAKE"
