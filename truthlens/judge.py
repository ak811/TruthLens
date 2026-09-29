"""Step 4 - final decision ``(y, r) = f_LM(S)`` (paper Section 2.4).

Changes relative to the released ``evaluation_gpt.py`` (see docs/AUDIT.md):

* migrated from the removed ``openai==0.28`` ``ChatCompletion`` API to
  ``openai>=1.0``; the API key is read from ``OPENAI_API_KEY`` only (it was a
  command-line argument, which leaks into shell history and process lists);
* the model is configurable; the released script used ``gpt-3.5-turbo`` while
  the paper states GPT-4;
* response parsing no longer falls back to ``"fake" in response.lower()``,
  which labelled any reply mentioning the word "fake" (e.g. "no signs of being
  fake") as FAKE. Unparseable replies are now recorded as such;
* transient API failures are retried by the client; persistent failures are
  recorded per image instead of being silently counted as REAL.
"""
from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

VERDICTS = ("REAL", "FAKE")
CONFIDENCES = ("HIGH", "MEDIUM", "LOW")

# --------------------------------------------------------------------------- prompts
_INDENT = " " * 24
_V1_LINES = [
    "You are a forensic image analyst detecting AI-generated or manipulated images.",
    "Analyze the image description and determine if it's real or fake.",
    "",
    "Key indicators to analyze:",
    "1. Lighting and shadows consistency",
    "2. Skin texture and natural imperfections",
    "3. Facial feature symmetry and proportions",
    "4. Reflections and highlights",
    "5. Eye and pupil details",
    "",
    "Provide your analysis in this exact format (ensure it's valid JSON):",
    "{",
    '    "verdict": "FAKE",',
    '    "justification": "Brief explanation of key factors"',
    "}",
    "",
    'Use "FAKE" or "REAL" for verdict (all caps).',
    "Keep justification concise but informative.",
]
# Reproduces the released triple-quoted string, including its leading newline
# and 24-space source indentation, which were sent to the API verbatim.
SYSTEM_PROMPT_V1 = "\n" + "\n".join(_INDENT + line for line in _V1_LINES) + "\n" + _INDENT

# Optional variant that additionally asks for the confidence level shown in
# Figure 1 of the paper. NOT used by the released code; off by default.
SYSTEM_PROMPT_V1_CONFIDENCE = "\n".join([
    "You are a forensic image analyst detecting AI-generated or manipulated images.",
    "Analyze the image description and determine if it's real or fake.",
    "",
    "Key indicators to analyze:",
    "1. Lighting and shadows consistency",
    "2. Skin texture and natural imperfections",
    "3. Facial feature symmetry and proportions",
    "4. Reflections and highlights",
    "5. Eye and pupil details",
    "",
    "Provide your analysis in this exact format (ensure it's valid JSON):",
    "{",
    '    "verdict": "FAKE",',
    '    "confidence": "HIGH",',
    '    "justification": "Brief explanation of key factors"',
    "}",
    "",
    'Use "FAKE" or "REAL" for verdict and "HIGH", "MEDIUM" or "LOW" for confidence (all caps).',
    "Keep justification concise but informative.",
])

USER_TEMPLATE_V1 = "Analyze this image description and determine if it's real or fake: {description}"

JUDGE_PROMPTS = {
    "truthlens_v1": (SYSTEM_PROMPT_V1, USER_TEMPLATE_V1),
    "truthlens_v1_confidence": (SYSTEM_PROMPT_V1_CONFIDENCE, USER_TEMPLATE_V1),
}


def build_messages(summary: str, prompt_name: str = "truthlens_v1"):
    if prompt_name not in JUDGE_PROMPTS:
        raise ValueError(f"Unknown judge prompt '{prompt_name}'. Choose from {list(JUDGE_PROMPTS)}.")
    system, user = JUDGE_PROMPTS[prompt_name]
    return [{"role": "system", "content": system},
            {"role": "user", "content": user.format(description=summary)}]


# --------------------------------------------------------------------------- parsing
@dataclass
class ParsedVerdict:
    verdict: Optional[str]          # "REAL" | "FAKE" | None
    justification: Optional[str]
    confidence: Optional[str]       # "HIGH" | "MEDIUM" | "LOW" | None
    parse_status: str               # "json" | "literal" | "regex" | "unparseable"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*|\s*```$")
_VERDICT_RE = re.compile(r"""["']?verdict["']?\s*[:=]\s*["']?\s*(REAL|FAKE)\b""", re.IGNORECASE)
_CONF_RE = re.compile(r"""["']?confidence["']?\s*[:=]\s*["']?\s*(HIGH|MEDIUM|LOW)\b""", re.IGNORECASE)
_JUST_RE = re.compile(r"""["']?justification["']?\s*[:=]\s*["'](.*?)["']\s*[,}]?\s*$""",
                      re.IGNORECASE | re.DOTALL)


def _from_mapping(obj: Any, status: str) -> Optional[ParsedVerdict]:
    if not isinstance(obj, dict):
        return None
    lowered = {str(k).strip().lower(): v for k, v in obj.items()}
    verdict = str(lowered.get("verdict", "")).strip().upper()
    if verdict not in VERDICTS:
        return None
    conf = lowered.get("confidence")
    conf = str(conf).strip().upper() if conf is not None else None
    if conf not in CONFIDENCES:
        conf = None
    just = lowered.get("justification")
    return ParsedVerdict(verdict, None if just is None else str(just), conf, status)


def parse_judge_response(text: Optional[str]) -> ParsedVerdict:
    """Parse the LLM reply into a verdict without guessing.

    Order: strict JSON -> Python literal (the released parser tried
    ``ast.literal_eval`` first; single-quoted dicts are accepted) -> JSON/literal
    object embedded in surrounding prose -> an explicit ``verdict: REAL|FAKE``
    field. Anything else is ``unparseable`` (verdict ``None``).
    """
    if text is None:
        return ParsedVerdict(None, None, None, "unparseable")
    cleaned = _FENCE_RE.sub("", text.strip()).strip()
    candidates = [cleaned]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if 0 <= start < end:
        candidates.append(cleaned[start:end + 1])
    for cand in candidates:
        try:
            got = _from_mapping(json.loads(cand), "json")
            if got:
                return got
        except (json.JSONDecodeError, TypeError):
            pass
        try:
            got = _from_mapping(ast.literal_eval(cand), "literal")
            if got:
                return got
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            pass
    m = _VERDICT_RE.search(cleaned)
    if m:
        conf = _CONF_RE.search(cleaned)
        just = _JUST_RE.search(cleaned)
        return ParsedVerdict(m.group(1).upper(), just.group(1) if just else cleaned,
                             conf.group(1).upper() if conf else None, "regex")
    return ParsedVerdict(None, cleaned, None, "unparseable")


# --------------------------------------------------------------------------- backends
class JudgeBackend:
    name = "base"
    model = "unknown"

    def complete(self, messages) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class OpenAIJudge(JudgeBackend):
    """OpenAI Chat Completions judge (``openai>=1.0``).

    Also works with OpenAI-compatible servers via ``OPENAI_BASE_URL``.
    ``temperature=None`` omits the parameter (API default), which is what the
    released code did; the paper does not report the decoding temperature.
    """

    name = "openai"

    def __init__(self, model: str = "gpt-4", temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None, seed: Optional[int] = None,
                 timeout: float = 120.0, max_retries: int = 6, client: Any = None):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        if client is None:
            if not os.environ.get("OPENAI_API_KEY"):
                raise EnvironmentError("OPENAI_API_KEY is not set. Export it in your shell; API keys are "
                                       "intentionally not accepted as command-line arguments.")
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover
                raise ImportError("Install the judge extra: pip install -e '.[judge]'") from exc
            # The client retries connection errors, 408/409/429 and 5xx with exponential backoff.
            client = OpenAI(timeout=timeout, max_retries=max_retries)
        self.client = client

    def complete(self, messages) -> str:
        kwargs: Dict[str, Any] = {"model": self.model, "messages": messages}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.seed is not None:
            kwargs["seed"] = self.seed
        response = self.client.chat.completions.create(**kwargs)
        self.last_system_fingerprint = getattr(response, "system_fingerprint", None)
        self.last_model = getattr(response, "model", self.model)
        return response.choices[0].message.content or ""


class MockJudge(JudgeBackend):
    """Deterministic keyword judge used ONLY for tests and smoke runs (not a scientific baseline)."""

    name = "mock"
    model = "mock-keyword-judge"
    ARTIFACT_CUES = ("unnatural", "inconsistent", "artificial", "distorted", "abnormal", "unusual",
                     "manipulat", "generated", "overly smooth")

    def __init__(self, **_ignored):
        pass

    def complete(self, messages) -> str:
        text = messages[-1]["content"].lower()
        hits = [c for c in self.ARTIFACT_CUES if c in text]
        verdict = "FAKE" if hits else "REAL"
        just = f"Artifact cues: {', '.join(hits)}" if hits else "No artifact cues found."
        return json.dumps({"verdict": verdict, "justification": just})


def make_judge(backend: str, **kwargs) -> JudgeBackend:
    if backend == "openai":
        allowed = {k: kwargs[k] for k in ("model", "temperature", "max_tokens", "seed", "timeout", "max_retries")
                   if k in kwargs}
        return OpenAIJudge(**allowed)
    if backend == "mock":
        return MockJudge()
    raise ValueError(f"Unknown judge backend '{backend}'. Choose from ['openai', 'mock'].")
