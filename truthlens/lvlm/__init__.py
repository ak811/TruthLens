"""LVLM backend registry. Heavy dependencies are imported only when a backend is built."""
from __future__ import annotations

from typing import Any, Dict

from .base import GenerationConfig, LVLMBackend

BACKENDS = ("chatunivi", "llava15", "blip2", "cogvlm", "mock")


def make_lvlm(backend: str, **kwargs: Any) -> LVLMBackend:
    """Instantiate an LVLM backend by name. ``kwargs`` may include ``generation`` as a dict."""
    kwargs = dict(kwargs)
    gen = kwargs.pop("generation", None)
    kwargs["generation"] = gen if isinstance(gen, GenerationConfig) else GenerationConfig.from_dict(gen)
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if backend == "mock":
        from .mock import MockLVLM
        return MockLVLM(**kwargs)
    if backend == "chatunivi":
        from .chatunivi import ChatUniViBackend
        return ChatUniViBackend(**kwargs)
    if backend == "llava15":
        from .hf_models import LlavaHFBackend
        return LlavaHFBackend(**kwargs)
    if backend == "blip2":
        from .hf_models import Blip2HFBackend
        return Blip2HFBackend(**kwargs)
    if backend == "cogvlm":
        from .hf_models import CogVLMHFBackend
        return CogVLMHFBackend(**kwargs)
    raise ValueError(f"Unknown LVLM backend '{backend}'. Choose from {BACKENDS}.")


__all__ = ["BACKENDS", "GenerationConfig", "LVLMBackend", "make_lvlm"]
