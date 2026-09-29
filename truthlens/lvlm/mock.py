"""Deterministic mock LVLM for tests and smoke runs. NOT a scientific baseline.

It answers from global image brightness only: images with mean intensity above
``threshold`` receive an "artifact" answer, others a "natural" answer. The
smoke-test data generator makes fake images bright and real images dark, so a
correct pipeline must reach 100% accuracy end to end.
"""
from __future__ import annotations

from .base import LVLMBackend


class MockLVLM(LVLMBackend):
    name = "mock"

    def __init__(self, threshold: float = 127.5, **kwargs):
        super().__init__(**kwargs)
        self.threshold = float(threshold)

    def _is_bright(self, image) -> bool:
        gray = image.convert("L")
        hist = gray.histogram()
        total = sum(hist)
        mean = sum(i * c for i, c in enumerate(hist)) / max(total, 1)
        return mean > self.threshold

    def generate(self, image, prompt: str) -> str:
        bright = self._is_bright(image)
        if prompt.lower().startswith("is this image fake"):
            return "Yes." if bright else "No."
        if bright:
            return "The region looks unnatural and inconsistent with the rest of the photograph."
        return "The region looks natural and consistent with a real photograph."
