"""LVLM backend interface: ``f_MM(I, p) -> a`` (paper Section 2.2)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class GenerationConfig:
    """Decoding parameters. Defaults reproduce the released Chat-UniVi script
    (sampling, temperature 0.2, 1 beam, 1024 new tokens). The paper does not
    report decoding parameters."""

    do_sample: bool = True
    temperature: float = 0.2
    top_p: Optional[float] = None
    num_beams: int = 1
    max_new_tokens: int = 1024

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "GenerationConfig":
        d = dict(d or {})
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown generation options: {sorted(unknown)}")
        return cls(**d)

    def hf_kwargs(self) -> Dict[str, Any]:
        kw: Dict[str, Any] = {"do_sample": self.do_sample, "num_beams": self.num_beams,
                              "max_new_tokens": self.max_new_tokens}
        if self.do_sample:
            kw["temperature"] = self.temperature
            if self.top_p is not None:
                kw["top_p"] = self.top_p
        return kw

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LVLMBackend(ABC):
    """Base class. Subclasses load their model lazily in ``__init__``."""

    name = "base"

    def __init__(self, model_path: Optional[str] = None, device: str = "cuda",
                 generation: Optional[GenerationConfig] = None, **options: Any):
        self.model_path = model_path
        self.device = device
        self.generation = generation or GenerationConfig()
        self.options = options

    @abstractmethod
    def generate(self, image, prompt: str) -> str:
        """Answer ``prompt`` about the RGB PIL ``image``. Seeding is handled by the caller."""

    def describe(self) -> Dict[str, Any]:
        return {"backend": self.name, "model_path": self.model_path, "device": self.device,
                "generation": self.generation.to_dict(), "options": self.options}
