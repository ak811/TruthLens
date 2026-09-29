"""Hugging Face ``transformers`` backends for the other LVLMs of Table 2.

The paper does not state which checkpoints/sizes of BLIP-2, LLaVA-1.5 and
CogVLM were used, nor their prompt templates. The defaults below are common
public checkpoints and are ASSUMPTIONS; set ``probe.model_path`` to the
checkpoint you actually used. These backends were not executed in the
refactoring environment (no GPU / weights available); they follow the
documented ``transformers`` APIs.
"""
from __future__ import annotations

from .base import LVLMBackend


def _torch_dtype(name: str):
    import torch

    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


class LlavaHFBackend(LVLMBackend):
    """LLaVA-1.5 via ``LlavaForConditionalGeneration`` (default: llava-hf/llava-1.5-7b-hf)."""

    name = "llava15"

    def __init__(self, model_path: str = "llava-hf/llava-1.5-7b-hf", dtype: str = "float16", **kwargs):
        super().__init__(model_path=model_path, **kwargs)
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self.dtype = _torch_dtype(dtype)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=self.dtype).to(self.device).eval()

    def generate(self, image, prompt: str) -> str:
        import torch

        text = f"USER: <image>\n{prompt} ASSISTANT:"
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device, self.dtype)
        with torch.inference_mode():
            out = self.model.generate(**inputs, **self.generation.hf_kwargs())
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(new_tokens, skip_special_tokens=True).strip()


class Blip2HFBackend(LVLMBackend):
    """BLIP-2 via ``Blip2ForConditionalGeneration`` (default: Salesforce/blip2-opt-2.7b)."""

    name = "blip2"

    def __init__(self, model_path: str = "Salesforce/blip2-opt-2.7b", dtype: str = "float16", **kwargs):
        super().__init__(model_path=model_path, **kwargs)
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        self.dtype = _torch_dtype(dtype)
        self.processor = Blip2Processor.from_pretrained(model_path)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=self.dtype).to(self.device).eval()

    def generate(self, image, prompt: str) -> str:
        import torch

        text = f"Question: {prompt} Answer:"
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device, self.dtype)
        with torch.inference_mode():
            out = self.model.generate(**inputs, **self.generation.hf_kwargs())
        decoded = self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()
        # Depending on the transformers version the output may or may not echo the prompt.
        if decoded.startswith(text):
            decoded = decoded[len(text):]
        return decoded.strip()


class CogVLMHFBackend(LVLMBackend):
    """CogVLM-chat via ``trust_remote_code`` (default: THUDM/cogvlm-chat-hf).

    EXPERIMENTAL: the remote code pins older ``transformers`` versions; follow
    the model card for a compatible environment.
    """

    name = "cogvlm"

    def __init__(self, model_path: str = "THUDM/cogvlm-chat-hf", tokenizer_path: str = "lmsys/vicuna-7b-v1.5",
                 dtype: str = "bfloat16", **kwargs):
        super().__init__(model_path=model_path, **kwargs)
        from transformers import AutoModelForCausalLM, LlamaTokenizer

        self.dtype = _torch_dtype(dtype)
        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=self.dtype, low_cpu_mem_usage=True, trust_remote_code=True
        ).to(self.device).eval()

    def generate(self, image, prompt: str) -> str:
        import torch

        built = self.model.build_conversation_input_ids(self.tokenizer, query=prompt, history=[], images=[image])
        inputs = {
            "input_ids": built["input_ids"].unsqueeze(0).to(self.device),
            "token_type_ids": built["token_type_ids"].unsqueeze(0).to(self.device),
            "attention_mask": built["attention_mask"].unsqueeze(0).to(self.device),
            "images": [[built["images"][0].to(self.device).to(self.dtype)]],
        }
        with torch.inference_mode():
            out = self.model.generate(**inputs, **self.generation.hf_kwargs())
        out = out[:, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
