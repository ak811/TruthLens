"""Chat-UniVi backend (primary LVLM of the paper; Table 2 best row).

Ported from the released ``inference_image_chatunivi.py`` with these changes:
the query is a parameter instead of a hard-coded string, images are converted
to RGB, ``CUDA_VISIBLE_DEVICES`` is no longer hard-coded to "1" (set it in the
shell), the ineffective runtime ``PYTHONPATH`` assignment was removed, and
errors are raised to the caller (they were previously written into the output
JSON as if they were model answers).

Requires Chat-UniVi installed in the same environment:
    git clone https://github.com/PKU-YuanGroup/Chat-UniVi && pip install -e Chat-UniVi
"""
from __future__ import annotations

from .base import LVLMBackend


class ChatUniViBackend(LVLMBackend):
    name = "chatunivi"

    def __init__(self, model_path: str = "Chat-UniVi/Chat-UniVi", conv_mode: str = "simple", **kwargs):
        super().__init__(model_path=model_path, **kwargs)
        self.conv_mode = conv_mode
        try:
            import torch  # noqa: F401
            from ChatUniVi.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                                             DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IMAGE_TOKEN,
                                             IMAGE_TOKEN_INDEX)
            from ChatUniVi.conversation import SeparatorStyle, conv_templates
            from ChatUniVi.mm_utils import KeywordsStoppingCriteria, tokenizer_image_token
            from ChatUniVi.model.builder import load_pretrained_model
            from ChatUniVi.utils import disable_torch_init
        except ImportError as exc:
            raise ImportError("Chat-UniVi is not installed. See docs/REPRODUCIBILITY.md "
                              "(Environment B: Chat-UniVi).") from exc
        self._c = dict(IM_END=DEFAULT_IM_END_TOKEN, IM_START=DEFAULT_IM_START_TOKEN,
                       PATCH=DEFAULT_IMAGE_PATCH_TOKEN, IMAGE=DEFAULT_IMAGE_TOKEN, INDEX=IMAGE_TOKEN_INDEX)
        self._SeparatorStyle, self._conv_templates = SeparatorStyle, conv_templates
        self._Stopping, self._tokenizer_image_token = KeywordsStoppingCriteria, tokenizer_image_token

        disable_torch_init()
        # Same initialisation as the released script / Chat-UniVi README.
        tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, "ChatUniVi")
        if getattr(model.config, "mm_use_im_patch_token", True):
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if getattr(model.config, "mm_use_im_start_end", False):
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))
        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        self.tokenizer, self.model = tokenizer, model
        self.image_processor = vision_tower.image_processor

    def generate(self, image, prompt: str) -> str:
        import torch

        c = self._c
        if self.model.config.mm_use_im_start_end:
            qs = c["IM_START"] + c["IMAGE"] + c["IM_END"] + "\n" + prompt
        else:
            qs = c["IMAGE"] + "\n" + prompt
        conv = self._conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        input_ids = self._tokenizer_image_token(full_prompt, self.tokenizer, c["INDEX"],
                                                return_tensors="pt").unsqueeze(0).cuda()
        image_tensor = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        stop_str = conv.sep if conv.sep_style != self._SeparatorStyle.TWO else conv.sep2
        stopping = self._Stopping([stop_str], self.tokenizer, input_ids)
        g = self.generation
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=g.do_sample,
                temperature=g.temperature if g.do_sample else None,
                top_p=g.top_p,
                num_beams=g.num_beams,
                max_new_tokens=g.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping],
            )
        n_in = input_ids.shape[1]
        out = self.tokenizer.batch_decode(output_ids[:, n_in:], skip_special_tokens=True)[0].strip()
        if out.endswith(stop_str):
            out = out[: -len(stop_str)]
        return out.strip()
