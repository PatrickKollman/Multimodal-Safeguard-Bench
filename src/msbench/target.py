"""LLaVA-1.6 Mistral-7B target VLM adapter."""
from __future__ import annotations
import torch
from PIL import Image


class LLaVATarget:
    def __init__(self, cfg: dict):
        self.model_id = cfg["model_id"]
        self.revision = cfg.get("revision")
        self.dtype = torch.bfloat16 if cfg.get("dtype", "bfloat16") == "bfloat16" else torch.float16
        self.device = cfg.get("device", "auto")
        self.max_new_tokens = cfg.get("max_new_tokens", 512)
        self.do_sample = cfg.get("do_sample", False)
        self._model = None
        self._processor = None

    def load(self) -> None:
        from transformers import AutoProcessor, LlavaNextForConditionalGeneration
        from ._download import load_with_retry
        rev_kwargs = {"revision": self.revision} if self.revision else {}
        self._processor = load_with_retry(
            AutoProcessor.from_pretrained, self.model_id, **rev_kwargs
        )
        self._model = load_with_retry(
            LlavaNextForConditionalGeneration.from_pretrained,
            self.model_id,
            **rev_kwargs,
            torch_dtype=self.dtype,
            device_map=self.device,
        )
        self._model.eval()

    def unload(self) -> None:
        del self._model, self._processor
        self._model = self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(self, prompt: str, image: Image.Image | None = None) -> str:
        assert self._model is not None, "Call load() first"
        if image is not None:
            conversation = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
            ]
        else:
            conversation = [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ]

        text = self._processor.apply_chat_template(conversation, add_generation_prompt=True)

        if image is not None:
            inputs = self._processor(images=image, text=text, return_tensors="pt")
        else:
            inputs = self._processor(text=text, return_tensors="pt")

        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
            )

        out = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self._processor.decode(out, skip_special_tokens=True)
