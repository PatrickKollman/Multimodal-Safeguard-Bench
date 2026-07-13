"""Target VLM adapters: LLaVA-1.6, Qwen2-VL-7B. Factory: build_target()."""
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


class Qwen2VLTarget:
    """
    Qwen2-VL-7B-Instruct target VLM.

    Native in transformers >= 4.45.0 (no trust_remote_code).
    Loaded 4-bit NF4 by default (~4 GB VRAM); set load_in_4bit: false for bfloat16 (~14 GB).
    PIL images are passed directly to Qwen2VLProcessor — no qwen_vl_utils required.
    """

    def __init__(self, cfg: dict):
        self.model_id = cfg["model_id"]
        self.revision = cfg.get("revision")
        self.dtype = torch.bfloat16 if cfg.get("dtype", "bfloat16") == "bfloat16" else torch.float16
        self.device = cfg.get("device", "auto")
        self.load_in_4bit = cfg.get("load_in_4bit", True)
        self.max_new_tokens = cfg.get("max_new_tokens", 512)
        self.do_sample = cfg.get("do_sample", False)
        self._model = None
        self._processor = None

    def load(self) -> None:
        from transformers import (
            Qwen2VLForConditionalGeneration,
            AutoProcessor,
            BitsAndBytesConfig,
        )
        from ._download import load_with_retry

        rev_kwargs = {"revision": self.revision} if self.revision else {}
        quant_cfg = None
        if self.load_in_4bit:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self._processor = load_with_retry(
            AutoProcessor.from_pretrained, self.model_id, **rev_kwargs
        )
        self._model = load_with_retry(
            Qwen2VLForConditionalGeneration.from_pretrained,
            self.model_id,
            **rev_kwargs,
            torch_dtype=self.dtype,
            quantization_config=quant_cfg,
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
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        else:
            messages = [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if image is not None:
            inputs = self._processor(
                text=[text], images=[image], return_tensors="pt", padding=True
            )
        else:
            inputs = self._processor(text=[text], return_tensors="pt", padding=True)

        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
            )

        # Strip input tokens from the generated output
        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self._processor.decode(generated, skip_special_tokens=True)


def build_target(cfg: dict):
    """Dispatch target VLM class from config.

    Dispatches on cfg['type'] if present; falls back to model_id substring matching.
    Supported: 'llava' (default), 'qwen2vl'.
    """
    target_type = cfg.get("type", "").lower()
    model_id    = cfg.get("model_id", "").lower()

    if target_type == "qwen2vl" or "qwen2-vl" in model_id or "qwen2vl" in model_id:
        return Qwen2VLTarget(cfg)
    # 'llava' is the default / fallback
    return LLaVATarget(cfg)
