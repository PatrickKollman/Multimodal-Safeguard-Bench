"""
Safety guards — three implementations, staged sequentially to fit 24 GB VRAM.

LlamaGuard4       : multimodal (image + text) → safe/unsafe  [Guard #1]
LlamaGuard3Vision : multimodal, text-in-image focused → safe/unsafe  [Guard #2]
ShieldGemma2      : image-content only → safe/unsafe          [Guard #3]

NOTE: ShieldGemma2 cannot classify text-modality items (no image to evaluate).
It returns "safe" / "n/a" for those, which is by design — the point is to measure
how little coverage an image-content-only guard provides against text-rendered attacks.

HF access requirements:
  - LlamaGuard4:        meta-llama/Llama-Guard-4-12B           (Llama 4 Community License)
  - LlamaGuard3Vision:  meta-llama/Llama-Guard-3-11B-Vision    (Llama 3 Community License)
  - ShieldGemma2:       google/shieldgemma-2-4b-it             (Google Gemma License)
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import torch
from PIL import Image


class BaseGuard(ABC):
    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def unload(self) -> None: ...

    @abstractmethod
    def classify(self, prompt: str, image: Image.Image | None) -> tuple[str, str]:
        """Returns (label, category).  label ∈ {"safe", "unsafe"}."""
        ...

    @property
    @abstractmethod
    def name(self) -> str: ...


class LlamaGuard4(BaseGuard):
    """Llama Guard 4 12B, 4-bit quantized. Input: (image?, text) → safe/unsafe + S-category."""

    def __init__(self, cfg: dict):
        self.model_id = cfg["model_id"]
        self.revision = cfg.get("revision")
        self.load_in_4bit = cfg.get("load_in_4bit", True)
        self.dtype = torch.bfloat16 if cfg.get("dtype", "bfloat16") == "bfloat16" else torch.float16
        self.device = cfg.get("device", "auto")
        self._model = None
        self._processor = None

    @property
    def name(self) -> str:
        return "llama_guard_4"

    def load(self) -> None:
        from transformers import AutoProcessor, Llama4ForConditionalGeneration, BitsAndBytesConfig
        from ._download import load_with_retry
        quant_cfg = None
        if self.load_in_4bit:
            quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=self.dtype)
        self._processor = load_with_retry(
            AutoProcessor.from_pretrained, self.model_id, revision=self.revision
        )
        self._model = load_with_retry(
            Llama4ForConditionalGeneration.from_pretrained,
            self.model_id,
            revision=self.revision,
            torch_dtype=self.dtype,
            device_map=self.device,
            quantization_config=quant_cfg,
        )
        self._model.eval()
        # Llama Guard 4's config.json has attention_chunk_size=null (dropped when pruning
        # from Scout MoE to dense). DynamicCache reads this to size the per-layer sliding
        # window; None causes a TypeError. Restore the Llama4TextConfig documented default.
        text_cfg = self._model.config.get_text_config(decoder=True)
        if getattr(text_cfg, "attention_chunk_size", None) is None:
            text_cfg.attention_chunk_size = 8192

    def unload(self) -> None:
        del self._model, self._processor
        self._model = self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def classify(self, prompt: str, image: Image.Image | None) -> tuple[str, str]:
        assert self._model is not None, "Call load() first"
        content = (
            [{"type": "image"}, {"type": "text", "text": prompt}]
            if image is not None
            else [{"type": "text", "text": prompt}]
        )
        messages = [{"role": "user", "content": content}]
        # Two-step: get formatted text, then run the full processor with pixel values.
        # apply_chat_template alone only produces token ids; images need a separate pass.
        text = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        if image is not None:
            inputs = self._processor(text=text, images=[image], return_tensors="pt")
        else:
            inputs = self._processor(text=text, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs, max_new_tokens=20, do_sample=False, cache_implementation="dynamic"
            )

        response = self._processor.batch_decode(
            output_ids[:, inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )[0].strip()

        if response.startswith("unsafe"):
            parts = response.split("\n", 1)
            category = parts[1].strip() if len(parts) > 1 else ""
            return "unsafe", category
        return "safe", ""


class LlamaGuard3Vision(BaseGuard):
    """
    LlamaGuard 3 11B Vision — Llama 3.2 Vision fine-tuned as a multimodal safety classifier.
    Unlike LG4 (Llama 4 Scout derivative, text-intent focused), LG3V was explicitly trained
    on text-in-image scenarios, making it the strongest expected baseline for this benchmark.
    Input: (image?, text) → safe/unsafe + S-category.
    """

    def __init__(self, cfg: dict):
        self.model_id = cfg["model_id"]
        self.revision = cfg.get("revision")
        self.load_in_4bit = cfg.get("load_in_4bit", True)
        self.dtype = torch.bfloat16 if cfg.get("dtype", "bfloat16") == "bfloat16" else torch.float16
        self.device = cfg.get("device", "auto")
        self._model = None
        self._processor = None

    @property
    def name(self) -> str:
        return "llama_guard_3_vision"

    def load(self) -> None:
        from transformers import AutoProcessor, MllamaForConditionalGeneration, BitsAndBytesConfig
        from ._download import load_with_retry
        quant_cfg = None
        if self.load_in_4bit:
            quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=self.dtype)
        self._processor = load_with_retry(
            AutoProcessor.from_pretrained, self.model_id, revision=self.revision
        )
        self._model = load_with_retry(
            MllamaForConditionalGeneration.from_pretrained,
            self.model_id,
            revision=self.revision,
            torch_dtype=self.dtype,
            device_map=self.device,
            quantization_config=quant_cfg,
        )
        self._model.eval()

    def unload(self) -> None:
        del self._model, self._processor
        self._model = self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def classify(self, prompt: str, image: Image.Image | None) -> tuple[str, str]:
        assert self._model is not None, "Call load() first"
        content = (
            [{"type": "image"}, {"type": "text", "text": prompt}]
            if image is not None
            else [{"type": "text", "text": prompt}]
        )
        messages = [{"role": "user", "content": content}]
        text = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        if image is not None:
            inputs = self._processor(text=text, images=[image], return_tensors="pt")
        else:
            inputs = self._processor(text=text, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=20, do_sample=False)

        response = self._processor.batch_decode(
            output_ids[:, inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )[0].strip()

        if response.startswith("unsafe"):
            parts = response.split("\n", 1)
            category = parts[1].strip() if len(parts) > 1 else ""
            return "unsafe", category
        return "safe", ""


class ShieldGemma2(BaseGuard):
    """
    ShieldGemma 2 4B: image-content classifier against three harm policies.
    Policies checked (in order): dangerous_content, sexually_explicit, violence_gore.
    Text-modality items (image=None) are returned as "safe" / "n/a" — not applicable.
    """

    POLICY_NAMES = ["dangerous_content", "sexually_explicit", "violence_gore"]

    def __init__(self, cfg: dict):
        self.model_id = cfg["model_id"]
        self.revision = cfg.get("revision")
        self.dtype = torch.bfloat16 if cfg.get("dtype", "bfloat16") == "bfloat16" else torch.float16
        self.device = cfg.get("device", "auto")
        self.threshold = cfg.get("threshold", 0.5)
        self._model = None
        self._processor = None
        self.last_scores: dict | None = None  # raw per-policy probs, set after each classify()

    @property
    def name(self) -> str:
        return "shield_gemma_2"

    def load(self) -> None:
        from transformers import AutoProcessor, ShieldGemma2ForImageClassification
        from ._download import load_with_retry
        self._processor = load_with_retry(
            AutoProcessor.from_pretrained, self.model_id, revision=self.revision
        )
        self._model = load_with_retry(
            ShieldGemma2ForImageClassification.from_pretrained,
            self.model_id,
            revision=self.revision,
            torch_dtype=self.dtype,
            device_map=self.device,
        )
        self._model.eval()

    def unload(self) -> None:
        del self._model, self._processor
        self._model = self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def classify(self, prompt: str, image: Image.Image | None) -> tuple[str, str]:
        assert self._model is not None, "Call load() first"
        if image is None:
            self.last_scores = None
            return "safe", "n/a"

        inputs = self._processor(images=[image], return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            scores = self._model(**inputs)

        # probabilities.shape = [num_policies, 2]: P(violated), P(not_violated) per policy
        self.last_scores = {
            pol_name: round(pol_probs[0].item(), 6)
            for pol_name, pol_probs in zip(self.POLICY_NAMES, scores.probabilities)
        }
        for pol_name, prob in self.last_scores.items():
            if prob > self.threshold:
                return "unsafe", pol_name
        return "safe", ""


def build_guard(cfg: dict) -> BaseGuard:
    name = cfg["name"]
    if name == "llama_guard_4":
        return LlamaGuard4(cfg)
    if name == "llama_guard_3_vision":
        return LlamaGuard3Vision(cfg)
    if name == "shield_gemma_2":
        return ShieldGemma2(cfg)
    raise ValueError(f"Unknown guard: {name!r}")
