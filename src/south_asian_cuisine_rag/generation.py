from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from .config import GENERATION_MODEL_ID, Settings
from .errors import ConfigurationError


class TextGenerator(Protocol):
    model_id: str

    def generate(self, messages: Sequence[dict[str, str]]) -> str: ...


class QwenGenerator:
    """Lazy deterministic generator locked to Qwen2.5-0.5B-Instruct."""

    def __init__(self, settings: Settings) -> None:
        if settings.generation_model != GENERATION_MODEL_ID:
            raise ConfigurationError(f"Only {GENERATION_MODEL_ID} is permitted")
        self.settings = settings
        self.model_id = settings.generation_model
        self._tokenizer: Any = None
        self._model: Any = None

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None and self._tokenizer is not None:
            return self._tokenizer, self._model
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("torch and transformers are required for generation") from exc

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            revision=self.settings.generation_revision,
            trust_remote_code=False,
            local_files_only=self.settings.offline_mode,
        )
        model_kwargs: dict[str, Any] = {
            "revision": self.settings.generation_revision,
            "trust_remote_code": False,
            "low_cpu_mem_usage": True,
            "local_files_only": self.settings.offline_mode,
        }
        if self.settings.device == "auto":
            model_kwargs.update(torch_dtype="auto", device_map="auto")
        elif self.settings.device == "cuda":
            model_kwargs.update(torch_dtype=torch.float16, device_map={"": "cuda"})
        else:
            model_kwargs.update(torch_dtype=torch.float32, device_map={"": "cpu"})

        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
        self._model.eval()
        return self._tokenizer, self._model

    def generate(self, messages: Sequence[dict[str, str]]) -> str:
        import torch

        tokenizer, model = self._load()
        rendered = tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer(
            rendered,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=self.settings.max_input_tokens,
        ).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **model_inputs,
                max_new_tokens=self.settings.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        response_tokens = generated[0, model_inputs["input_ids"].shape[1] :]
        return str(tokenizer.decode(response_tokens, skip_special_tokens=True)).strip()
