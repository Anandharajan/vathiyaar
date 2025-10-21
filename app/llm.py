from __future__ import annotations

from threading import Thread
from typing import Generator, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from app.config import CFG
from app.prompts import SYSTEM_PROMPT

_TOKENIZER: Optional[AutoTokenizer] = None
_MODEL: Optional[AutoModelForCausalLM] = None


def _load_model() -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    global _TOKENIZER, _MODEL
    if _MODEL is None or _TOKENIZER is None:
        llm_id = CFG["llm_id"]
        kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        }
        try:  # try 4-bit quantization when bitsandbytes is present
            import bitsandbytes as _  # noqa: F401

            kwargs.update(load_in_4bit=True)
        except Exception:
            pass

        _TOKENIZER = AutoTokenizer.from_pretrained(llm_id, use_fast=True)
        _MODEL = AutoModelForCausalLM.from_pretrained(llm_id, **kwargs)
    return _TOKENIZER, _MODEL


def format_prompt(user_text: str, context: str, citations: str, history: str) -> str:
    parts = [SYSTEM_PROMPT, history, f"User: {user_text}"]
    if context.strip():
        parts.append("Context:")
        parts.append(context)
    if citations.strip():
        parts.append(f"Use citations: {citations}")
    return "\n\n".join(part for part in parts if part)


def generate(prompt: str) -> Generator[str, None, None]:
    tokenizer, model = _load_model()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    cfg = CFG["gen"]
    if cfg["stream"]:
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(
            **inputs,
            max_new_tokens=cfg["max_new_tokens"],
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            top_k=cfg["top_k"],
            do_sample=True,
            streamer=streamer,
        )
        thread = Thread(target=model.generate, kwargs=kwargs)
        thread.start()
        for chunk in streamer:
            yield chunk
    else:
        outputs = model.generate(
            **inputs,
            max_new_tokens=cfg["max_new_tokens"],
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            top_k=cfg["top_k"],
            do_sample=True,
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        yield text
