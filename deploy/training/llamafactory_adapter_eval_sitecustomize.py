"""Enable BF16 autocast for merged FP8 LoRA evaluation.

LLaMA Factory dequantizes FP8 modules targeted by a merged adapter to FP32.
Its Hugging Face API generation path otherwise invokes those modules with
BF16 activations outside autocast, which makes torch.linear reject the mixed
dtypes. This file is mounted as ``sitecustomize.py`` only in the frozen
adapter-evaluation container.
"""

from __future__ import annotations

from functools import wraps

import torch
from llamafactory.chat.hf_engine import HuggingfaceEngine
from llamafactory.model import adapter as adapter_module


_original_chat = HuggingfaceEngine._chat
_original_setup_lora = adapter_module._setup_lora_tuning


@wraps(_original_setup_lora)
def _load_fp8_adapter_without_merge(
    config,
    model,
    model_args,
    finetuning_args,
    is_trainable,
    cast_trainable_params_to_fp32,
):
    marker = getattr(model, "quantization_method", None)
    model.quantization_method = marker or "fp8"
    try:
        return _original_setup_lora(
            config,
            model,
            model_args,
            finetuning_args,
            is_trainable,
            cast_trainable_params_to_fp32,
        )
    finally:
        if marker is None:
            delattr(model, "quantization_method")
        else:
            model.quantization_method = marker


@wraps(_original_chat)
def _bf16_autocast_chat(*args, **kwargs):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return _original_chat(*args, **kwargs)


_bf16_autocast_chat.__adapter_eval_autocast__ = True
if not getattr(HuggingfaceEngine._chat, "__adapter_eval_autocast__", False):
    HuggingfaceEngine._chat = staticmethod(_bf16_autocast_chat)

_load_fp8_adapter_without_merge.__adapter_eval_no_merge__ = True
if not getattr(
    adapter_module._setup_lora_tuning,
    "__adapter_eval_no_merge__",
    False,
):
    adapter_module._setup_lora_tuning = _load_fp8_adapter_without_merge
