from __future__ import annotations

import logging

import streamlit as st
import torch

logger = logging.getLogger(__name__)


class ModelWrapper:
    """Uniform interface over nnterp StandardizedTransformer or raw nnsight LanguageModel."""

    def __init__(self, model, is_standardized: bool, model_name: str):
        self._model = model
        self._is_standardized = is_standardized
        self._model_name = model_name
        self._config = model.config if hasattr(model, "config") else model.model.config

    @property
    def raw(self):
        return self._model

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def is_standardized(self) -> bool:
        return self._is_standardized

    @property
    def config(self):
        return self._config

    @property
    def num_layers(self) -> int:
        return self._config.num_hidden_layers

    @property
    def hidden_size(self) -> int:
        return self._config.hidden_size

    @property
    def num_heads(self) -> int:
        return self._config.num_attention_heads

    @property
    def vocab_size(self) -> int:
        return self._config.vocab_size

    @property
    def device(self) -> torch.device:
        if self._is_standardized:
            try:
                return next(self._model.parameters()).device
            except StopIteration:
                return torch.device("cpu")
        else:
            try:
                return next(self._model.parameters()).device
            except StopIteration:
                return torch.device("cpu")

    @property
    def tokenizer(self):
        if self._is_standardized:
            return self._model.tokenizer
        return self._model.tokenizer

    def trace(self, prompt, **kwargs):
        return self._model.trace(prompt, **kwargs)

    def _get_layers_module(self):
        if self._is_standardized:
            return self._model.layers
        model_type = getattr(self._config, "model_type", "")
        inner = self._model
        if model_type in ("gpt2", "gpt_neo", "gpt_neox"):
            return inner.transformer.h
        elif model_type in ("llama", "mistral", "gemma", "gemma2", "phi", "qwen2", "qwen3"):
            return inner.model.layers
        elif model_type == "opt":
            return inner.model.decoder.layers
        else:
            for attr in ("model.layers", "transformer.h", "transformer.layers"):
                parts = attr.split(".")
                obj = inner
                try:
                    for p in parts:
                        obj = getattr(obj, p)
                    return obj
                except AttributeError:
                    continue
            raise ValueError(
                f"Cannot detect layer structure for model_type='{model_type}'. "
                "Try using a supported architecture."
            )

    def get_layer_module(self, layer_idx: int):
        return self._get_layers_module()[layer_idx]

    def get_layer_output(self, layer_idx: int):
        return self.get_layer_module(layer_idx).output[0]

    def get_attention_output(self, layer_idx: int):
        layer = self.get_layer_module(layer_idx)
        return layer.self_attn.output[0]

    def get_mlp_output(self, layer_idx: int):
        layer = self.get_layer_module(layer_idx)
        return layer.mlp.output

    def set_layer_output(self, layer_idx: int, value):
        self.get_layer_module(layer_idx).output[0][:] = value

    def get_ln_final(self):
        if self._is_standardized:
            return self._model.ln_final
        model_type = getattr(self._config, "model_type", "")
        if model_type in ("gpt2", "gpt_neo"):
            return self._model.transformer.ln_f
        elif model_type in ("llama", "mistral", "gemma", "gemma2", "phi", "qwen2", "qwen3"):
            return self._model.model.norm
        elif model_type == "opt":
            return self._model.model.decoder.final_layer_norm
        elif model_type == "gpt_neox":
            return self._model.gpt_neox.final_layer_norm
        raise ValueError(f"Cannot detect ln_final for model_type='{model_type}'")

    def get_lm_head(self):
        if self._is_standardized:
            return self._model.lm_head
        if hasattr(self._model, "lm_head"):
            return self._model.lm_head
        if hasattr(self._model, "embed_out"):
            return self._model.embed_out
        raise ValueError("Cannot detect lm_head module")

    def project_on_vocab(self, hidden_state):
        normalized = self.get_ln_final()(hidden_state)
        return self.get_lm_head()(normalized)

    def get_logits(self):
        return self._model.output[0] if not self._is_standardized else self._model.logits

    def total_params(self) -> int:
        return sum(p.numel() for p in self._model.parameters())


def _build_quant_kwargs(quant: str) -> dict:
    if quant == "8bit":
        return {"load_in_8bit": True}
    elif quant == "4bit":
        return {"load_in_4bit": True}
    return {}


@st.cache_resource
def load_model(model_name: str, quant: str = "none") -> ModelWrapper:
    quant_kwargs = _build_quant_kwargs(quant)
    device_map = "auto" if torch.cuda.is_available() else "cpu"

    if quant != "none" and not torch.cuda.is_available():
        logger.warning("Quantization requested but no CUDA available. Loading in FP32.")
        quant_kwargs = {}
        quant = "none"

    try:
        from nnterp import StandardizedTransformer

        model = StandardizedTransformer(
            model_name,
            device_map=device_map,
            dispatch=True,
            **quant_kwargs,
        )
        logger.info(f"Loaded '{model_name}' via nnterp (standardized)")
        return ModelWrapper(model, is_standardized=True, model_name=model_name)

    except Exception as e:
        logger.warning(f"nnterp failed for '{model_name}': {e}. Falling back to nnsight.")

    from nnsight import LanguageModel

    model = LanguageModel(
        model_name,
        device_map=device_map,
        dispatch=True,
        **quant_kwargs,
    )
    logger.info(f"Loaded '{model_name}' via nnsight (raw)")
    return ModelWrapper(model, is_standardized=False, model_name=model_name)
