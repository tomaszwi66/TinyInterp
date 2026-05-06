from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from core.loader import ModelWrapper


def run_activation_patching(
    wrapper: ModelWrapper,
    clean_prompt: str,
    corrupted_prompt: str,
    target_token: str | None = None,
    layers: list[int] | None = None,
    patch_mode: str = "per_layer",
    progress_callback: Callable[[float, str], bool] | None = None,
) -> dict:
    if layers is None:
        layers = list(range(wrapper.num_layers))

    tokenizer = wrapper.tokenizer
    clean_ids = tokenizer.encode(clean_prompt)
    corrupted_ids = tokenizer.encode(corrupted_prompt)
    tokens = tokenizer.convert_ids_to_tokens(clean_ids)
    num_positions = len(clean_ids)

    # Step 1: Clean run - save all layer outputs + logits
    clean_hidden = {}
    with wrapper.trace(clean_prompt):
        for layer_idx in layers:
            clean_hidden[layer_idx] = wrapper.get_layer_output(layer_idx).save()
        clean_logits = wrapper.get_logits().save()

    clean_logits_val = clean_logits.detach().float()

    if target_token is not None:
        target_id = tokenizer.encode(target_token, add_special_tokens=False)
        if len(target_id) == 0:
            raise ValueError(f"Target token '{target_token}' not found in vocabulary")
        target_id = target_id[0]
    else:
        target_id = clean_logits_val[0, -1, :].argmax().item()
        target_token = tokenizer.decode([target_id])

    clean_logit = clean_logits_val[0, -1, target_id].item()

    # Step 2: Corrupted baseline
    with wrapper.trace(corrupted_prompt):
        corrupted_logits = wrapper.get_logits().save()

    corrupted_logit = corrupted_logits.detach().float()[0, -1, target_id].item()

    logit_range = clean_logit - corrupted_logit
    if abs(logit_range) < 1e-6:
        logit_range = 1.0

    # Step 3: Patching - one trace per (layer, position) or per layer
    if patch_mode == "per_position":
        patching_matrix = torch.zeros(len(layers), num_positions)
        total_steps = len(layers) * num_positions
        step = 0

        for i, layer_idx in enumerate(layers):
            for pos in range(num_positions):
                if progress_callback:
                    cont = progress_callback(step / total_steps, f"Patching layer {layer_idx}, pos {pos}")
                    if cont is False:
                        return {"cancelled": True}

                with wrapper.trace(corrupted_prompt):
                    layer_out = wrapper.get_layer_output(layer_idx)
                    clean_val = clean_hidden[layer_idx].detach()
                    # layer output is 2D [seq_len, hidden] - no batch dim
                    layer_out[pos, :] = clean_val[pos, :]
                    patched_logits = wrapper.get_logits().save()

                patched_logit = patched_logits.detach().float()[0, -1, target_id].item()
                recovery = (patched_logit - corrupted_logit) / logit_range
                patching_matrix[i, pos] = recovery
                step += 1
    else:
        patching_matrix = torch.zeros(len(layers), 1)
        total_steps = len(layers)

        for i, layer_idx in enumerate(layers):
            if progress_callback:
                cont = progress_callback(i / total_steps, f"Patching layer {layer_idx}")
                if cont is False:
                    return {"cancelled": True}

            with wrapper.trace(corrupted_prompt):
                layer_out = wrapper.get_layer_output(layer_idx)
                clean_val = clean_hidden[layer_idx].detach()
                layer_out[:] = clean_val
                patched_logits = wrapper.get_logits().save()

            patched_logit = patched_logits.detach().float()[0, -1, target_id].item()
            recovery = (patched_logit - corrupted_logit) / logit_range
            patching_matrix[i, 0] = recovery

    if progress_callback:
        progress_callback(1.0, "Complete")

    return {
        "clean_prompt": clean_prompt,
        "corrupted_prompt": corrupted_prompt,
        "target_token": target_token,
        "target_id": target_id,
        "clean_logit": clean_logit,
        "corrupted_logit": corrupted_logit,
        "patching_matrix": patching_matrix.detach().cpu(),
        "tokens": tokens,
        "layers": layers,
        "patch_mode": patch_mode,
    }
