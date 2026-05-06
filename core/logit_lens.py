from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from core.loader import ModelWrapper


def run_logit_lens(
    wrapper: ModelWrapper,
    prompt: str,
    top_k: int = 10,
    layers: list[int] | None = None,
    all_positions: bool = True,
) -> dict:
    if layers is None:
        layers = list(range(wrapper.num_layers))

    tokenizer = wrapper.tokenizer
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    saved_hidden = {}
    with wrapper.trace(prompt):
        for layer_idx in layers:
            saved_hidden[layer_idx] = wrapper.get_layer_output(layer_idx).save()

    all_probs = []
    all_top_k_tokens = []
    all_top_k_probs = []

    for layer_idx in layers:
        hidden = saved_hidden[layer_idx]
        if isinstance(hidden, tuple):
            hidden = hidden[0]

        logits = wrapper.project_on_vocab(hidden)
        # logits shape: [seq_len, vocab] (nnsight strips the batch dim)
        if logits.dim() == 3:
            logits = logits[0]  # [1, seq_len, vocab] -> [seq_len, vocab]

        if not all_positions:
            logits = logits[-1:, :]  # [1, vocab]

        probs = F.softmax(logits.float(), dim=-1)  # [seq_len, vocab] or [1, vocab]

        top_probs, top_indices = probs.topk(top_k, dim=-1)  # [seq_len, top_k]

        layer_top_tokens = []
        for pos_idx in range(top_indices.shape[0]):
            pos_tokens = tokenizer.convert_ids_to_tokens(top_indices[pos_idx].tolist())
            layer_top_tokens.append(pos_tokens)

        all_probs.append(probs.detach().cpu())
        all_top_k_tokens.append(layer_top_tokens)
        all_top_k_probs.append(top_probs.detach().cpu())

    probs_tensor = torch.stack(all_probs)
    top_k_probs_tensor = torch.stack(all_top_k_probs)

    return {
        "prompt": prompt,
        "tokens": tokens,
        "layers": layers,
        "probs": probs_tensor,
        "top_k_tokens": all_top_k_tokens,
        "top_k_probs": top_k_probs_tensor,
        "all_positions": all_positions,
    }


def get_prediction_at_layer(result: dict, layer_idx: int, position: int = -1) -> tuple[str, float]:
    layer_pos = result["layers"].index(layer_idx)
    if position == -1:
        position = result["top_k_probs"].shape[1] - 1
    token = result["top_k_tokens"][layer_pos][position][0]
    prob = result["top_k_probs"][layer_pos][position][0].item()
    return token, prob


def build_logit_lens_matrix(result: dict) -> tuple[list[list[str]], list[list[float]]]:
    token_matrix = []
    prob_matrix = []
    for layer_pos in range(len(result["layers"])):
        layer_tokens = []
        layer_probs = []
        num_positions = result["top_k_probs"].shape[1]
        for pos in range(num_positions):
            layer_tokens.append(result["top_k_tokens"][layer_pos][pos][0])
            layer_probs.append(result["top_k_probs"][layer_pos][pos][0].item())
        token_matrix.append(layer_tokens)
        prob_matrix.append(layer_probs)
    return token_matrix, prob_matrix
