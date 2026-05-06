from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from core.loader import ModelWrapper


def compute_mean_activations(
    wrapper: ModelWrapper,
    prompts: list[str],
    component: str = "layers",
    batch_size: int = 8,
    progress_callback: Callable[[float, str], bool] | None = None,
) -> dict[int, torch.Tensor]:
    means: dict[int, torch.Tensor] = {}
    counts: dict[int, int] = {}
    num_layers = wrapper.num_layers
    total = len(prompts)

    for start in range(0, total, batch_size):
        batch = prompts[start : start + batch_size]
        if progress_callback:
            progress_callback(start / total, f"Computing means: batch {start // batch_size + 1}")

        for prompt in batch:
            saved = {}
            with wrapper.trace(prompt):
                for layer_idx in range(num_layers):
                    if component == "attention":
                        saved[layer_idx] = wrapper.get_attention_output(layer_idx).save()
                    elif component == "mlp":
                        saved[layer_idx] = wrapper.get_mlp_output(layer_idx).save()
                    else:
                        saved[layer_idx] = wrapper.get_layer_output(layer_idx).save()

            for layer_idx in range(num_layers):
                val = saved[layer_idx]
                if isinstance(val, tuple):
                    val = val[0]
                # Normalize to 2D [seq_len, hidden] regardless of whether batch dim exists
                v = val.detach().float()
                if v.dim() == 3:
                    v = v[0]
                val_mean = v.mean(dim=0)  # [hidden]

                if layer_idx not in means:
                    means[layer_idx] = val_mean
                    counts[layer_idx] = 1
                else:
                    means[layer_idx] = means[layer_idx] + val_mean
                    counts[layer_idx] += 1

    for layer_idx in means:
        means[layer_idx] = means[layer_idx] / counts[layer_idx]  # [hidden]

    return means


def run_ablation(
    wrapper: ModelWrapper,
    prompt: str,
    ablation_type: str,
    component: str = "layers",
    layers: list[int] | None = None,
    mean_activations: dict[int, torch.Tensor] | None = None,
    progress_callback: Callable[[float, str], bool] | None = None,
) -> dict:
    if layers is None:
        layers = list(range(wrapper.num_layers))

    tokenizer = wrapper.tokenizer
    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(prompt))

    # Baseline run
    with wrapper.trace(prompt):
        baseline_logits = wrapper.get_logits().save()

    baseline_logits_val = baseline_logits.detach().float()
    baseline_probs = F.softmax(baseline_logits_val[0, -1, :], dim=-1)
    baseline_top_id = baseline_probs.argmax().item()
    baseline_top_token = tokenizer.decode([baseline_top_id])
    baseline_top_prob = baseline_probs[baseline_top_id].item()

    ablated_results = []
    total = len(layers)

    for i, layer_idx in enumerate(layers):
        if progress_callback:
            cont = progress_callback(i / total, f"Ablating layer {layer_idx}")
            if cont is False:
                return {"cancelled": True}

        with wrapper.trace(prompt):
            if component == "attention":
                output_ref = wrapper.get_attention_output(layer_idx)
            elif component == "mlp":
                output_ref = wrapper.get_mlp_output(layer_idx)
            else:
                output_ref = wrapper.get_layer_output(layer_idx)

            if ablation_type == "zero":
                output_ref[:] = 0
            elif ablation_type == "mean" and mean_activations is not None:
                mean_val = mean_activations[layer_idx]
                output_ref[:] = mean_val.to(output_ref.device)
            else:
                output_ref[:] = 0

            ablated_logits = wrapper.get_logits().save()

        ablated_logits_val = ablated_logits.detach().float()
        ablated_probs = F.softmax(ablated_logits_val[0, -1, :], dim=-1)
        ablated_top_id = ablated_probs.argmax().item()
        ablated_top_token = tokenizer.decode([ablated_top_id])
        ablated_top_prob = ablated_probs[ablated_top_id].item()

        # KL divergence: KL(baseline || ablated)
        kl = F.kl_div(
            ablated_probs.log().unsqueeze(0),
            baseline_probs.unsqueeze(0),
            reduction="batchmean",
            log_target=False,
        ).item()

        logit_diff = baseline_logits_val[0, -1, baseline_top_id].item() - ablated_logits_val[0, -1, baseline_top_id].item()

        ablated_results.append({
            "layer": layer_idx,
            "top_token": ablated_top_token,
            "top_prob": ablated_top_prob,
            "kl_divergence": kl,
            "logit_diff": logit_diff,
            "prediction_changed": ablated_top_id != baseline_top_id,
        })

    if progress_callback:
        progress_callback(1.0, "Complete")

    return {
        "prompt": prompt,
        "tokens": tokens,
        "ablation_type": ablation_type,
        "component": component,
        "baseline_top_token": baseline_top_token,
        "baseline_top_prob": baseline_top_prob,
        "ablated_results": ablated_results,
        "layers": layers,
    }
