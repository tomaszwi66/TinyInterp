from __future__ import annotations

import logging
from typing import Generator

import torch

import config

logger = logging.getLogger(__name__)


def check_ollama_available() -> tuple[bool, list[str]]:
    try:
        import ollama

        models_response = ollama.list()
        model_names = []
        if hasattr(models_response, "models"):
            model_names = [m.model for m in models_response.models]
        elif isinstance(models_response, dict) and "models" in models_response:
            model_names = [m.get("model", m.get("name", "")) for m in models_response["models"]]
        return True, model_names
    except Exception as e:
        logger.warning(f"Ollama not available: {e}")
        return False, []


def format_results_for_prompt(analysis_type: str, results: dict, model_info: dict) -> str:
    sections = [
        f"## Model: {model_info['model_name']}",
        f"Architecture: {model_info['model_type']}, {model_info['num_params_human']} parameters, "
        f"{model_info['num_layers']} layers, {model_info['hidden_size']} hidden dim, "
        f"{model_info['num_heads']} attention heads",
        "",
    ]

    if analysis_type == "logit_lens":
        sections.append(f"## Logit Lens Analysis")
        sections.append(f"Prompt: \"{results['prompt']}\"")
        sections.append(f"Tokens: {results['tokens']}")
        sections.append(f"Layers analyzed: {results['layers']}")
        sections.append("")
        sections.append("Top-1 predictions by layer (last token position):")
        for i, layer_idx in enumerate(results["layers"]):
            num_pos = len(results["top_k_tokens"][i])
            last_pos = num_pos - 1
            top_token = results["top_k_tokens"][i][last_pos][0]
            top_prob = results["top_k_probs"][i][last_pos][0].item() if isinstance(results["top_k_probs"], torch.Tensor) else results["top_k_probs"][i][last_pos][0]
            sections.append(f"  Layer {layer_idx}: '{top_token}' (p={top_prob:.4f})")

    elif analysis_type == "activation_patching":
        sections.append("## Activation Patching Results")
        sections.append(f"Clean prompt: \"{results['clean_prompt']}\"")
        sections.append(f"Corrupted prompt: \"{results['corrupted_prompt']}\"")
        sections.append(f"Target token: '{results['target_token']}'")
        sections.append(f"Clean logit: {results['clean_logit']:.4f}")
        sections.append(f"Corrupted logit: {results['corrupted_logit']:.4f}")
        sections.append(f"Logit difference: {results['clean_logit'] - results['corrupted_logit']:.4f}")
        sections.append("")
        matrix = results["patching_matrix"]
        if isinstance(matrix, torch.Tensor):
            matrix = matrix.numpy()
        sections.append("Recovery fraction by layer:")
        for i, layer_idx in enumerate(results["layers"]):
            if matrix.shape[1] == 1:
                sections.append(f"  Layer {layer_idx}: {matrix[i, 0]:.4f}")
            else:
                vals = ", ".join(f"{matrix[i, j]:.3f}" for j in range(min(matrix.shape[1], 10)))
                sections.append(f"  Layer {layer_idx}: [{vals}]")

    elif analysis_type == "ablation":
        sections.append("## Ablation Study Results")
        sections.append(f"Prompt: \"{results['prompt']}\"")
        sections.append(f"Ablation type: {results['ablation_type']}")
        sections.append(f"Component: {results['component']}")
        sections.append(f"Baseline prediction: '{results['baseline_top_token']}' (p={results['baseline_top_prob']:.4f})")
        sections.append("")
        sections.append("Per-layer ablation effects:")
        for r in results["ablated_results"]:
            sections.append(
                f"  Layer {r['layer']}: prediction='{r['top_token']}' (p={r['top_prob']:.4f}), "
                f"KL={r['kl_divergence']:.4f}, logit_diff={r['logit_diff']:.4f}"
                f"{' [CHANGED]' if r['prediction_changed'] else ''}"
            )

    elif analysis_type == "sae_features":
        sections.append("## SAE Feature Analysis")
        sections.append(f"Active features: {results['num_active_features']} / {results['total_features']}")
        sections.append(f"Reconstruction MSE: {results['reconstruction_mse']:.6f}")
        sections.append("")
        sections.append("Top features by mean activation:")
        for idx, act in zip(results["top_features"], results["top_feature_mean_acts"]):
            density = results["feature_densities"][idx].item() if isinstance(results["feature_densities"], torch.Tensor) else results["feature_densities"][idx]
            sections.append(f"  Feature {idx}: mean_act={act:.4f}, density={density:.4f}")

    return "\n".join(sections)


def generate_report(
    analysis_type: str,
    results: dict,
    model_info: dict,
    ollama_model: str = "llama3.2",
    stream: bool = True,
) -> Generator[str, None, None] | str:
    import ollama

    formatted = format_results_for_prompt(analysis_type, results, model_info)

    user_prompt = (
        f"Analyze the following mechanistic interpretability experiment results. "
        f"Provide a detailed scientific interpretation.\n\n{formatted}"
    )

    messages = [
        {"role": "system", "content": config.OLLAMA_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    if stream:
        def _stream():
            response = ollama.chat(
                model=ollama_model,
                messages=messages,
                stream=True,
            )
            for chunk in response:
                try:
                    content = chunk.message.content or ""
                except AttributeError:
                    content = (chunk.get("message") or {}).get("content", "")
                if content:
                    yield content
        return _stream()
    else:
        response = ollama.chat(
            model=ollama_model,
            messages=messages,
            stream=False,
        )
        try:
            return response.message.content
        except AttributeError:
            return response["message"]["content"]
