from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

import torch
import torch.nn.functional as F

import config

if TYPE_CHECKING:
    from core.loader import ModelWrapper

logger = logging.getLogger(__name__)


def get_hook_name_for_layer(wrapper: ModelWrapper, layer: int, component: str = "resid_post") -> str:
    model_type = getattr(wrapper.config, "model_type", "")

    if model_type in ("gpt2", "gpt_neo"):
        base = f"transformer.h.{layer}"
    elif model_type in ("llama", "mistral", "gemma", "gemma2", "phi", "qwen2", "qwen3"):
        base = f"model.layers.{layer}"
    elif model_type == "opt":
        base = f"model.decoder.layers.{layer}"
    elif model_type == "gpt_neox":
        base = f"gpt_neox.layers.{layer}"
    else:
        base = f"model.layers.{layer}"
        logger.warning(f"Unknown model_type '{model_type}', guessing hook name: {base}")

    return base


def get_available_pretrained_saes(model_name: str) -> list[dict]:
    return config.PRETRAINED_SAE_RELEASES.get(model_name, [])


def load_pretrained_sae(release: str, sae_id: str, device: str = "cpu"):
    from sae_lens import SAE

    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
        device=device,
    )
    return sae


def extract_activations(
    wrapper: ModelWrapper,
    prompts: list[str],
    layer: int,
    batch_size: int = 8,
    progress_callback: Callable[[float, str], bool] | None = None,
) -> torch.Tensor:
    all_activations = []
    total = len(prompts)

    for start in range(0, total, batch_size):
        batch = prompts[start : start + batch_size]
        if progress_callback:
            progress_callback(start / total, f"Extracting activations: batch {start // batch_size + 1}")

        for prompt in batch:
            with wrapper.trace(prompt):
                hidden = wrapper.get_layer_output(layer).save()

            val = hidden.detach()
            if isinstance(val, tuple):
                val = val[0]
            v = val.float().cpu()
            if v.dim() == 3:
                v = v[0]  # [1, seq_len, hidden] -> [seq_len, hidden]
            all_activations.append(v)

    return torch.cat(all_activations, dim=0)


def train_sae(
    wrapper: ModelWrapper,
    layer: int,
    training_tokens: int = 10_000_000,
    expansion_factor: int = 16,
    lr: float = 5e-5,
    l1_coefficient: float = 5.0,
    batch_size: int = 4096,
    context_size: int = 256,
    dataset_path: str = "apollo-research/Skylion007-openwebtext-tokenizer-gpt2",
    progress_callback: Callable[[float, str], bool] | None = None,
):
    from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner
    from sae_lens.config import LoggingConfig
    from sae_lens.saes.standard_sae import StandardTrainingSAEConfig

    hook_name = get_hook_name_for_layer(wrapper, layer)
    d_in = wrapper.hidden_size
    d_sae = d_in * expansion_factor

    if progress_callback:
        progress_callback(0.0, "Configuring SAE training...")

    sae_cfg = StandardTrainingSAEConfig(
        d_in=d_in,
        d_sae=d_sae,
        l1_coefficient=l1_coefficient,
        device=str(wrapper.device),
    )

    cfg = LanguageModelSAERunnerConfig(
        sae=sae_cfg,
        model_name=wrapper.name,
        model_class_name="AutoModelForCausalLM",
        hook_name=hook_name,
        dataset_path=dataset_path,
        streaming=True,
        context_size=context_size,
        is_dataset_tokenized=True,
        training_tokens=training_tokens,
        train_batch_size_tokens=batch_size,
        lr=lr,
        n_checkpoints=5,
        checkpoint_path=str(config.CHECKPOINTS_DIR),
        output_path=str(config.RESULTS_DIR / "sae_output"),
        logger=LoggingConfig(log_to_wandb=False),
        device=str(wrapper.device),
    )

    if progress_callback:
        progress_callback(0.1, "Starting SAE training (this may take a while)...")

    runner = SAETrainingRunner(cfg)
    sae = runner.run()

    if progress_callback:
        progress_callback(1.0, "Training complete")

    return sae


def analyze_sae_features(
    sae,
    activations: torch.Tensor,
    top_k: int = 20,
    token_position: int | None = None,
) -> dict:
    device = next(sae.parameters()).device
    acts = activations.to(device)

    with torch.no_grad():
        feature_acts = sae.encode(acts)
        reconstructed = sae.decode(feature_acts)

    mse = F.mse_loss(reconstructed, acts).item()

    if token_position is None:
        feature_scores = feature_acts.mean(dim=0)
        score_label = "Mean activation"
    else:
        if token_position < 0 or token_position >= feature_acts.shape[0]:
            raise ValueError(f"token_position must be between 0 and {feature_acts.shape[0] - 1}")
        feature_scores = feature_acts[token_position]
        score_label = f"Activation at token {token_position}"

    top_features = feature_scores.topk(min(top_k, feature_scores.shape[0])).indices.tolist()

    density = (feature_acts > 0).float().mean(dim=0)

    return {
        "feature_activations": feature_acts.cpu(),
        "top_features": top_features,
        "top_feature_mean_acts": feature_scores[top_features].cpu().tolist(),
        "top_feature_scores": feature_scores[top_features].cpu().tolist(),
        "score_label": score_label,
        "token_position": token_position,
        "feature_densities": density.cpu(),
        "reconstruction_mse": mse,
        "num_active_features": (feature_scores > 0).sum().item(),
        "total_features": feature_acts.shape[-1],
    }


def get_feature_top_tokens(
    sae,
    wrapper: ModelWrapper,
    feature_idx: int,
    top_k: int = 20,
) -> list[tuple[str, float]]:
    W_dec = sae.W_dec.data[feature_idx].detach().float()

    ln_final = wrapper.get_ln_final()
    lm_head = wrapper.get_lm_head()

    with torch.no_grad():
        normed = ln_final(W_dec.unsqueeze(0).to(next(ln_final.parameters()).device))
        logits = lm_head(normed)[0]

    top_logits, top_indices = logits.topk(top_k)
    tokenizer = wrapper.tokenizer
    tokens = tokenizer.convert_ids_to_tokens(top_indices.cpu().tolist())
    return list(zip(tokens, top_logits.cpu().tolist()))
