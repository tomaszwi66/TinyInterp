from __future__ import annotations

from typing import TYPE_CHECKING

from utils.device import estimate_model_vram

if TYPE_CHECKING:
    from core.loader import ModelWrapper


def get_model_info(wrapper: ModelWrapper) -> dict:
    num_params = wrapper.total_params()
    dtype_str = "unknown"
    try:
        first_param = next(wrapper.raw.parameters())
        dtype_str = str(first_param.dtype).replace("torch.", "")
    except StopIteration:
        pass

    return {
        "model_name": wrapper.name,
        "model_type": getattr(wrapper.config, "model_type", "unknown"),
        "num_layers": wrapper.num_layers,
        "hidden_size": wrapper.hidden_size,
        "num_heads": wrapper.num_heads,
        "head_dim": wrapper.hidden_size // wrapper.num_heads,
        "vocab_size": wrapper.vocab_size,
        "intermediate_size": getattr(wrapper.config, "intermediate_size", None),
        "max_position_embeddings": getattr(wrapper.config, "max_position_embeddings", None),
        "num_params": num_params,
        "num_params_human": _format_params(num_params),
        "dtype": dtype_str,
        "device": str(wrapper.device),
        "is_standardized": wrapper.is_standardized,
    }


def _format_params(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    if n >= 1e6:
        return f"{n / 1e6:.0f}M"
    if n >= 1e3:
        return f"{n / 1e3:.0f}K"
    return str(n)


def format_model_info_markdown(info: dict) -> str:
    rows = [
        ("Model", info["model_name"]),
        ("Architecture", info["model_type"]),
        ("Parameters", info["num_params_human"]),
        ("Layers", info["num_layers"]),
        ("Hidden Size", info["hidden_size"]),
        ("Attention Heads", info["num_heads"]),
        ("Head Dimension", info["head_dim"]),
        ("Vocabulary Size", f"{info['vocab_size']:,}"),
        ("Dtype", info["dtype"]),
        ("Device", info["device"]),
        ("Backend", "nnterp (standardized)" if info["is_standardized"] else "nnsight (raw)"),
    ]
    if info.get("intermediate_size"):
        rows.insert(6, ("FFN Intermediate", f"{info['intermediate_size']:,}"))
    if info.get("max_position_embeddings"):
        rows.append(("Max Sequence Length", f"{info['max_position_embeddings']:,}"))

    lines = ["| Property | Value |", "|----------|-------|"]
    for label, val in rows:
        lines.append(f"| {label} | {val} |")
    return "\n".join(lines)
