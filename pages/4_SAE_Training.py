import streamlit as st

from utils.state import init_session_state

import config
from core.sae import (
    get_available_pretrained_saes,
    load_pretrained_sae,
    train_sae,
    get_hook_name_for_layer,
)
from utils.background import submit_task, is_task_running, render_progress_fragment
from utils.device import get_device_info, estimate_model_vram

st.header("SAE Training & Loading")

init_session_state()

st.caption(
    "Sparse Autoencoders (SAE) decompose the model's internal activations into individual interpretable features. "
    "Think of it as separating white light into a spectrum: one dense activation vector becomes thousands of sparse, named directions. "
    "Load a pre-trained SAE for GPT-2 or Gemma, or train your own on any layer. "
    "After loading, go to Feature Explorer to see what each feature represents."
)

if st.session_state.get("model") is None:
    st.warning("Load a model first on the **Model Loader** page.")
    st.stop()

wrapper = st.session_state["model"]
info = st.session_state["model_info"]

TASK_ID = "sae_training"

tab_pretrained, tab_train = st.tabs(["Load Pre-trained SAE", "Train New SAE"])

# --- Pre-trained SAE loading ---
with tab_pretrained:
    st.subheader("Load Pre-trained SAE")

    releases = get_available_pretrained_saes(info["model_name"])

    if not releases:
        st.info(
            f"No known pre-trained SAEs for **{info['model_name']}**. "
            "Try GPT-2 or Gemma-2-2B, or train your own in the other tab."
        )
    else:
        release_names = [r["release"] for r in releases]
        selected_release_name = st.selectbox("SAE Release", release_names)
        selected_release = next(r for r in releases if r["release"] == selected_release_name)

        selected_layer = st.selectbox(
            "Layer",
            selected_release["layers"],
            format_func=lambda l: f"Layer {l}",
        )

        hook_name = get_hook_name_for_layer(wrapper, selected_layer)
        sae_id = f"blocks.{selected_layer}.hook_resid_pre"
        st.caption(f"SAE ID: `{sae_id}` | Hook: `{hook_name}`")

        if st.button("Load Pre-trained SAE", type="primary"):
            with st.spinner("Loading SAE from SAELens registry..."):
                try:
                    device = "cuda" if get_device_info()["cuda_available"] else "cpu"
                    sae = load_pretrained_sae(selected_release_name, sae_id, device=device)
                    st.session_state["sae_model"] = sae
                    st.session_state["sae_features"] = None
                    st.success(
                        f"Loaded SAE: {selected_release_name} / {sae_id} "
                        f"(d_sae={sae.cfg.d_sae})"
                    )
                except Exception as e:
                    st.error(f"Failed to load SAE: {e}")

# --- Train new SAE ---
with tab_train:
    st.subheader("Train New SAE")

    col1, col2 = st.columns(2)
    with col1:
        train_layer = st.selectbox(
            "Target layer",
            list(range(info["num_layers"])),
            format_func=lambda l: f"Layer {l}",
            key="sae_train_layer",
        )
        expansion_factor = st.number_input(
            "Expansion factor (d_sae / d_in)",
            min_value=1,
            max_value=128,
            value=config.SAE_DEFAULT_CONFIG["expansion_factor"],
        )
        training_tokens = st.number_input(
            "Training tokens",
            min_value=100_000,
            max_value=1_000_000_000,
            value=config.SAE_DEFAULT_CONFIG["training_tokens"],
            step=1_000_000,
        )

    with col2:
        lr = st.number_input(
            "Learning rate",
            min_value=1e-6,
            max_value=1e-2,
            value=config.SAE_DEFAULT_CONFIG["lr"],
            format="%.1e",
        )
        l1_coeff = st.number_input(
            "L1 coefficient",
            min_value=0.0,
            max_value=100.0,
            value=config.SAE_DEFAULT_CONFIG["l1_coefficient"],
        )
        batch_size = st.number_input(
            "Batch size (tokens)",
            min_value=256,
            max_value=65536,
            value=config.SAE_DEFAULT_CONFIG["batch_size"],
            step=256,
        )
        context_size = st.number_input(
            "Context size",
            min_value=32,
            max_value=2048,
            value=config.SAE_DEFAULT_CONFIG["context_size"],
            step=32,
        )

    dataset_path = st.text_input(
        "Dataset (HuggingFace path)",
        value="apollo-research/Skylion007-openwebtext-tokenizer-gpt2",
    )

    # VRAM estimate
    d_sae = expansion_factor * info["hidden_size"]
    sae_params = 2 * d_sae * info["hidden_size"]
    sae_vram_gb = (sae_params * 4 * 3) / (1024 ** 3)  # weights + grads + optimizer
    st.caption(
        f"SAE dimensions: d_in={info['hidden_size']}, d_sae={d_sae:,} | "
        f"Estimated SAE VRAM: ~{sae_vram_gb:.1f} GB"
    )

    dev_info = get_device_info()
    if dev_info["cuda_available"] and sae_vram_gb > dev_info["vram_free_gb"] * 0.5:
        st.warning("SAE training may use significant VRAM. Consider reducing expansion factor.")

    def _train_sae_task(progress_callback, wrapper, layer, tokens, ef, lr, l1, bs, cs, ds):
        return train_sae(
            wrapper, layer,
            training_tokens=tokens,
            expansion_factor=ef,
            lr=lr,
            l1_coefficient=l1,
            batch_size=bs,
            context_size=cs,
            dataset_path=ds,
            progress_callback=progress_callback,
        )

    if st.button("Start Training", type="primary", disabled=is_task_running(TASK_ID)):
        submit_task(
            TASK_ID,
            _train_sae_task,
            args=(wrapper, train_layer, training_tokens, expansion_factor, lr, l1_coeff, batch_size, context_size, dataset_path),
        )

    if is_task_running(TASK_ID):
        render_progress_fragment(TASK_ID, "sae_model", "SAE training complete!")

        if st.button("Cancel Training"):
            from utils.background import cancel_task
            cancel_task(TASK_ID)

# --- Current SAE status ---
st.divider()
sae = st.session_state.get("sae_model")
if sae is not None:
    st.success(f"SAE loaded: d_sae={sae.cfg.d_sae}, d_in={sae.cfg.d_in}")
else:
    st.info("No SAE loaded. Load a pre-trained one or train a new one above.")
