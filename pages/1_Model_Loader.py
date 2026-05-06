import streamlit as st

from utils.state import init_session_state

import config
from core.loader import load_model
from core.model_info import get_model_info, format_model_info_markdown
from utils.device import get_device_info, get_recommended_settings, estimate_model_vram

st.header("Model Loader")

init_session_state()

st.caption(
    "Load a transformer model from HuggingFace to your GPU. "
    "All other pages require a loaded model, so start here. "
    "GPT-2 is a good first choice: small, free, and ready in under a minute."
)


@st.cache_data(show_spinner=False)
def _fetch_model_config(model_name: str):
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_name)
    return {
        "hidden_size": cfg.hidden_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "vocab_size": cfg.vocab_size,
    }

dev_info = get_device_info()

# --- Device info panel ---
with st.expander("Device Information", expanded=False):
    col1, col2, col3 = st.columns(3)
    col1.metric("Device", dev_info["device_name"])
    col2.metric("VRAM Total", f"{dev_info['vram_total_gb']:.1f} GB" if dev_info["cuda_available"] else "N/A")
    col3.metric("VRAM Free", f"{dev_info['vram_free_gb']:.1f} GB" if dev_info["cuda_available"] else "N/A")

    if not dev_info["cuda_available"]:
        st.warning(
            "No CUDA GPU detected. Models will run on CPU (slow). "
            "Quantization requires CUDA and will be disabled."
        )

    recs = get_recommended_settings(dev_info["vram_free_gb"])
    for w in recs["warnings"]:
        st.warning(w)

# --- Model selection ---
st.subheader("Select Model")

model_source = st.radio("Source", ["HuggingFace Hub", "Custom model ID"], horizontal=True)

if model_source == "HuggingFace Hub":
    model_name = st.selectbox("Model", config.DEFAULT_MODELS)
else:
    model_name = st.text_input("Model ID or local path", placeholder="meta-llama/Llama-3.2-1B")

# --- Quantization ---
quant_label = st.radio(
    "Quantization",
    list(config.QUANT_OPTIONS.keys()),
    horizontal=True,
    help="8-bit and 4-bit reduce VRAM usage but require CUDA.",
)
quant = config.QUANT_OPTIONS[quant_label]

if quant != "none" and not dev_info["cuda_available"]:
    st.warning("Quantization requires CUDA. Will load in full precision on CPU.")
    quant = "none"

# --- Load / Unload ---
col_load, col_unload = st.columns(2)

with col_load:
    if st.button("Load Model", type="primary", disabled=not model_name, use_container_width=True):
        with st.spinner(f"Loading {model_name}..."):
            try:
                wrapper = load_model(model_name, quant)
                st.session_state["model"] = wrapper
                st.session_state["model_name"] = model_name
                st.session_state["model_info"] = get_model_info(wrapper)
                st.session_state["quantization"] = quant
                # Clear stale analysis results
                for key in ("logit_lens_result", "patching_result", "ablation_result", "sae_model", "sae_features"):
                    st.session_state[key] = None
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load model: {e}")

with col_unload:
    if st.button(
        "Unload Model",
        disabled=st.session_state["model"] is None,
        use_container_width=True,
    ):
        load_model.clear()
        for key in ("model", "model_info", "logit_lens_result", "patching_result",
                     "ablation_result", "sae_model", "sae_features"):
            st.session_state[key] = None
        st.session_state["model_name"] = ""
        st.session_state["quantization"] = "none"
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        st.rerun()

# --- VRAM estimate (cached, shown after buttons to avoid blocking initial render) ---
if model_name:
    try:
        model_cfg = _fetch_model_config(model_name)
        est_params = (
            model_cfg["hidden_size"] * model_cfg["hidden_size"] * 4 * model_cfg["num_hidden_layers"]
            + model_cfg["hidden_size"] * model_cfg["vocab_size"] * 2
        )
        est_vram = estimate_model_vram(est_params, quant)
        st.caption(f"Estimated VRAM: ~{est_vram:.1f} GB")
        if dev_info["cuda_available"] and est_vram > dev_info["vram_free_gb"]:
            st.warning(
                f"Estimated VRAM ({est_vram:.1f} GB) exceeds available ({dev_info['vram_free_gb']:.1f} GB). "
                "Consider using stronger quantization."
            )
    except Exception:
        st.caption("Could not estimate VRAM for this model.")

# --- Current model info ---
if st.session_state["model_info"] is not None:
    st.subheader("Loaded Model")
    st.markdown(format_model_info_markdown(st.session_state["model_info"]))
