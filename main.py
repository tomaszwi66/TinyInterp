import streamlit as st

from utils.state import init_session_state

st.set_page_config(
    page_title="TinyInterp",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()


# --- Sidebar ---
with st.sidebar:
    st.title("TinyInterp")
    st.caption("Mechanistic Interpretability Explorer")
    st.divider()

    if st.session_state["model"] is not None:
        info = st.session_state["model_info"]
        st.success(f"**{info['model_name']}**")
        st.caption(
            f"{info['num_params_human']} params | {info['num_layers']}L | "
            f"{info['device']} | {info['dtype']}"
        )
    else:
        st.warning("No model loaded")

    st.divider()
    from utils.device import get_device_info

    dev = get_device_info()
    st.caption(f"**Device:** {dev['device_name']}")
    if dev["cuda_available"]:
        st.caption(f"VRAM: {dev['vram_used_gb']:.1f} / {dev['vram_total_gb']:.1f} GB")
    st.caption(f"PyTorch {dev['torch_version']}")


# --- Main page ---
st.title("TinyInterp")
st.markdown("### Local Mechanistic Interpretability Explorer")

st.markdown("""
**TinyInterp** lets you explore the internal mechanisms of transformer language models -
attention patterns, residual stream evolution, causal tracing, sparse autoencoders, and ablation studies -
all running locally on your machine.

**Quick Start:**
1. Go to **Model Loader** in the sidebar to load a HuggingFace model
2. Run **Basic Analysis** to see logit lens projections
3. Try **Activation Patching** for causal tracing experiments
4. Train or load a **Sparse Autoencoder** to find interpretable features
5. Run **Ablations** to test component necessity
6. Generate **Reports** via local Ollama for scientific analysis
""")

st.info(
    "Navigate using the sidebar pages. Start by loading a model on the **Model Loader** page.",
    icon="👈",
)
