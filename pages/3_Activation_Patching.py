import streamlit as st

from utils.state import init_session_state
import plotly.graph_objects as go
import numpy as np

from core.patching import run_activation_patching
from utils.background import submit_task, is_task_running, render_progress_fragment
from utils.export import get_download_button

TASK_ID = "activation_patching"

st.header("Activation Patching")

init_session_state()

if st.session_state.get("model") is None:
    st.warning("Load a model first on the **Model Loader** page.")
    st.stop()

wrapper = st.session_state["model"]
info = st.session_state["model_info"]

st.caption(
    "Activation patching answers: which layer or position in the model stores a specific piece of information? "
    "You provide a clean prompt and a corrupted version. "
    "The tool replaces activations in the corrupted run with clean ones, one layer at a time, "
    "and measures how much the correct prediction recovers. "
    "High recovery means that component is critical."
)

# --- Inputs ---
col1, col2 = st.columns(2)
with col1:
    clean_prompt = st.text_input(
        "Clean prompt",
        value="The Eiffel Tower is in the city of",
    )
with col2:
    corrupted_prompt = st.text_input(
        "Corrupted prompt",
        value="The Colosseum is in the city of",
    )

col3, col4, col5 = st.columns(3)
with col3:
    target_token = st.text_input(
        "Target token (optional)",
        value="",
        help="Leave empty to use the model's top prediction on the clean prompt.",
    )
with col4:
    patch_mode = st.radio("Patch mode", ["per_layer", "per_position"], horizontal=True)
with col5:
    layer_range = st.slider(
        "Layer range",
        min_value=0,
        max_value=info["num_layers"] - 1,
        value=(0, info["num_layers"] - 1),
    )

layers = list(range(layer_range[0], layer_range[1] + 1))

# --- Tokenization check ---
tokenizer = wrapper.tokenizer
clean_ids = tokenizer.encode(clean_prompt)
corrupted_ids = tokenizer.encode(corrupted_prompt)

if patch_mode == "per_position" and len(clean_ids) != len(corrupted_ids):
    st.error(
        f"Per-position patching requires prompts of equal token length. "
        f"Clean: {len(clean_ids)} tokens, Corrupted: {len(corrupted_ids)} tokens. "
        f"Adjust one of the prompts or use per-layer mode."
    )

# --- Run ---
run_disabled = is_task_running(TASK_ID) or (
    patch_mode == "per_position" and len(clean_ids) != len(corrupted_ids)
)

def _run_patching_task(progress_callback, wrapper, clean_prompt, corrupted_prompt, **kwargs):
    return run_activation_patching(
        wrapper, clean_prompt, corrupted_prompt, progress_callback=progress_callback, **kwargs
    )


if st.button("Run Patching", type="primary", disabled=run_disabled):
    submit_task(
        TASK_ID,
        _run_patching_task,
        args=(wrapper, clean_prompt, corrupted_prompt),
        kwargs={
            "target_token": target_token if target_token else None,
            "layers": layers,
            "patch_mode": patch_mode,
        },
    )

if is_task_running(TASK_ID):
    render_progress_fragment(TASK_ID, "patching_result", "Activation patching complete!")

# --- Display results ---
result = st.session_state.get("patching_result")
if result is not None and not result.get("cancelled"):
    st.subheader("Results")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Target Token", f"'{result['target_token']}'")
    col_b.metric("Clean Logit", f"{result['clean_logit']:.2f}")
    col_c.metric("Corrupted Logit", f"{result['corrupted_logit']:.2f}")

    matrix = result["patching_matrix"].numpy()
    result_layers = result["layers"]

    if result["patch_mode"] == "per_position":
        x_labels = [f"{i}: {t}" for i, t in enumerate(result["tokens"])]
    else:
        x_labels = ["All positions"]

    y_labels = [f"Layer {l}" for l in result_layers]

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=x_labels,
            y=y_labels,
            colorscale="RdBu",
            zmid=0.5,
            colorbar_title="Recovery",
            hovertemplate="Layer %{y}<br>Position %{x}<br>Recovery: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Activation Patching: Recovery Fraction",
        xaxis_title="Position" if result["patch_mode"] == "per_position" else "",
        yaxis_title="Layer",
        height=max(400, len(result_layers) * 25 + 150),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Highlight most important component
    max_idx = np.unravel_index(matrix.argmax(), matrix.shape)
    max_layer = result_layers[max_idx[0]]
    max_recovery = matrix[max_idx]
    if result["patch_mode"] == "per_position":
        st.info(
            f"Highest recovery: **Layer {max_layer}, Position {max_idx[1]}** "
            f"(recovery = {max_recovery:.3f})"
        )
    else:
        st.info(f"Highest recovery: **Layer {max_layer}** (recovery = {max_recovery:.3f})")

    with st.expander("Export"):
        get_download_button(result, "json", "patching_result.json")
