import streamlit as st

from utils.state import init_session_state
import plotly.graph_objects as go
import pandas as pd

from core.ablation import run_ablation, compute_mean_activations
from utils.background import submit_task, is_task_running, render_progress_fragment
from utils.export import get_download_button

TASK_ID = "ablation"

st.header("Ablation Studies")

init_session_state()

if st.session_state.get("model") is None:
    st.warning("Load a model first on the **Model Loader** page.")
    st.stop()

wrapper = st.session_state["model"]
info = st.session_state["model_info"]

st.caption(
    "Ablation studies measure how important each component is to the model's prediction. "
    "Zero ablation replaces the output of a layer with zeros. "
    "Mean ablation replaces it with the average activation computed over a reference dataset. "
    "A large KL divergence or a changed top prediction means that component is necessary."
)

# --- Inputs ---
prompt = st.text_area("Prompt", value="Berlin is the capital of", height=80)

col1, col2 = st.columns(2)
with col1:
    ablation_type = st.radio("Ablation type", ["zero", "mean"], horizontal=True)
with col2:
    component = st.radio("Component", ["layers", "attention", "mlp"], horizontal=True)

layer_range = st.slider(
    "Layer range",
    min_value=0,
    max_value=info["num_layers"] - 1,
    value=(0, info["num_layers"] - 1),
)
layers = list(range(layer_range[0], layer_range[1] + 1))

# --- Mean ablation dataset ---
mean_dataset = None
if ablation_type == "mean":
    st.markdown("**Dataset for mean activation computation:**")
    mean_text = st.text_area(
        "Prompts (one per line)",
        value="Paris is the capital of\nRome is the capital of\nMadrid is the capital of\nTokyo is the capital of\nBrasilia is the capital of",
        height=120,
    )
    mean_dataset = [line.strip() for line in mean_text.split("\n") if line.strip()]
    if len(mean_dataset) < 2:
        st.warning("Provide at least 2 prompts for mean ablation.")


# --- Run ---
def _run_ablation_task(progress_callback, wrapper, prompt, ablation_type, component, layers, mean_dataset):
    mean_acts = None
    if ablation_type == "mean" and mean_dataset:
        mean_acts = compute_mean_activations(
            wrapper, mean_dataset, component,
            progress_callback=lambda p, m: progress_callback(p * 0.3, f"Mean activations: {m}"),
        )
    return run_ablation(
        wrapper, prompt, ablation_type, component, layers,
        mean_activations=mean_acts,
        progress_callback=lambda p, m: progress_callback(0.3 + p * 0.7, m),
    )


run_disabled = is_task_running(TASK_ID) or (ablation_type == "mean" and (not mean_dataset or len(mean_dataset) < 2))

if st.button("Run Ablation", type="primary", disabled=run_disabled):
    submit_task(
        TASK_ID,
        _run_ablation_task,
        args=(wrapper, prompt, ablation_type, component, layers, mean_dataset),
    )

if is_task_running(TASK_ID):
    render_progress_fragment(TASK_ID, "ablation_result", "Ablation analysis complete!")

# --- Display results ---
result = st.session_state.get("ablation_result")
if result is not None and not result.get("cancelled"):
    st.subheader("Results")

    st.markdown(
        f"**Baseline prediction:** `{result['baseline_top_token']}` "
        f"(p = {result['baseline_top_prob']:.4f})"
    )

    ablated = result["ablated_results"]

    # KL divergence bar chart
    kl_values = [r["kl_divergence"] for r in ablated]
    layer_labels = [f"Layer {r['layer']}" for r in ablated]

    colors = ["red" if r["prediction_changed"] else "steelblue" for r in ablated]

    fig = go.Figure(
        data=go.Bar(
            x=layer_labels,
            y=kl_values,
            marker_color=colors,
            hovertemplate=(
                "%{x}<br>"
                "KL divergence: %{y:.4f}<br>"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=f"KL Divergence per Layer ({ablation_type} ablation, {component})",
        xaxis_title="Layer",
        yaxis_title="KL Divergence (nats)",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Red bars indicate the ablation changed the top prediction.")

    # Table
    df = pd.DataFrame([
        {
            "Layer": r["layer"],
            "Ablated Top Token": r["top_token"],
            "Ablated Top Prob": f"{r['top_prob']:.4f}",
            "KL Divergence": f"{r['kl_divergence']:.4f}",
            "Logit Diff": f"{r['logit_diff']:.4f}",
            "Changed": "Yes" if r["prediction_changed"] else "No",
        }
        for r in ablated
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)

    with st.expander("Export"):
        get_download_button(result, "json", "ablation_result.json")
