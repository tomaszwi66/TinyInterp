import streamlit as st

from utils.state import init_session_state
import plotly.graph_objects as go
import numpy as np

from core.logit_lens import run_logit_lens, build_logit_lens_matrix
from core.model_info import format_model_info_markdown
from utils.export import get_download_button

st.header("Basic Analysis")

init_session_state()

if st.session_state.get("model") is None:
    st.warning("Load a model first on the **Model Loader** page.")
    st.stop()

wrapper = st.session_state["model"]
info = st.session_state["model_info"]

# --- Model info ---
with st.expander("Model Architecture", expanded=False):
    st.markdown(format_model_info_markdown(info))

st.divider()

# --- Logit Lens ---
st.subheader("Logit Lens")
st.caption(
    "Logit Lens shows what the model predicts at each layer of the network. "
    "Enter a prompt and watch how the prediction forms as information flows deeper through the transformer. "
    "For example: type \"The Eiffel Tower is in the city of\" and see the model converge on Paris layer by layer."
)

prompt = st.text_area(
    "Prompt",
    value="The Eiffel Tower is in the city of",
    height=80,
)

col1, col2, col3 = st.columns(3)
with col1:
    top_k = st.slider("Top-K tokens", min_value=1, max_value=20, value=5)
with col2:
    mode = st.radio("Positions", ["All positions", "Last token only"], horizontal=True)
with col3:
    layer_range = st.slider(
        "Layer range",
        min_value=0,
        max_value=info["num_layers"] - 1,
        value=(0, info["num_layers"] - 1),
    )

layers = list(range(layer_range[0], layer_range[1] + 1))
all_positions = mode == "All positions"

if st.button("Run Logit Lens", type="primary"):
    with st.spinner("Running logit lens analysis..."):
        try:
            result = run_logit_lens(
                wrapper,
                prompt,
                top_k=top_k,
                layers=layers,
                all_positions=all_positions,
            )
            st.session_state["logit_lens_result"] = result
        except Exception as e:
            st.error(f"Logit lens failed: {e}")

# --- Display results ---
result = st.session_state.get("logit_lens_result")
if result is not None:
    token_matrix, prob_matrix = build_logit_lens_matrix(result)

    prob_array = np.array(prob_matrix)
    tokens = result["tokens"]
    result_layers = result["layers"]

    if result.get("all_positions", True):
        x_labels = [f"{i}: {t}" for i, t in enumerate(tokens)]
    else:
        x_labels = [f"{len(tokens)-1}: {tokens[-1]}"]

    y_labels = [f"Layer {l}" for l in result_layers]

    hover_text = []
    for i in range(len(result_layers)):
        row = []
        for j in range(len(x_labels)):
            top_tokens = result["top_k_tokens"][i][j][:5]
            top_probs = result["top_k_probs"][i][j][:5].tolist()
            lines = [f"{t}: {p:.3f}" for t, p in zip(top_tokens, top_probs)]
            row.append(f"Layer {result_layers[i]}<br>Pos {j}<br><br>" + "<br>".join(lines))
        hover_text.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            z=prob_array,
            x=x_labels,
            y=y_labels,
            text=[[token_matrix[i][j] for j in range(len(x_labels))] for i in range(len(result_layers))],
            texttemplate="%{text}",
            textfont_size=9,
            hovertext=hover_text,
            hoverinfo="text",
            colorscale="Viridis",
            colorbar_title="P(top-1)",
        )
    )
    fig.update_layout(
        title="Logit Lens: Top-1 Token Probability by Layer",
        xaxis_title="Token Position",
        yaxis_title="Layer",
        height=max(400, len(result_layers) * 25 + 150),
        yaxis=dict(autorange="reversed"),
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- Per-layer detail ---
    with st.expander("Per-Layer Top-K Details"):
        selected_layer = st.selectbox(
            "Layer",
            result_layers,
            format_func=lambda l: f"Layer {l}",
        )
        layer_pos = result_layers.index(selected_layer)
        num_positions = len(result["top_k_tokens"][layer_pos])

        for pos_idx in range(num_positions):
            pos_label = f"Position {pos_idx}" if all_positions else "Last position"
            tk = result["top_k_tokens"][layer_pos][pos_idx]
            tp = result["top_k_probs"][layer_pos][pos_idx].tolist()
            items = [f"`{t}` ({p:.4f})" for t, p in zip(tk, tp)]
            st.markdown(f"**{pos_label}:** {', '.join(items)}")

    # --- Export ---
    with st.expander("Export"):
        get_download_button(result, "json", "logit_lens_result.json")
