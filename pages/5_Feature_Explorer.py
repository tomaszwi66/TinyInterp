import streamlit as st

from utils.state import init_session_state
import plotly.graph_objects as go
import plotly.express as px
import torch

from core.sae import extract_activations, analyze_sae_features, get_feature_top_tokens
from utils.export import get_download_button

st.header("Feature Explorer")

init_session_state()

if st.session_state.get("model") is None:
    st.warning("Load a model first on the **Model Loader** page.")
    st.stop()

sae = st.session_state.get("sae_model")
if sae is None:
    st.warning("Load or train an SAE first on the **SAE Training** page.")
    st.stop()

wrapper = st.session_state["model"]
info = st.session_state["model_info"]

st.caption(
    "Explore individual SAE features: enter text, encode it through the SAE, and see which features activate. "
    "For each feature you can inspect which tokens trigger it and what vocabulary items it is associated with. "
    "You need to load or train an SAE on the previous page first."
)

# --- Input ---
input_text = st.text_area(
    "Input text",
    value="The Eiffel Tower is in Paris, France.",
    height=80,
)

token_ids = wrapper.tokenizer.encode(input_text)
token_strs = wrapper.tokenizer.convert_ids_to_tokens(token_ids)
default_token_index = next(
    (i for i, token in enumerate(token_strs) if "Paris" in token),
    max(len(token_strs) - 1, 0),
)

col1, col2 = st.columns(2)
with col1:
    target_layer = st.selectbox(
        "Layer (must match SAE training layer)",
        list(range(info["num_layers"])),
        format_func=lambda l: f"Layer {l}",
        index=0,
    )
with col2:
    top_n = st.slider("Top-N features to show", min_value=5, max_value=50, value=20)

target_token_index = st.selectbox(
    "Token to rank features by",
    list(range(len(token_strs))),
    format_func=lambda i: f"{i}: {token_strs[i]}",
    index=default_token_index,
)

if st.button("Encode", type="primary"):
    with st.spinner("Extracting activations and encoding through SAE..."):
        try:
            activations = extract_activations(wrapper, [input_text], target_layer)
            result = analyze_sae_features(sae, activations, top_k=top_n, token_position=target_token_index)
            st.session_state["sae_features"] = result
            st.session_state["sae_activations"] = activations
            st.session_state["sae_input_text"] = input_text
            st.session_state["sae_target_token_index"] = target_token_index
        except Exception as e:
            st.error(f"Encoding failed: {e}")

# --- Display results ---
result = st.session_state.get("sae_features")
if result is not None:
    st.subheader("SAE Analysis Summary")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Active Features", f"{result['num_active_features']:,} / {result['total_features']:,}")
    col_b.metric("Reconstruction MSE", f"{result['reconstruction_mse']:.6f}")
    col_c.metric("Top Features Shown", len(result["top_features"]))

    # Top feature bar chart
    fig = go.Figure(
        data=go.Bar(
            x=[f"F{idx}" for idx in result["top_features"]],
            y=result["top_feature_mean_acts"],
            marker_color="steelblue",
            hovertemplate=f"Feature %{{x}}<br>{result.get('score_label', 'Activation')}: %{{y:.4f}}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Top-{top_n} Features for Selected Token",
        xaxis_title="Feature Index",
        yaxis_title=result.get("score_label", "Activation"),
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- Per-feature exploration ---
    st.subheader("Feature Detail")

    selected_feature = st.selectbox(
        "Select feature",
        result["top_features"],
        format_func=lambda f: f"Feature {f} (mean act: {result['top_feature_mean_acts'][result['top_features'].index(f)]:.4f})",
    )

    if selected_feature is not None:
        feature_idx_in_list = result["top_features"].index(selected_feature)

        # Token-level activation heatmap
        feature_acts = result["feature_activations"][:, selected_feature]
        tokenizer = wrapper.tokenizer
        input_text_stored = st.session_state.get("sae_input_text", input_text)
        token_ids = tokenizer.encode(input_text_stored)
        token_strs = tokenizer.convert_ids_to_tokens(token_ids)

        if len(feature_acts) >= len(token_strs):
            token_acts = feature_acts[: len(token_strs)].tolist()

            fig_heat = go.Figure(
                data=go.Heatmap(
                    z=[token_acts],
                    x=[f"{i}: {t}" for i, t in enumerate(token_strs)],
                    y=[f"Feature {selected_feature}"],
                    colorscale="Hot",
                    hovertemplate="Token: %{x}<br>Activation: %{z:.4f}<extra></extra>",
                )
            )
            fig_heat.update_layout(
                title=f"Feature {selected_feature}: Token-Level Activation",
                xaxis_title="Token Position",
                height=200,
            )
            st.plotly_chart(fig_heat, use_container_width=True)

        # Density
        density = result["feature_densities"][selected_feature].item()
        st.metric(f"Feature {selected_feature} Density", f"{density:.4f}")
        st.caption("Fraction of tokens where this feature activates (> 0).")

        # Decoder weight interpretation
        st.markdown("**Decoder Weight Interpretation** (top tokens by logit)")
        try:
            top_tokens = get_feature_top_tokens(sae, wrapper, selected_feature, top_k=15)
            cols = st.columns(5)
            for i, (tok, logit) in enumerate(top_tokens):
                cols[i % 5].code(f"{tok}: {logit:.2f}")
        except Exception as e:
            st.warning(f"Could not compute decoder interpretation: {e}")

    # --- Export ---
    with st.expander("Export"):
        get_download_button(result, "json", "sae_features.json")
