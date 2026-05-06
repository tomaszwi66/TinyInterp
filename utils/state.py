import streamlit as st

_DEFAULTS = {
    "model": None,
    "model_name": "",
    "model_info": None,
    "quantization": "none",
    "logit_lens_result": None,
    "patching_result": None,
    "ablation_result": None,
    "sae_model": None,
    "sae_features": None,
    "bg_task_id": None,
    "ollama_model": "llama3.2",
    "report_history": [],
}


def init_session_state() -> None:
    for key, default in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default
