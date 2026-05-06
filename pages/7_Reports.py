from datetime import datetime

import streamlit as st

from utils.state import init_session_state

from core.ollama_reporter import check_ollama_available, generate_report

st.header("Scientific Reports")

init_session_state()

st.caption(
    "Generate a structured scientific analysis of your experiment results using a local Ollama model. "
    "The system prompt instructs the LLM to separate observations from hypotheses, "
    "back every claim with numbers from the data, and propose follow-up experiments. "
    "Run at least one analysis on another page first, then come back here."
)

if st.session_state.get("model") is None:
    st.warning("Load a model first on the **Model Loader** page.")
    st.stop()

model_info = st.session_state["model_info"]

# --- Check Ollama ---
available, models = check_ollama_available()

if not available:
    st.error(
        "Ollama is not running or not installed.\n\n"
        "**Setup instructions:**\n"
        "1. Install Ollama from [ollama.com](https://ollama.com)\n"
        "2. Start the Ollama service: `ollama serve`\n"
        "3. Pull a model: `ollama pull llama3.2`\n"
        "4. Reload this page"
    )
    st.stop()

if not models:
    st.warning(
        "Ollama is running but has no models installed. "
        "Run `ollama pull llama3.2` in your terminal, then reload this page."
    )
    st.stop()

# --- Model selection ---
ollama_model = st.selectbox("Ollama Model", models)
if ollama_model:
    st.session_state["ollama_model"] = ollama_model

# --- Analysis selection ---
analysis_options = {}
if st.session_state.get("logit_lens_result") is not None:
    analysis_options["Logit Lens"] = ("logit_lens", st.session_state["logit_lens_result"])
if st.session_state.get("patching_result") is not None:
    analysis_options["Activation Patching"] = ("activation_patching", st.session_state["patching_result"])
if st.session_state.get("ablation_result") is not None:
    analysis_options["Ablation Study"] = ("ablation", st.session_state["ablation_result"])
if st.session_state.get("sae_features") is not None:
    analysis_options["SAE Features"] = ("sae_features", st.session_state["sae_features"])

if not analysis_options:
    st.info("Run at least one analysis (Logit Lens, Patching, Ablation, or SAE) to generate a report.")
    st.stop()

selected_analysis = st.selectbox("Analysis to report on", list(analysis_options.keys()))
analysis_type, results = analysis_options[selected_analysis]

# --- Generate ---
if st.button("Generate Report", type="primary"):
    st.subheader("Report")
    with st.container():
        try:
            stream = generate_report(
                analysis_type=analysis_type,
                results=results,
                model_info=model_info,
                ollama_model=ollama_model,
                stream=True,
            )
            report_text = st.write_stream(stream)

            st.session_state["report_history"].append({
                "analysis_type": selected_analysis,
                "report": report_text,
                "model": ollama_model,
                "timestamp": datetime.now().isoformat(),
            })
        except Exception as e:
            st.error(f"Report generation failed: {e}")

# --- History ---
history = st.session_state.get("report_history", [])
if history:
    st.divider()
    st.subheader("Report History")

    for i, entry in enumerate(reversed(history)):
        with st.expander(
            f"{entry['analysis_type']} - {entry['timestamp'][:19]} ({entry['model']})",
            expanded=(i == 0),
        ):
            st.markdown(entry["report"])
            st.download_button(
                label="Download as Markdown",
                data=entry["report"].encode("utf-8") if isinstance(entry["report"], str) else b"",
                file_name=f"report_{entry['analysis_type']}_{entry['timestamp'][:10]}.md",
                mime="text/markdown",
                key=f"dl_report_{i}",
            )
