from pathlib import Path

APP_DIR = Path(__file__).parent
CACHE_DIR = APP_DIR / ".cache"
RESULTS_DIR = APP_DIR / "results"
CHECKPOINTS_DIR = APP_DIR / "checkpoints"

CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
CHECKPOINTS_DIR.mkdir(exist_ok=True)

DEFAULT_MODELS = [
    "openai-community/gpt2",
    "openai-community/gpt2-medium",
    "openai-community/gpt2-large",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    "google/gemma-2-2b",
    "microsoft/phi-2",
]

QUANT_OPTIONS = {
    "None (FP16/FP32)": "none",
    "8-bit (bitsandbytes)": "8bit",
    "4-bit (bitsandbytes)": "4bit",
}

SAE_DEFAULT_CONFIG = {
    "expansion_factor": 16,
    "training_tokens": 10_000_000,
    "batch_size": 4096,
    "lr": 5e-5,
    "l1_coefficient": 5.0,
    "context_size": 256,
}

PRETRAINED_SAE_RELEASES = {
    "openai-community/gpt2": [
        {"release": "gpt2-small-res-jb", "layers": list(range(12))},
    ],
    "google/gemma-2-2b": [
        {"release": "gemma-scope-2b-pt-res-canonical", "layers": list(range(26))},
    ],
}

OLLAMA_SYSTEM_PROMPT = """You are a mechanistic interpretability research assistant analyzing transformer neural networks. You write in precise scientific language suitable for ML research papers.

When given experimental results (logit lens outputs, activation patching matrices, ablation effects, or SAE feature activations), you must:

1. DESCRIBE the data: State what was measured, on which model, at which layers/positions, and with what inputs.
2. IDENTIFY patterns: Note which layers show sharp transitions in prediction, which components have high causal effect, which features activate strongly, and any unexpected behaviors.
3. INTERPRET mechanistically: Propose what computational role each layer/head/feature plays. Use standard terminology: "induction heads", "name mover heads", "factual recall", "residual stream", "superposition", etc.
4. QUANTIFY claims: Reference specific numerical values from the data. Say "Layer 8 accounts for 73% of the logit difference" not "Layer 8 is important".
5. NOTE limitations: State what the evidence does NOT show, what alternative explanations exist, and what follow-up experiments would be informative.

Format your response with markdown headers. Include a one-paragraph executive summary at the top. Do not speculate beyond what the data supports. If the results are ambiguous, say so explicitly."""

OLLAMA_DEFAULT_MODEL = "llama3.2"
