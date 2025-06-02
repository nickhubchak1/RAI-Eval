# === llm_clients/local_client.py ===

import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from functools import lru_cache
from config import USE_LOCAL_MODELS

# Only allow if USE_LOCAL_MODELS=True in config or env var
# Map a short key (e.g. "llama3") → HuggingFace model ID
HF_MODELS = {
    "llama3":     "meta-llama/Meta-Llama-3-8B",
    "mistral7b":  "mistralai/Mistral-7B-Instruct-v0.2",
     "llama2-7b":     "meta-llama/Llama-2-7b-chat-hf",
    # Add more local keys here if desired
}


@lru_cache(maxsize=2)
def load_model_and_tokenizer(model_key: str):
    if model_key not in HF_MODELS:
        raise ValueError(f"[Local] Model key {model_key!r} not in HF_MODELS.")
    model_name = HF_MODELS[model_key]

    logging.info(f"[Local] Loading model & tokenizer for {model_key!r} → {model_name!r}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    return model, tokenizer


def query_local_model(prompt: str, model_key: str = "llama3", max_new_tokens: int = 200) -> str:
    """
    Generates output from a local HF model. Requires USE_LOCAL_MODELS=True in config.
    """
    if not USE_LOCAL_MODELS:
        logging.error("[Local] USE_LOCAL_MODELS=False in config; cannot query local model.")
        return "[ERROR: Local models disabled]"

    try:
        model, tokenizer = load_model_and_tokenizer(model_key)
        logging.debug(f"[Local] Generating with {model_key!r}")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text.strip()

    except Exception as e:
        logging.error(f"[Local] Error generating with {model_key}: {e}")
        return f"[ERROR: Local model {model_key} failure]"