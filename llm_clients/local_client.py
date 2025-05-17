# === llm_clients/local_client.py ===

import logging
import requests
import time

# Option A: call a local HTTP endpoint (e.g., vLLM, Text Generation WebUI)
# Option B: use transformers directly if model is loaded in-memory (to be added later)

LOCAL_MODEL_ENDPOINTS = {
    "llama3": "http://localhost:8000/v1/completions",
    "mistral7b": "http://localhost:8001/v1/completions"
}


def query_local_model(prompt, model, max_retries=3):
    url = LOCAL_MODEL_ENDPOINTS.get(model)
    if not url:
        logging.error(f"No endpoint configured for local model: {model}")
        return "[ERROR: Model endpoint not found]"

    payload = {
        "prompt": prompt,
        "max_tokens": 512,
        "temperature": 0.7
    }

    retries = 0
    while retries < max_retries:
        try:
            logging.info(f"Querying local model: {model} at {url}")
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result.get("text", "").strip()
        except Exception as e:
            logging.warning(f"Retry {retries+1} for local model {model} failed: {e}")
            retries += 1
            time.sleep(2 * retries)

    logging.error(f"All retries failed for prompt with local model {model}")
    return f"[ERROR: Local model {model} failed]"

