# === llm_clients/openai_client.py ===

import openai
import logging
import time
from config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY


def query_openai(prompt: str, model: str = "gpt-4", max_retries: int = 3) -> str:
    """
    Sends a single-user message prompt to OpenAI’s ChatCompletion endpoint.
    Retries up to `max_retries` times on transient errors.
    Returns the assistant’s reply as plain text, or an error string if it fails.
    """
    messages = [{"role": "user", "content": prompt}]
    retries = 0

    while retries < max_retries:
        try:
            logging.debug(f"[OpenAI] Attempt {retries + 1} using model {model!r}")
            response = openai.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=512,
            )
            # In openai>=1.0.0, the structure is:
            # response.choices[0].message.content
            text = response.choices[0].message.content.strip()
            logging.debug(f"[OpenAI] Received response: {text[:50]}…")
            return text

        except Exception as e:
            retries += 1
            wait = 2 * retries
            logging.warning(f"[OpenAI] Retry {retries} after error: {e}; waiting {wait}s")
            time.sleep(wait)

    logging.error(f"[OpenAI] All {max_retries} retries failed for prompt.")
    return "[ERROR: OpenAI API failure]"
