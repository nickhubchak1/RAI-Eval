# === llm_clients/anthropic_client.py ===

import anthropic
import logging
import time
from config import ANTHROPIC_API_KEY

client = anthropic.Client(api_key=ANTHROPIC_API_KEY)


def query_claude(prompt: str, model: str = "claude-3-opus-20240229", max_retries: int = 3) -> str:
    """
    Sends a user prompt to Anthropic’s Claude endpoint.
    Retries on errors, returns "[ERROR: Claude API failure]" if all tries fail.
    """
    retries = 0
    while retries < max_retries:
        try:
            logging.debug(f"[Anthropic] Attempt {retries + 1} using model {model!r}")
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens_to_sample=512,
                temperature=0.7,
            )
            # The response object structure may vary by SDK version; this is one common pattern:
            text = response.choices[0].message["content"].strip()
            logging.debug(f"[Anthropic] Received response: {text[:50]}…")
            return text

        except Exception as e:
            retries += 1
            wait = 2 * retries
            logging.warning(f"[Anthropic] Retry {retries} after error: {e}; waiting {wait}s")
            time.sleep(wait)

    logging.error(f"[Anthropic] All {max_retries} retries failed for prompt.")
    return "[ERROR: Claude API failure]"
