# === llm_clients/anthropic_client.py ===

import anthropic
import logging
import time
from config import ANTHROPIC_API_KEY

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def query_claude(prompt, model="claude-3-opus-20240229", max_retries=3):
    retries = 0

    while retries < max_retries:
        try:
            logging.info(f"Querying Anthropic model: {model}")
            response = client.messages.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.7
            )
            return response.content[0].text.strip()
        except Exception as e:
            logging.warning(f"Retry {retries+1} for Claude failed: {e}")
            retries += 1
            time.sleep(2 * retries)

    logging.error(f"All retries failed for prompt with Claude {model}")
    return "[ERROR: Claude API failure]"
