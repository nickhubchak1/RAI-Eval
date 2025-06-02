# === llm_clients/gemini_client.py ===

import os
import json
import logging
import time
import requests
from config import GEMINI_API_KEY, GEMINI_MODEL

# Base endpoint for Google’s Generative Language API (Vertex AI)
BASE_URL = "https://generativelanguage.googleapis.com/v1beta2/models"


def query_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    max_retries: int = 3,
    temperature: float = 0.7,
    max_output_tokens: int = 512,
) -> str:
    """
    Queries Google’s Gemini (Vertex AI) REST endpoint.
    Requires that GEMINI_API_KEY is a valid Google OAuth 2.0 Bearer token.
    """
    url = f"{BASE_URL}/{model}:generateText"
    headers = {
        "Authorization": f"Bearer {GEMINI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "prompt": {"text": prompt},
        "temperature": temperature,
        "maxOutputTokens": max_output_tokens,
    }

    retries = 0
    while retries < max_retries:
        try:
            logging.debug(f"[Gemini] Attempt {retries+1} → POST {url}")
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            if resp.status_code == 200:
                resp_json = resp.json()
                # Structure: { "candidates": [ { "output": "…" } ], … }
                if "candidates" in resp_json and resp_json["candidates"]:
                    text = resp_json["candidates"][0]["output"].strip()
                    logging.debug(f"[Gemini] Success; got {len(text)} chars")
                    return text
                else:
                    logging.error(f"[Gemini] Unexpected response JSON: {resp_json}")
                    return "[ERROR: Gemini returned no candidates]"
            else:
                logging.warning(f"[Gemini] HTTP {resp.status_code}: {resp.text}")
                retries += 1
                time.sleep(2 * retries)

        except Exception as e:
            retries += 1
            wait = 2 * retries
            logging.warning(f"[Gemini] Retry {retries} after exception: {e}; waiting {wait}s")
            time.sleep(wait)

    logging.error(f"[Gemini] All {max_retries} retries failed.")
    return "[ERROR: Gemini API failure]"
