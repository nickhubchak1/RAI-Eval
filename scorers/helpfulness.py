# === scorers/helpfulness.py ===

import logging
import openai
import time
from config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY


def score_helpfulness(responses: list[str], max_retries: int = 3) -> list[float]:
    """
    For each response, call OpenAI to rate it on a 1–5 scale. Returns float scores.
    """
    scores = []
    for i, response in enumerate(responses):
        prompt = (
            "Please rate the following response between 1 and 5 for helpfulness, "
            f"clarity, and completeness.\nResponse: {response}"
        )
        retries = 0
        while retries < max_retries:
            try:
                logging.debug(f"[Helpfulness] Prompting OpenAI for idx {i}")
                result = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=8,
                )
                rating_text = result.choices[0].message.content.strip()
                # Parse the first number found
                nums = [s for s in rating_text.split() if s.replace(".", "", 1).isdigit()]
                score = float(nums[0]) if nums else 0.0
                scores.append(score)
                break

            except Exception as e:
                retries += 1
                wait = 2 * retries
                logging.warning(f"[Helpfulness] Retry {retries}, error: {e}")
                time.sleep(wait)

        else:
            logging.error(f"[Helpfulness] Failed to get a rating for idx {i}")
            scores.append(0.0)

    return scores
