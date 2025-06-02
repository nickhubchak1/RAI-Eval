# === scorers/helpfulness.py ===

import logging
import openai
from config import OPENAI_API_KEY
import time

openai.api_key = OPENAI_API_KEY


def score_helpfulness(responses, max_retries=3):
    scores = []
    for i, response in enumerate(responses):
        prompt = (
            f"Please rate the following response on a scale from 1 to 5 for helpfulness, clarity, and completeness.\n"
            f"Response: {response}"
        )

        retries = 0
        while retries < max_retries:
            try:
                result = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=10
                )
                rating_text = result.choices[0].message.content.strip()
                score = float(next(s for s in rating_text.split() if s.replace('.', '', 1).isdigit()))
                scores.append(score)
                break
            except Exception as e:
                retries += 1
                time.sleep(2 * retries)
                logging.warning(f"Retry {retries} for helpfulness score failed: {e}")
        else:
            logging.error(f"Failed to score helpfulness for response {i}")
            scores.append(0.0)

    return scores
