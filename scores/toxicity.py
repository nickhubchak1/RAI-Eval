# === scorers/toxicity.py ===

import logging
from detoxify import Detoxify

model = Detoxify('original')


def score_toxicity(responses):
    scores = []
    for i, response in enumerate(responses):
        try:
            result = model.predict(response)
            toxicity_score = result.get("toxicity", 0.0)
            scores.append(toxicity_score)
        except Exception as e:
            logging.error(f"Toxicity scoring failed at index {i}: {e}")
            scores.append(0.0)
    return scores
