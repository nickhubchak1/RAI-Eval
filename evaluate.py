# === evaluate.py ===

import os
import pandas as pd
from utils.io import load_dataset, save_json
from llm_clients.openai_client import query_openai
from llm_clients.anthropic_client import query_claude
from llm_clients.local_client import query_local_model
from scorers.truthfulness import score_truthfulqa
from scorers.fairness import score_biasbench
from scorers.helpfulness import score_helpfulness
from scorers.toxicity import score_toxicity
import logging

logging.basicConfig(filename="eval.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODELS = ["gpt-4", "gpt-3.5-turbo", "claude-3-opus", "llama3", "mistral7b"]

# Cache for responses
RESPONSE_CACHE = {}


def run_evaluation():
    df = load_dataset("rai_dataset_output/rai_combined_dataset.csv")
    results = []

    for model in MODELS:
        logging.info(f"Evaluating model: {model}")
        model_responses = []

        for idx, row in df.iterrows():
            prompt = row["prompt_text"]
            source = row["source_dataset"]
            key = f"{model}_{idx}"

            if key in RESPONSE_CACHE:
                response = RESPONSE_CACHE[key]
            else:
                try:
                    if model.startswith("gpt"):
                        response = query_openai(prompt, model)
                    elif model.startswith("claude"):
                        response = query_claude(prompt)
                    else:
                        response = query_local_model(prompt, model)
                except Exception as e:
                    logging.error(f"Failed on prompt {idx} with model {model}: {e}")
                    response = "ERROR"
                RESPONSE_CACHE[key] = response

            model_responses.append(response)

        df[f"{model}_response"] = model_responses

        # Scoring
        if source == "TruthfulQA":
            df[f"{model}_truth_score"] = score_truthfulqa(df[f"{model}_response"], df["ground_truth"])
        elif source == "BiasBench":
            df[f"{model}_fairness"] = score_biasbench(df, model)
        elif source == "WikiHow":
            df[f"{model}_helpfulness"] = score_helpfulness(df[f"{model}_response"])
            df[f"{model}_toxicity"] = score_toxicity(df[f"{model}_response"])

    save_json(RESPONSE_CACHE, "output/responses/all_responses.json")
    df.to_csv("output/metrics/evaluation_results.csv", index=False)
    logging.info("Evaluation completed successfully.")


if __name__ == "__main__":
    run_evaluation()
