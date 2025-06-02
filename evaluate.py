# === evaluate.py ===

import os
import sys
import time
import logging
import pandas as pd

from utils.io import load_dataset, save_json
from llm_clients.openai_client   import query_openai
from llm_clients.anthropic_client import query_claude
from llm_clients.local_client    import query_local_model
from llm_clients.gemini_client   import query_gemini

from scorers.truthfulness import score_truthfulqa
from scorers.fairness     import score_biasbench
from scorers.helpfulness  import score_helpfulness
from scorers.toxicity     import score_toxicity

# ---------------- Logging Setup ----------------

LOG_DIR = "eval.log"
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

# -------------- Main Evaluation Flow --------------

def run_evaluation():
    try:
        start_time = time.time()
        logging.info(">>> Logging is active. Starting evaluation pipeline…")

        # 1) Load the combined dataset (or any CSV with columns: prompt_text, ground_truth, subpopulation, source_dataset)
        DATA_CSV    = os.getenv("RAI_DATASET_PATH", "rai_dataset_output/rai_combined_dataset.csv")
        OUTPUT_BASE = "output"

        df = load_dataset(DATA_CSV)
        logging.info(f"[Eval] Loaded {len(df)} rows. Columns: {list(df.columns)}")

        # 2) Prepare output folders
        RESP_DIR    = os.path.join(OUTPUT_BASE, "responses")
        METRICS_DIR = os.path.join(OUTPUT_BASE, "metrics")
        os.makedirs(RESP_DIR, exist_ok=True)
        os.makedirs(METRICS_DIR, exist_ok=True)

        # 3) Decide which LLMs to call
        models_to_run = {
            "openai-gpt4":      lambda prompt: query_openai(prompt, model="gpt-4"),
            "openai-gpt3.5":    lambda prompt: query_openai(prompt, model="gpt-3.5-turbo"),
            "anthropic-claude": lambda prompt: query_claude(prompt, model="claude-3-opus-20240229"),
            "gemini":           lambda prompt: query_gemini(prompt, model=None),
        }

        from config import USE_LOCAL_MODELS
        if USE_LOCAL_MODELS:
            models_to_run["local-llama3"]    = lambda prompt: query_local_model(prompt, model_key="llama3")
            models_to_run["local-mistral7b"] = lambda prompt: query_local_model(prompt, model_key="mistral7b")
            # If you added llama2-7b or others, include them here:
            models_to_run["local-llama2-7b"]  = lambda prompt: query_local_model(prompt, model_key="llama2-7b")

        logging.info(f"[Eval] Models to run: {list(models_to_run.keys())}")

        # 4) Loop over each model, generate responses for all prompts, store in new columns
        RESPONSE_CACHE = {}
        for model_name, query_fn in models_to_run.items():
            resp_col = f"{model_name}_response"
            logging.info(f"[Eval] Starting generation with '{model_name}'")
            df[resp_col] = ""  # initialize empty column

            for idx, row in df.iterrows():
                # Use "prompt_text" if available, otherwise fallback to "question"
                if "prompt_text" in df.columns:
                    prompt = row["prompt_text"]
                else:
                    prompt = row.get("question", "")
                    if prompt == "":
                        logging.error(f"[Eval] Row {idx} has no 'prompt_text' or 'question' column.")
                        sys.exit(1)

                cache_key = f"{model_name}||{prompt}"
                if cache_key in RESPONSE_CACHE:
                    response = RESPONSE_CACHE[cache_key]
                else:
                    response = query_fn(prompt)
                    RESPONSE_CACHE[cache_key] = response

                df.at[idx, resp_col] = response

                if idx > 0 and idx % 20 == 0:
                    logging.info(f"[Eval][{model_name}] Generated {idx} / {len(df)} prompts")

            # Save partial JSON so you can resume if needed
            save_json(RESPONSE_CACHE, os.path.join(RESP_DIR, f"{model_name}_responses.json"))

        # 5) Scoring Phase: compute a column for each metric per model
        logging.info("[Eval] Starting scoring of all metrics…")

        # 5a) Truthfulness
        for model_name in models_to_run.keys():
            resp_col  = f"{model_name}_response"
            truth_col = f"{model_name}_truthscore"
            try:
                scores = score_truthfulqa(
                    responses=df[resp_col].tolist(),
                    ground_truths=df["ground_truth"].tolist()
                )
                df[truth_col] = scores
                logging.info(f"[Eval] Completed Truthfulness scoring for {model_name}")
            except Exception as e:
                logging.warning(f"[Eval] Truthfulness scoring failed for {model_name}: {e}")
                df[truth_col] = [0.0] * len(df)

        # 5b) Fairness (BiasBench)
        for model_name in models_to_run.keys():
            resp_col = f"{model_name}_response"
            fair_col = f"{model_name}_fairness_gap"
            try:
                df_for_bias = df.copy()
                df_for_bias[f"{model_name}_response"] = df[resp_col]
                # Ensure these columns exist:
                df_for_bias["ground_truth"]  = df["ground_truth"]
                df_for_bias["subpopulation"] = df["subpopulation"]
                gap_list = score_biasbench(df_for_bias, model_column_prefix=model_name)
                df[fair_col] = gap_list
                logging.info(f"[Eval] Completed Fairness scoring for {model_name}")
            except Exception as e:
                logging.warning(f"[Eval] Fairness scoring failed for {model_name}: {e}")
                df[fair_col] = [0.0] * len(df)

        # 5c) Helpfulness & Toxicity (WikiHow only)
        wh_mask = df["source_dataset"] == "WikiHow"
        for model_name in models_to_run.keys():
            resp_col = f"{model_name}_response"
            help_col = f"{model_name}_helpfulness"
            tox_col  = f"{model_name}_toxicity"
            try:
                wh_responses = df.loc[wh_mask, resp_col].tolist()
                helps = score_helpfulness(wh_responses)
                toxis = score_toxicity(wh_responses)
                df.loc[wh_mask, help_col]  = helps
                df.loc[wh_mask, tox_col]   = toxis
                df.loc[~wh_mask, help_col] = None
                df.loc[~wh_mask, tox_col]  = None
                logging.info(f"[Eval] Completed Helpfulness & Toxicity for {model_name}")
            except Exception as e:
                logging.warning(f"[Eval] Helpfulness/Toxicity scoring failed for {model_name}: {e}")
                df[help_col] = [None] * len(df)
                df[tox_col]  = [None] * len(df)

        # 6) Save final CSV and full JSON caches
        final_csv = os.path.join(METRICS_DIR, "evaluation_results.csv")
        df.to_csv(final_csv, index=False)
        logging.info(f"[Eval] Saved final results as {final_csv}")

        end_time = time.time()
        elapsed = end_time - start_time
        logging.info(f"[Eval] Done! Total time: {elapsed:.1f}s")
        print("✓ Evaluation complete. See eval.log and output/metrics for results.")

    except Exception as e:
        logging.critical(f"[Eval] Pipeline crashed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    run_evaluation()
