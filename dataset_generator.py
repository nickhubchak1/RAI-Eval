# === dataset_generator.py ===

import os
import time
import json
import pandas as pd
from datasets import load_dataset
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_truthfulqa() -> pd.DataFrame:
    """
    Load the TruthfulQA dataset via HuggingFace Datasets.
    Returns a DataFrame with columns: ["question", "ground_truth", "source_dataset"].
    """
    logging.info("[DatasetGen] Loading TruthfulQA from HuggingFace.")
    ds = load_dataset("truthful_qa", "mc1", split="validation")
    # The HF split might have fields like 'question', 'correct_answer'
    df = pd.DataFrame({
        "question":          ds["question"],
        "ground_truth":      ds["correct_answer"],
        "subpopulation":     "n/a",
        "source_dataset":    "TruthfulQA"
    })
    return df


def load_biasbench() -> pd.DataFrame:
    """
    Load the BBQ (Bias Benchmark for QA) dataset via HuggingFace, focusing on fairness.
    Returns a DataFrame with columns: ["question", "ground_truth", "subpopulation", "source_dataset"].
    """
    logging.info("[DatasetGen] Loading BBQ (BiasBench) from HuggingFace.")
    ds = load_dataset("bbq", split="test")  # Adjust if needed
    # Assume 'question', 'answer', and 'group_name' fields exist
    df = pd.DataFrame({
        "question":          ds["question"]["stem"],
        "ground_truth":      ds["answer"]["text"][0],  # pick first correct text
        "subpopulation":     ds["group_name"],
        "source_dataset":    "BiasBench"
    })
    return df


def load_wikihow_with_selenium(num_prompts: int = 100) -> pd.DataFrame:
    """
    Scrape the first `num_prompts` article titles from WikiHow via Selenium.
    Returns DataFrame with columns: ["question", "ground_truth", "subpopulation", "source_dataset"].
    (Here we treat a WikiHow “title” as a “question”; ground_truth = same as question.)
    """
    logging.info("[DatasetGen] Launching Selenium to scrape WikiHow titles.")
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(options=chrome_options)

    driver.get("https://www.wikihow.com/wikiHowTo?find=search&search=")
    time.sleep(3)

    titles = []
    try:
        items = driver.find_elements(By.CSS_SELECTOR, ".result_link")
        for idx, el in enumerate(items):
            if idx >= num_prompts:
                break
            titles.append(el.text.strip())
    except Exception as e:
        logging.error(f"[DatasetGen] Selenium error: {e}")
    finally:
        driver.quit()

    df = pd.DataFrame({
        "question":       titles,
        "ground_truth":   titles,
        "subpopulation":  "n/a",
        "source_dataset": "WikiHow"
    })
    return df


def main():
    """
    Auto-aggregate all three datasets (TruthfulQA, BiasBench, WikiHow) and save combined CSV.
    """
    logging.info("[DatasetGen] Starting dataset aggregation…")
    df_truthful = load_truthfulqa()
    df_bias     = load_biasbench()
    df_wikihow  = load_wikihow_with_selenium(num_prompts=100)

    full_df = pd.concat([df_truthful, df_bias, df_wikihow], ignore_index=True)
    logging.info(f"[DatasetGen] Total aggregated prompts: {len(full_df)}")

    output_dir = "rai_dataset_output"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "rai_combined_dataset.csv")
    full_df.to_csv(out_path, index=False)
    logging.info(f"[DatasetGen] Saved combined dataset to {out_path}")


if __name__ == "__main__":
    main()
