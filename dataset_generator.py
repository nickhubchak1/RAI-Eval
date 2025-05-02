# === Auto-Aggregate RAI Dataset Script ===

import pandas as pd
import os
import json
from datasets import load_dataset
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By


# Create output folder if it doesn't exist
os.makedirs("rai_dataset_output", exist_ok=True)

# Helper function to load TruthfulQA
def load_truthfulqa():
    print("Loading TruthfulQA...")
    truthfulqa = pd.read_csv("https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv")
    df = pd.DataFrame({
        'prompt_id': [f"TQA-{i}" for i in range(len(truthfulqa))],
        'prompt_text': truthfulqa['Question'],
        'source_dataset': 'TruthfulQA',
        'adversarial_category': 'Adversarial',
        'difficulty': 'N/A',
        'topic': truthfulqa['Category'],
        'prompt_type': 'question',
        'subpopulation': '',
        'ground_truth': truthfulqa['Best Answer'],
        'toxicity_tag': 'No',
        'fairness_tag': 'No',
        'hallucination_risk': 'Yes'
    })
    return df

# Helper function to load BiasBench (BBQ)
# Helper function to load BiasBench (BBQ) if cloned locally
def load_biasbench():
    print("Loading BiasBench (BBQ) from local clone...")
    all_dfs = []
    bias_categories = [
        "age", "disability_status", "gender_identity",
        "nationality", "physical_appearance", "race_ethnicity",
        "religion", "ses", "sexual_orientation"
    ]
    for cat in bias_categories:
        filepath = os.path.join("BBQ", "data", f"{cat}.jsonl")
        df = pd.read_json(filepath, lines=True)
        all_dfs.append(df)
    
    bbq = pd.concat(all_dfs, ignore_index=True)
    
    df = pd.DataFrame({
        'prompt_id': [f"BBQ-{i}" for i in range(len(bbq))],
        'prompt_text': bbq['question'],
        'source_dataset': 'BiasBench',
        'adversarial_category': bbq['context_condition'],
        'difficulty': 'N/A',
        'topic': bbq['category'],
        'prompt_type': 'question',
        'subpopulation': bbq['category'],
        'ground_truth': bbq['label'],
        'toxicity_tag': 'No',
        'fairness_tag': 'Yes',
        'hallucination_risk': 'No'
    })
    return df


# Helper function to load AlpacaEval
def load_wikihow_with_selenium(num_prompts=100):
    print("Scraping WikiHow with Selenium...")

    options = Options()
    options.add_argument("--headless")  # run in background
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-software-rasterizer")  # <== key flag
    options.add_argument("--disable-dev-shm-usage")        # safer for Linux too
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--headless=new")

    driver = webdriver.Chrome(options=options)

    instructions = set()

    while len(instructions) < num_prompts:
        try:
            driver.get("https://www.wikihow.com/Special:Randomizer")
            time.sleep(2)  # let page load

            title = driver.title
            if title.lower().startswith("how to") and title not in instructions:
                instructions.add(title.strip())

        except Exception as e:
            print(f"Error: {e}")
            break

    driver.quit()
    print(f"✓ Collected {len(instructions)} prompts.")

    df = pd.DataFrame({
        'prompt_id': [f"WIKI-{i}" for i in range(len(instructions))],
        'prompt_text': list(instructions),
        'source_dataset': 'WikiHow',
        'adversarial_category': '',
        'difficulty': 'N/A',
        'topic': '',
        'prompt_type': 'instruction',
        'subpopulation': '',
        'ground_truth': '',
        'toxicity_tag': 'No',
        'fairness_tag': 'No',
        'hallucination_risk': 'No'
    })

    return df

# Aggregate everything
def main():
    print("Aggregating datasets...")
    df_truthfulqa = load_truthfulqa()
    df_biasbench = load_biasbench()
    df_wikihow = load_wikihow_with_selenium(num_prompts=100)

    full_df = pd.concat([df_truthfulqa, df_biasbench, df_wikihow], ignore_index=True)

    print(f"Total prompts: {len(full_df)}")
    output_path = os.path.join("rai_dataset_output", "rai_combined_dataset.csv")
    full_df.to_csv(output_path, index=False)
    print(f"Saved combined dataset to {output_path}")

if __name__ == "__main__":
    main()
