# === utils/io.py ===

import pandas as pd
import json
import os
import logging


def load_dataset(path):
    try:
        df = pd.read_csv(path)
        logging.info(f"Loaded dataset with {len(df)} rows from {path}")
        return df
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        raise


def save_json(data, path):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        logging.info(f"Saved JSON output to {path}")
    except Exception as e:
        logging.error(f"Failed to save JSON file {path}: {e}")
        raise
