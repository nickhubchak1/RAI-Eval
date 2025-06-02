# === utils/io.py ===

import pandas as pd
import json
import os
import logging


def load_dataset(path: str) -> pd.DataFrame:
    """
    Reads a CSV from `path` into a DataFrame. Logs errors if they occur.
    """
    try:
        df = pd.read_csv(path)
        logging.info(f"[IO] Loaded dataset with {len(df)} rows from {path!r}")
        return df
    except Exception as e:
        logging.error(f"[IO] Failed to load dataset from {path!r}: {e}")
        raise


def save_json(data, path: str):
    """
    Saves a Python object as JSON to `path`. Creates parent directories if needed.
    """
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        logging.info(f"[IO] Saved JSON to {path!r}")
    except Exception as e:
        logging.error(f"[IO] Failed to save JSON to {path!r}: {e}")
        raise
