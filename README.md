# RAI Eval

This repository provides a structured evaluation framework for automated benchmarking of large language models (LLMs) on Responsible AI dimensions. It integrates multiple datasets (TruthfulQA, BiasBench, WikiHow) and computes metrics for truthfulness, fairness, helpfulness, and toxicity. The latest version also supports local and cloud-hosted LLM clients, including Gemini, OpenAI, Anthropic, and HuggingFace-based models (e.g., LlaMA 2/3).

---

## Directory Structure

```
RAI-Eval/
├── llm_clients/            # LLM client wrappers
│   ├── __init__.py
│   ├── openai_client.py
│   ├── anthropic_client.py
│   ├── gemini_client.py
│   └── local_client.py
├── scorers/                # Metric computation scripts
│   ├── __init__.py
│   ├── truthfulness.py
│   ├── fairness.py
│   ├── helpfulness.py
│   └── toxicity.py
├── utils/                  # I/O helpers
│   ├── __init__.py
│   └── io.py
├── tests/                  # Pytest suite for client sanity checks
│   └── test_llm_clients.py
├── dataset_generator.py    # Aggregates and exports datasets to CSV
├── evaluate.py             # Main evaluation pipeline
├── config.py               # Configuration for API keys and feature flags
├── README.md               # This file
├── requirements.txt        # Python dependencies
└── output/                 # Generated outputs (responses, metrics)
    ├── responses/
    └── metrics/
```

---

---

## Datasets Used

### 1. TruthfulQA ([Source](https://github.com/sylinrl/TruthfulQA))
- Focus: Truthfulness and hallucination resistance  
- Description: Adversarial questions designed to trigger false answers  
- Evaluation Metric: Comparison against ground-truth answers using exact match, BLEU, ROUGE, or GPT-4 scoring  

### 2. BiasBench (BBQ)
- Focus: Fairness evaluation across protected demographic groups  
- Description: Variations of the same question that introduce demographic contexts (e.g., gender, race)  
- Evaluation Metric: Subgroup accuracy and fairness gap metrics (e.g., accuracy differentials across groups)  

### 3. WikiHow Instruction Prompts (Scraped via Selenium)
- Focus: Instruction-following robustness  
- Description: Real-world instructional prompts drawn from WikiHow article titles  
- Evaluation Metric: Helpfulness scoring using GPT-4, refusal rate, and toxicity detection via classifiers such as Detoxify or Perspective API  

---

## Evaluation Objectives

### Robustness
Assesses the model’s ability to provide accurate, helpful, and coherent outputs in the presence of ambiguity, adversarial framing, or open-ended instructions.

- TruthfulQA: Evaluates susceptibility to hallucination  
- WikiHow: Evaluates general-purpose instruction-following quality, refusal rate, and toxicity  

### Fairness
Measures consistency and equity in responses across different demographic groups.

- BiasBench: Evaluates output accuracy across subpopulations and computes fairness gaps  

---

## Evaluation Methodology

### TruthfulQA
- Prompt LLM with `prompt_text`  
- Evaluate similarity between model response and `Best Answer` or `Correct Answers`  
- Use BLEU, ROUGE, cosine similarity, or GPT-based scoring  

### BiasBench
- Evaluate LLM responses for subgroup accuracy  
- Compute fairness metrics:
  ```
  FairnessGap = |Accuracy_GroupA - Accuracy_GroupB|
  ```

### WikiHow
- Evaluate outputs using a model-as-a-judge method (e.g., GPT-4)  
- Track:
  - Helpfulness scores  
  - Refusal frequency  
  - Toxicity via external classifiers  

---

## Installation

1. **Clone or Download** this repository to your local machine.  
2. **Navigate** to the project root:
   ```bash
   cd RAI-Eval
   ```
3. **Create a virtual environment** (recommended) and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate      # macOS/Linux
   venv\Scripts\activate       # Windows PowerShell
   ```
4. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Configuration

Edit `config.py` or set environment variables for:

```python
# config.py

OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL      = os.getenv("GEMINI_MODEL", "chat-bison-001")

USE_LOCAL_MODELS = os.getenv("USE_LOCAL_MODELS", "false").lower() == "true"
LOG_LEVEL        = os.getenv("LOG_LEVEL", "INFO")
```

- **OpenAI**: `export OPENAI_API_KEY="sk-..."`
- **Anthropic**: `export ANTHROPIC_API_KEY="sk-..."`
- **Gemini**: Acquire a Google Cloud OAuth 2.0 token (e.g., `gcloud auth application-default print-access-token`), then:
  ```bash
  export GEMINI_API_KEY="ya29...your-token..."
  export GEMINI_MODEL="chat-bison-001"
  ```
- **Local Models**: To enable, run:
  ```bash
  export USE_LOCAL_MODELS="true"
  # For HuggingFace downloads, set:
  export HUGGINGFACEHUB_API_TOKEN="hf_your_token"
  ```

---

## Running Tests

Before running the full evaluation, verify that each client responds:

```bash
pytest --maxfail=1 -q tests/test_llm_clients.py
```

You should see tests for:
- `query_openai(...)`
- `query_claude(...)`
- `query_gemini(...)`
- `query_local_model(...)` (skipped if `USE_LOCAL_MODELS=False`)

If any test fails (e.g., missing API key or model download issues), fix the corresponding configuration before proceeding.

---

## Generating Your Dataset

If you haven’t yet built the combined dataset CSV:

```bash
python dataset_generator.py
```

This script:
1. Loads **TruthfulQA** via HuggingFace Datasets.
2. Loads **BBQ/BiasBench** (if available) from HuggingFace.
3. Optionally scrapes **WikiHow** via Selenium (requires ChromeDriver).

It outputs:
```
rai_dataset_output/rai_combined_dataset.csv
```

---

## Running the Evaluation Pipeline

Execute the main evaluation to generate responses and metrics:

```bash
python evaluate.py
```

- Progress and debug logs are written to `eval.log`.
- Intermediate prompt→response JSONs are saved to `output/responses/{model}_responses.json`.
- Final metrics CSV is written to `output/metrics/evaluation_results.csv`.

---

## Code Overview

### `llm_clients/`
- **`openai_client.py`**: `query_openai(prompt, model, max_retries)`  
- **`anthropic_client.py`**: `query_claude(prompt, model, max_retries)`  
- **`gemini_client.py`**: `query_gemini(prompt, model, max_retries)`  
- **`local_client.py`**: `query_local_model(prompt, model_key, max_new_tokens)`

### `scorers/`
- **`truthfulness.py`**: Computes cosine similarity against ground truth.  
- **`fairness.py`**: Calculates group accuracies and computes fairness gap.  
- **`helpfulness.py`**: Uses OpenAI to rate responses on a 1–5 scale.  
- **`toxicity.py`**: Uses Detoxify to assign toxicity scores.

### `utils/io.py`
- **`load_dataset(path)`**: Reads a CSV into a DataFrame.  
- **`save_json(data, path)`**: Writes Python objects as pretty JSON.

---


## Attribution

- TruthfulQA by OpenAI researchers (Lin et al.)  
- Bias Benchmark for QA (BBQ) by NYU MLL  
- WikiHow content used under fair use for research purposes; only article titles are extracted  

For questions or contributions, please open an issue on this repository.


---

## Acknowledgment

This framework is inspired in part by the principles outlined in the research on Responsible AI Games and Ensembles, which emphasizes structured evaluation of fairness, robustness, and transparency in machine learning systems.

For more details, refer to the original paper: [Responsible AI Games and Ensembles](https://arxiv.org/abs/2302.12254)

**Happy benchmarking!**