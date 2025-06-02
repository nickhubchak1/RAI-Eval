# === explainers/explainability.py ===

import os
import json
import torch
import shap
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from lime.lime_text import LimeTextExplainer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# We assume any local model lives under HF_MODELS in llm_clients/local_client.py.
# This helper is specific to local HuggingFace causal‐LM models.


def explain_shap(
    model_key: str,
    hf_tokenizer,
    hf_model,
    prompt: str,
    explainer_dir: str,
    max_tokens: int = 128,
):
    """
    Generate SHAP explanations for a local HF causal LM. We treat the model as a text classifier by
    measuring the change in log‐probability for each token in the generated response when we mask input tokens.
    For simplicity, we'll explain the model's prediction of the first token of its response.
    In practice you can expand to multiple tokens or a custom scalar metric (e.g. toxicity).
    """
    os.makedirs(explainer_dir, exist_ok=True)
    device = hf_model.device

    # 1) Tokenize prompt
    inputs = hf_tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"][0]  # shape: (seq_len,)

    # 2) Generate a single‐token prediction (for explainability). 
    #    In practice, you might want to run full generate() and pick the top output token.
    with torch.no_grad():
        outputs = hf_model(**inputs)
        # logits shape: (1, seq_len, vocab_size). We take the last position.
        next_token_logits = outputs.logits[0, -1, :]
        topk = torch.topk(next_token_logits, k=1)
        chosen_token_id = topk.indices.item()
        # NB: this is just an example metric. You can also pick e.g. token‐level
        #     probability of a “correct” word, or a custom function on the final text.

    # 3) Define a function f(masked_texts) that returns the log‐prob of chosen_token_id:
    def f_texts(text_list):
        """Given a list of perturbed prompts, return the scalar log‐prob of the chosen_token."""
        scores = []
        for txt in text_list:
            ins = hf_tokenizer(txt, return_tensors="pt", truncation=True, max_length=512).to(device)
            with torch.no_grad():
                out = hf_model(**ins)
                logprob = torch.log_softmax(out.logits[0, -1, :], dim=-1)[chosen_token_id].item()
                scores.append(logprob)
        return np.array(scores)

    # 4) Instantiate SHAP explainer for text
    #    We use shap.Explainer with a masker that splits on whitespace.
    masker = shap.maskers.Text(tokenizer=hf_tokenizer, mask_start='[MASK]')
    shap_explainer = shap.Explainer(f_texts, masker, output_names=[hf_tokenizer.decode([chosen_token_id])])

    # 5) Compute SHAP values on the prompt
    shap_values = shap_explainer(prompt, max_evals=200)
    # This yields a shap.Explanation object

    # 6) Save SHAP values to JSON and plot bar chart of top tokens
    base_filename = os.path.join(explainer_dir, f"{model_key}_shap_{abs(hash(prompt))}.json")
    with open(base_filename, "w") as f:
        json.dump({
            "data": shap_values.data.tolist(),
            "values": shap_values.values.tolist(),
            "tokens": shap_values.data.tolist(),
        }, f)

    # 7) Plot summary bar chart (absolute SHAP values)
    plt.figure(figsize=(8, 4))
    shap.summary_plot(
        shap_values.values, np.array(shap_values.data),
        show=False, feature_names=shap_values.data
    )
    plt.title("SHAP Explanation (Top Tokens)")
    plot_path = base_filename.replace(".json", ".png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    return (base_filename, plot_path)


def explain_lime(
    model_key: str,
    hf_tokenizer,
    hf_model,
    prompt: str,
    explainer_dir: str,
    num_samples: int = 1000,
    num_features: int = 10,
):
    """
    Generate a LIME explanation for the local HF model. We again treat the
    model as a function f(txt) → log-probability of the final chosen_token_id.
    LimeTextExplainer perturbs the input text by removing tokens randomly,
    so we only keep a bag-of-words representation for the surrogate.
    """
    os.makedirs(explainer_dir, exist_ok=True)
    device = hf_model.device

    # 1) Find chosen_token_id the same way as in SHAP:
    inputs = hf_tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = hf_model(**inputs)
        chosen_token_id = torch.topk(out.logits[0, -1], k=1).indices.item()

    # 2) Define a wrapper f(txt) → log‐prob of chosen_token_id
    def f_wrapper(texts):
        scores = []
        for txt in texts:
            tok = hf_tokenizer(txt, return_tensors="pt", truncation=True, max_length=512).to(device)
            with torch.no_grad():
                o = hf_model(**tok)
                lp = torch.log_softmax(o.logits[0, -1, :], dim=-1)[chosen_token_id].item()
                scores.append(lp)
        return np.array(scores)

    # 3) Instantiate LIME explainer
    class_names = [hf_tokenizer.decode([chosen_token_id])]
    lime_explainer = LimeTextExplainer(class_names=class_names)

    # 4) Explain instance
    exp = lime_explainer.explain_instance(
        prompt,
        f_wrapper,
        num_samples=num_samples,
        num_features=num_features,
        top_labels=1
    )

    # 5) Save explanation as HTML + JSON
    base_name = os.path.join(explainer_dir, f"{model_key}_lime_{abs(hash(prompt))}")
    html_path = base_name + ".html"
    json_path = base_name + ".json"
    exp.save_to_file(html_path)

    # Extract list of (token, weight) pairs
    token_weights = exp.as_list(label=0)
    with open(json_path, "w") as f:
        json.dump({"explanation": token_weights}, f)

    return (json_path, html_path)


def get_attention_heatmap(
    model_key: str,
    hf_tokenizer,
    hf_model,
    prompt: str,
    explainer_dir: str,
    layer: int = -1,     # last layer by default
    head: int = 0,       # first head by default
):
    """
    Extract attention weights from a transformer-based HF model. We pass `output_attentions=True`
    so that the forward pass returns a tuple including attentions. We then select the desired layer
    and head, and plot a heatmap of attention weights between input tokens and themselves.
    """
    os.makedirs(explainer_dir, exist_ok=True)
    device = hf_model.device

    # 1) Tokenize with return_tensors and request attentions
    inputs = hf_tokenizer(prompt, return_tensors="pt").to(device)
    # Move the model to output attentions
    hf_model.config.output_attentions = True

    with torch.no_grad():
        outputs = hf_model(**inputs)
        # outputs.attentions is a tuple: (layer1_attention, layer2_attention, ...)
        # each layer attention shape: (batch_size, num_heads, seq_len, seq_len)
        attentions = outputs.attentions

    # 2) Pick the specified layer and head
    attn_matrix = attentions[layer][0, head].cpu().numpy()  # shape: (seq_len, seq_len)

    # 3) Plot a heatmap (token-to-token)
    tokens = hf_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(attn_matrix, xticklabels=tokens, yticklabels=tokens, cmap="viridis")
    ax.set_title(f"Attention Heatmap: Layer {layer}, Head {head}")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    heatmap_path = os.path.join(explainer_dir, f"{model_key}_attn_heatmap_{abs(hash(prompt))}.png")
    plt.savefig(heatmap_path, bbox_inches='tight')
    plt.close()

    return heatmap_path
