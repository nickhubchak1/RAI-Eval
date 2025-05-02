
# RAI Eval

This repository provides a structured evaluation framework for assessing Large Language Models (LLMs) in terms of fairness, robustness, and truthfulness. It includes a composite dataset drawn from multiple sources and supports automated evaluation across different model architectures.

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

## Feeding Prompts to Language Models

Prompts from the combined dataset can be sent to any LLM. Example interface:

```python
for prompt in df['prompt_text']:
    response = call_model(prompt, model="gpt-4")
    evaluate_response(prompt, response)
```

This framework is model-agnostic and can support:
- OpenAI models (e.g., GPT-3.5, GPT-4)
- Anthropic (Claude)
- Open-weight models (LLaMA, Mistral, Falcon)

---

## Repository Structure

- `dataset_generator.py`: Aggregates TruthfulQA, BiasBench, and WikiHow prompts into a single dataset  
- `rai_combined_dataset.csv`: Unified dataset with consistent schema  
- `README.md`: Documentation and evaluation guide  

---

## Future Work

- Expand to include transparency/explainability scoring  
- Integrate multi-turn dialogue evaluation  
- Log performance, latency, and cost per model evaluation  

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
