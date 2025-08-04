
# mFARM: Multi-Faceted Fairness Assessment Based on HARMs in Clinical LLMs

This repository contains all the code, datasets, and evaluation scripts accompanying the paper:

> **mFARM: Towards Multi-Faceted Fairness Assessment based on HARMs in Clinical Decision Support**  

---

## Overview

The **mFARM framework** introduces a rigorous, multi-dimensional fairness evaluation suite for auditing LLMs in clinical decision-making tasks. We provide:

- Two large-scale, demographically-augmented benchmarks derived from MIMIC-IV:
  - **ED Triage Prediction**
  - **Opioid Analgesic Recommendation**
- A comprehensive fairness evaluation framework based on:
  - Allocational, Stability, and Latent Harms
  - The composite **mFARM Score** and **FAB Score**
- Evaluation pipelines for both **base** and **LoRA fine-tuned** open-source LLMs across:
  - Context tiers (High/Medium/Low)
  - Quantization levels (16-bit/8-bit/4-bit)

---

## Repository Structure

```
.
├── Dataset_Creation_And_Finetuning/
│   ├── ED_Triage_Prediction/
│   │   ├── 1_make_raw_dataset.ipynb
│   │   ├── High_Context_Data/
│   │   │   └── 1_make_pruned_prompts_from_original_dataset.ipynb
│   │   ├── Medium_Context/
│   │   ├── Low_Context/
│   │   ├── evaluate_llm_new.py
│   │   └── evaluate_llm_new_finetuned.py
│   ├── Opioid_Analgesics_Prediction/
│   │   ├── 1_make_raw_dataset.ipynb
│   │   ├── High_Context/
│   │   ├── Medium_Context/
│   │   ├── Low_Context/
│   │   ├── evaluate_llm_new.py
│   │   └── evaluate_llm_new_finetuned.py
├── Metric_Computation/
│   └── metric_funcs_copy_latest_modified.py
└── README.md
```

---

##  Getting Started

### 1. Access MIMIC-IV

This repo assumes access to [MIMIC-IV v3.1](https://physionet.org/content/mimiciv/1.0/). Please ensure:
- You have completed the necessary credentialing.
- You store the MIMIC-IV data locally in a structure compatible with the notebooks.

---

### 2. Environment Setup

Install dependencies (Python 3.10+ recommended):

```bash
pip install -r requirements.txt  # (Assuming you add this)
```

You'll need:
- `pandas`, `numpy`, `scikit-learn`
- `transformers`, `peft`, `datasets`, `evaluate`
- `matplotlib`, `seaborn`
- `torch`, `accelerate`

---

##  Benchmark Tasks

### A. ED Triage Prediction

- Predict whether a patient should receive **urgent care**.
- Built using vitals + ESI scores from MIMIC-IV.
- Context variants:
  - **High**: Full clinical context
  - **Medium**: Summary + vitals
  - **Low**: Only chief complaint + age

Navigate to:
```
Dataset_Creation_And_Finetuning/ED_Triage_Prediction/
```

#### Steps:

1. **Build Dataset**:
   ```bash
   Run: 1_make_raw_dataset.ipynb
   ```

2. **Generate Prompts**:
   ```bash
   Run the appropriate notebook in High_Context_Data/, Medium_Context/, or Low_Context/
   ```

3. **Evaluate LLMs**:
   - Use `evaluate_llm_new.py` (for base models)
   - Use `evaluate_llm_new_finetuned.py` (for LoRA-finetuned models)

---

### B. Opioid Analgesic Recommendation

- Predict whether a patient should be prescribed opioid painkillers.
- Based on hospital notes + prescription data.

 Navigate to:
```
Dataset_Creation_And_Finetuning/Opioid_Analgesics_Prediction/
```

#### Steps:

1. **Build Dataset**:
   ```bash
   Run: 1_make_raw_dataset.ipynb
   ```

2. **Generate Prompt Splits**:
   ```bash
   Run: 2_make_final_dataset_splits_with_all_prompts.ipynb
   ```

3. **Evaluate Models**:
   - Use `evaluate_llm_new.py` or `evaluate_llm_new_finetuned.py`

---

## Fairness Evaluation

Go to:
```
Metric_Computation/
```

Run:
```bash
python metric_funcs_copy_latest_modified.py
```

This computes:

| Harm Type       | Metric                     |
|----------------|----------------------------|
| Allocational    | Mean Difference            |
| Stability       | Absolute Deviation, Variance Heterogeneity |
| Latent          | KS Distribution, Correlation Difference |
| Composite       | Geometric Mean (mFARM)     |
| Combined        | Harmonic Mean with Accuracy (FAB Score) |

All metrics return scores ∈ [0, 1], where **1 = perfect fairness**.

---

## Models Evaluated

The following open-source models were evaluated in both base and LoRA fine-tuned forms:

- **Mistral-7B**
- **BioMistral-7B**
- **Qwen-2.5-7B**
- **BioLLaMA3-8B**

Evaluations are repeated across:

- **Quantization Levels**: 16-bit, 8-bit, 4-bit
- **Context Tiers**: High, Medium, Low

---

## Results Summary

| Task | Model | mFARM (High Ctx) | FAB Score (High Ctx) |
|------|--------|------------------|------------------------|
| ED Triage | Mistral-ft | 0.916 | 0.702 |
| Opioid Rx | Mistral-ft | 0.899 | 0.875 |
| ED Triage | BioMistral | 0.474 | 0.492 |

> LoRA-finetuning consistently improves accuracy and often fairness. See paper for full comparison across all metrics.

---
