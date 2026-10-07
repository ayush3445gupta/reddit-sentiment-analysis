# Reddit Comment Sentiment Analysis

An NLP project for **3-class sentiment classification** of Reddit comments into **Negative, Neutral, and Positive** categories.

The project systematically compares classical NLP/ML approaches with transformer-based fine-tuning and parameter-efficient fine-tuning using **LoRA**.

## Project Overview

The project was developed as an experimental NLP pipeline rather than a single-model implementation.

Experiments included:

- TF-IDF vs Bag-of-Words representations
- N-gram configurations
- Vocabulary-size experiments
- Class-imbalance handling
- Multiple classical ML classifiers
- Hyperparameter tuning
- Full DistilBERT fine-tuning
- LoRA-based parameter-efficient fine-tuning
- Model comparison on a common locked test set

Experiments for the classical ML stage were tracked using **MLflow**.

## Dataset

The project uses the Reddit portion of the **Twitter and Reddit Sentimental Analysis Dataset**.

After preprocessing and deduplication, approximately **37K Reddit comments** were used for 3-class sentiment classification.

Labels:

| Label | Sentiment |
|---|---|
| -1 | Negative |
| 0 | Neutral |
| 1 | Positive |

A fixed train/test split was created, with **7,305 samples reserved as a locked test set** for final model evaluation.

## Classical NLP Experiments

The classical ML pipeline evaluated:

- TF-IDF vs Bag-of-Words
- Unigrams, bigrams, and trigrams
- Multiple vocabulary sizes
- Logistic Regression
- Linear SVC
- Multinomial Naive Bayes
- Random Forest
- XGBoost
- LightGBM
- KNN
- Class weighting
- RandomOverSampler
- RandomUnderSampler
- SMOTE
- ADASYN
- SMOTEENN and other sampling strategies

Cross-validation was performed using **Stratified K-Fold CV**, with resampling restricted to training folds to avoid evaluation leakage.

### Final Classical Model

The strongest classical pipeline used:

**Bag-of-Words + RandomOverSampler + LightGBM**

| Metric | Result |
|---|---:|
| Accuracy | 92.3% |
| Macro F1 | 0.915 |
| Negative-class F1 | 0.864 |

## DistilBERT Fine-Tuning

`distilbert-base-uncased` was fine-tuned for 3-class sequence classification.

Model checkpoints were selected using **validation Macro F1**, while the locked test set was reserved for final evaluation.

### Final DistilBERT Results

| Metric | Result |
|---|---:|
| Accuracy | **92.7%** |
| Macro Precision | 0.922 |
| Macro Recall | 0.919 |
| Macro F1 | **0.920** |
| Weighted F1 | 0.927 |
| Negative-class F1 | **0.873** |

Full fine-tuned DistilBERT achieved the strongest predictive performance among the evaluated models.

## LoRA Experiments

Parameter-efficient fine-tuning was also evaluated using **LoRA adapters** on DistilBERT.

The experiments varied adapter capacity and training duration.

Best configuration:

- Rank (`r`): 16
- Alpha (`α`): 32
- Epochs: 5
- Target modules: `q_lin`, `v_lin`
- Trainable parameters: approximately **1.31% of total model parameters**

### Best LoRA Result

| Model | Macro F1 |
|---|---:|
| Full DistilBERT Fine-Tuning | **0.920** |
| Best LoRA | **0.849** |
| LightGBM | **0.915** |

The LoRA experiments demonstrate the trade-off between **parameter efficiency and predictive performance**.

## Key Findings

- Bag-of-Words slightly outperformed TF-IDF under the tested classical configurations.
- Unigram features performed better than higher-order n-gram configurations under the tested feature budgets.
- LightGBM provided a highly competitive classical alternative to transformer fine-tuning.
- Full DistilBERT fine-tuning achieved the strongest overall performance.
- Increasing LoRA adapter capacity and training duration substantially improved LoRA performance.
- LoRA trained only a small fraction of model parameters but remained below full fine-tuning in predictive performance.

## Tech Stack

**Python · pandas · NumPy · scikit-learn · LightGBM · XGBoost · PyTorch · Hugging Face Transformers · PEFT/LoRA · MLflow · imbalanced-learn**

## Repository Status

> **Work in progress:** experiment notebooks, preprocessing scripts, training pipelines, and model-evaluation code are currently being organized and will be added to this repository.

## Future Work

- Clean and modularize experiment code
- Add reproducible training scripts
- Package the final inference pipeline
- Build a backend API
- Integrate sentiment predictions with higher-level audience-insight generation

## Author

**Ayush Raj**
