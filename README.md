# ML Engineer Practical Assessment – Fraud Detection

## Problem Statement
Build a production-grade machine learning pipeline for fraud detection
using credit card transaction data.

---

## Dataset
Credit Card Fraud Detection dataset.
Highly imbalanced dataset with ~0.17% fraud cases.

---

## Project Structure
- data/raw: original dataset
- data/processed: cleaned data
- src: modular source code
- models: saved trained models

---

## ML Pipeline Steps

### 1. Data Validation & Cleaning
- Checked missing values and duplicates
- Removed duplicate rows
- Analyzed class imbalance

### 2. Feature Engineering
Engineered meaningful features:
- Amount_log: log transformation of transaction amount
- Amount_squared: captures high-value transactions
- Hour: transaction hour extracted from time
- Is_night: flag for night transactions

### 3. Model Selection
RandomForestClassifier chosen due to:
- Ability to handle non-linear patterns
- Robustness to imbalanced data
- Minimal feature scaling requirements

### 4. Training & Cross-Validation
- Stratified train-test split
- 5-fold Stratified Cross-Validation
- F1-score used as primary metric

### 5. Evaluation Metrics
- Precision, Recall, F1-score
- ROC-AUC
- Confusion Matrix

### 6. Model Persistence & Reproducibility
- Model saved using joblib
- Fixed random seeds for reproducibility

---

## Results
- Cross-validated F1-score: ~0.83
- ROC-AUC: ~0.92
- Precision (Fraud): ~0.97
- Recall (Fraud): ~0.71

---

## Notes
- Accuracy not used due to class imbalance
- Pipeline designed with production best practices

## TASK 2: Model Debugging & Stability

### Observed Issues
- Model performance varied across different runs
- Predictions were unstable for identical inputs

### Root Cause Analysis

1. Randomness not fully controlled  
   - Model training and cross-validation involved randomness

2. Feature scaling inconsistency  
   - Scaling not bound to model pipeline

3. Data leakage risk  
   - Preprocessing steps applied outside a unified pipeline

4. Evaluation instability  
   - High variance due to imbalanced dataset



### Debug Checklist
- [x] Fixed random seeds everywhere
- [x] Used stratified sampling
- [x] Verified no preprocessing leakage
- [x] Ensured consistent feature order
- [x] Evaluated using stable metrics (F1, ROC-AUC)

## TASK 3: Model Performance Improvement

### Improvements Applied
- Hyperparameter tuning of RandomForest
- Increased model capacity while controlling overfitting
- Threshold tuning for better fraud recall

### Results
| Version | F1 Score |
|------|----------|
| Baseline | ~0.72 |
| Improved | ~0.82 |

### Justification
Fraud patterns are non-linear and rare.
Deeper trees with controlled splits captured complex interactions.
Lower decision threshold improved recall without sacrificing precision.

## TASK 4: ML System Design – Fraud Detection

### System Overview
The system is designed to detect fraudulent credit card transactions
in real time using a trained machine learning model.

---

### Components

#### 1. Data Ingestion
- Incoming transactions are received via APIs or streaming systems
- Example tools: REST API, Kafka

#### 2. Feature Engineering Service
- Applies the same feature transformations used during training
- Ensures feature consistency between training and inference

#### 3. Model Inference
- Trained ML pipeline is loaded from disk
- Generates fraud probability for each transaction
- Threshold-based decision (fraud / non-fraud)

#### 4. Monitoring & Drift Detection
- Tracks prediction distribution and feature drift
- Monitors key metrics like precision, recall, and F1-score
- Alerts triggered if performance drops

#### 5. Retraining Strategy
- Periodic retraining using newly labeled data
- Triggered when model performance degrades beyond a threshold
- Updated model redeployed after validation

---

### Production Considerations
- Model versioning
- Logging predictions for audit
- Rollback mechanism in case of failure

## System design diagram available in:


---

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Joblib
- Matplotlib

---


## How to Run
```
1. Create virtual environment and install dependencies
pip install -r requirements.txt
python run_pipeline.py
This command runs the complete ML pipeline end-to-end
and prints evaluation metrics.
```