# ML Engineer Practical Assessment – Fraud Detection

## Problem Statement
Build a production-grade machine learning pipeline for fraud detection
using credit card transaction data.

---

## Dataset

### Credit Card Fraud Detection Dataset

- Transactions made by European cardholders  
- Highly imbalanced dataset (~0.17% fraud cases)  
- Target column: `Class`  
  - `0` → Normal transaction  
  - `1` → Fraud transaction  

🔗 Dataset link (Kaggle):  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

- Dataset is not included in this repository due to size limits.  
- After downloading, place the file here:

- data/raw/creditcard.csv

---

## Project Structure
- data/raw: original dataset
- data/processed: cleaned data
- src: modular source code
- models: saved trained models

---

## Project Structure
```
ml-engineer-assessment/
├── src/ # All ML logic (clean & modular)
├── models/ # Saved trained model
├── diagrams/ # System design diagram
├── run_pipeline.py # One command to run everything
├── requirements.txt # Dependencies
└── README.md
```

---


---

## What This Pipeline Does

### 1. Data Loading

- Loads the dataset from `data/raw/creditcard.csv`
- Verifies data shape and structure

### 2. Data Cleaning

- Removes duplicate rows  
- Confirms there are no missing values  
- Keeps class distribution intact  

### 3. Feature Engineering

Extra features are created to help the model learn better patterns:

- `Amount_log` → log transformation of transaction amount  
- `Amount_squared` → emphasizes high-value transactions  
- `Hour` → hour extracted from transaction time  
- `Is_night` → night-time transaction flag  

### 4. Model Training

- Model used: `RandomForestClassifier`  
- Stratified train-test split  
- Cross-validated using F1 score  
- Random seed fixed for reproducibility  
- Trained model saved to `models/fraud_model.pkl`  

### 5. Evaluation

The pipeline reports:

- Precision  
- Recall  
- F1 Score  
- ROC-AUC  

These metrics are more meaningful than accuracy for imbalanced fraud data.

---

## Model Stability & Debugging

During development:

- Performance variations were observed across runs  
- Preprocessing and training logic were stabilized  
- Randomness was controlled  
- Feature flow was made consistent between training and evaluation  

Final results are stable and reproducible across multiple runs.

---

## How to Run the Project (Very Important)

### 1. Create and activate virtual environment
```
python -m venv venv
venv\Scripts\activate
```
### 2. Install dependencies

```
pip install -r requirements.txt
```


### 3. Run the full pipeline

```
python run_pipeline.py
```


---

## Example Output
```
Starting Fraud Detection Pipeline

Data loaded
Data cleaned
Features engineered
Train-test split done
Model trained | CV F1 Score: 0.842

Evaluation Results
Precision: 0.885
Recall: 0.726
F1 Score: 0.798
ROC-AUC: 0.972

Pipeline executed successfully!
```

This single command proves that the entire ML system is working end-to-end.

---

## System Design

A simple real-world fraud detection system design is included.

### Flow

- Transaction data ingestion  
- Feature engineering service  
- Model inference  
- Fraud probability output  
- Monitoring & retraining  

Diagram available at:

diagrams/system_design.png


---

## Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Joblib  

---

## Final Notes

- Accuracy is intentionally not used due to class imbalance  
- Focus is on F1 score, stability, and reproducibility  
- Project is structured like a real ML engineering codebase, not a notebook experiment  

This repository is verified by running `run_pipeline.py` successfully.

