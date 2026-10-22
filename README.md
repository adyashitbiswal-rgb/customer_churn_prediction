# Customer Churn Prediction

## Goal
Train a binary classification model to estimate whether a customer may churn.

## Stack
Python, Pandas, scikit-learn.

## Run
```bash
pip install -r requirements.txt
python train_model.py
python predict_churn.py
```

## What to discuss
- Why stratified train/test splitting matters.
- Why accuracy alone can be misleading.
- Precision vs. recall and the cost of false negatives.
- How a company could validate the model before using it.

**Important:** `churn_synthetic.csv` is generated synthetic data, not customer records. Model performance is not evidence of real-world effectiveness.
