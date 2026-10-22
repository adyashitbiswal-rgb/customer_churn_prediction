"""Predict churn risk for a sample customer."""
from pathlib import Path
import joblib
import pandas as pd

HERE = Path(__file__).resolve().parent
model_path = HERE / "churn_model.joblib"
if not model_path.exists():
    raise SystemExit("Model file not found. Run `python train_model.py` first.")

customer = pd.DataFrame([{
    "tenure_months": 8,
    "monthly_charge": 95.0,
    "support_calls": 3,
    "contract_type": "Month-to-month",
    "internet_type": "Fiber"
}])
risk = model.predict_proba(customer)[0, 1]
print(f"Predicted churn probability on synthetic data: {risk:.1%}")
