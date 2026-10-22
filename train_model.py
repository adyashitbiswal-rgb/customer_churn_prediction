"""Train a churn classifier using a synthetic educational dataset."""
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE / "churn_synthetic.csv")
X = df.drop(columns="churn")
y = df["churn"]

numeric = ["tenure_months", "monthly_charge", "support_calls"]
categorical = ["contract_type", "internet_type"]
preprocess = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]), numeric),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
])
pipeline = Pipeline([
    ("preprocess", preprocess),
    ("model", RandomForestClassifier(n_estimators=250, class_weight="balanced",
                                     random_state=42))
])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)
proba = pipeline.predict_proba(X_test)[:, 1]

print("Confusion matrix:\n", confusion_matrix(y_test, pred))
print("\nClassification report:\n", classification_report(y_test, pred, zero_division=0))
print(f"ROC-AUC: {roc_auc_score(y_test, proba):.3f}")
joblib.dump(pipeline, HERE / "churn_model.joblib")
print("Saved model to churn_model.joblib")
