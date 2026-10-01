"""
train_model.py
--------------
Trains a Random Forest Classifier to predict student placement status.
Saves the complete preprocessing + model pipeline to model/placement_model.pkl.
"""

import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import joblib

# ── Paths ──────────────────────────────────────────────────────────────────
TRAIN_PATH = os.path.join("data", "train.csv")
TEST_PATH  = os.path.join("data", "test.csv")
MODEL_PATH = os.path.join("model", "placement_model.pkl")

# ── Feature definitions ────────────────────────────────────────────────────
# Student_ID is excluded — it is a unique identifier with no predictive value.
TARGET = "Placement_Status"

NUMERICAL_FEATURES = [
    "Age", "CGPA", "Internships", "Projects",
    "Coding_Skills", "Communication_Skills",
    "Aptitude_Test_Score", "Soft_Skills_Rating",
    "Certifications", "Backlogs"
]

CATEGORICAL_FEATURES = ["Gender", "Degree", "Branch"]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES   # 13 features total

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading data...")
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

print(f"  Train set : {train_df.shape[0]:,} rows, {train_df.shape[1]} columns")
print(f"  Test  set : {test_df.shape[0]:,} rows,  {test_df.shape[1]} columns")

X_train = train_df[ALL_FEATURES]
y_train = train_df[TARGET]

X_test  = test_df[ALL_FEATURES]
y_test  = test_df[TARGET]

# ── Preprocessing pipeline ─────────────────────────────────────────────────
# Numerical: StandardScaler
# Categorical: OneHotEncoder (handle_unknown='ignore' for robustness)
numerical_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformer, NUMERICAL_FEATURES),
    ("cat", categorical_transformer, CATEGORICAL_FEATURES)
])

# ── Full pipeline: preprocessing + classifier ──────────────────────────────
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier",   RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",   # addresses the ~64/36 class imbalance
        n_jobs=-1
    ))
])

# ── Train ──────────────────────────────────────────────────────────────────
print("\nTraining Random Forest Classifier...")
pipeline.fit(X_train, y_train)
print("Training complete.")

# ── Evaluate on test set ───────────────────────────────────────────────────
print("\n-- Evaluation on test.csv ------------------------------------------")
y_pred = pipeline.predict(X_test)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, pos_label="Placed")
rec  = recall_score(y_test, y_pred, pos_label="Placed")
f1   = f1_score(y_test, y_pred, pos_label="Placed")
cm   = confusion_matrix(y_test, y_pred, labels=["Placed", "Not Placed"])

print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1-Score  : {f1:.4f}")
print("\n  Confusion Matrix (rows=Actual, cols=Predicted):")
print("                Placed  Not Placed")
print(f"  Placed      : {cm[0,0]:>6}  {cm[0,1]:>10}")
print(f"  Not Placed  : {cm[1,0]:>6}  {cm[1,1]:>10}")
print("\n  Full Classification Report:")
print(classification_report(y_test, y_pred))

# ── Save pipeline ──────────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)
joblib.dump(pipeline, MODEL_PATH)
print(f"Model saved to: {MODEL_PATH}")
