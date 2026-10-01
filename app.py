"""
app.py
------
Flask web application for Student Placement Prediction.
Loads the pre-trained pipeline and serves a prediction form.
"""

import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

# ── App setup ──────────────────────────────────────────────────────────────
app = Flask(__name__)

MODEL_PATH = os.path.join("model", "placement_model.pkl")

# Load the trained pipeline once at startup (not on every request)
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at '{MODEL_PATH}'. "
        "Please run train_model.py first."
    )
pipeline = joblib.load(MODEL_PATH)
print(f"Model loaded from: {MODEL_PATH}")

# ── Feature order must exactly match train_model.py ────────────────────────
NUMERICAL_FEATURES   = [
    "Age", "CGPA", "Internships", "Projects",
    "Coding_Skills", "Communication_Skills",
    "Aptitude_Test_Score", "Soft_Skills_Rating",
    "Certifications", "Backlogs"
]
CATEGORICAL_FEATURES = ["Gender", "Degree", "Branch"]
ALL_FEATURES         = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


# ── Routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Render the prediction form."""
    return render_template("index.html", prediction=None, error=None)


@app.route("/predict", methods=["POST"])
def predict():
    """Receive form data, run the pipeline, and return prediction."""
    try:
        # ── Parse and validate inputs ──────────────────────────────────────
        age                  = int(request.form["Age"])
        cgpa                 = float(request.form["CGPA"])
        internships          = int(request.form["Internships"])
        projects             = int(request.form["Projects"])
        coding_skills        = int(request.form["Coding_Skills"])
        communication_skills = int(request.form["Communication_Skills"])
        aptitude_score       = int(request.form["Aptitude_Test_Score"])
        soft_skills          = int(request.form["Soft_Skills_Rating"])
        certifications       = int(request.form["Certifications"])
        backlogs             = int(request.form["Backlogs"])
        gender               = request.form["Gender"]
        degree               = request.form["Degree"]
        branch               = request.form["Branch"]

        # Basic range validation
        errors = []
        if not (18 <= age <= 30):
            errors.append("Age must be between 18 and 30.")
        if not (0.0 <= cgpa <= 10.0):
            errors.append("CGPA must be between 0.0 and 10.0.")
        if not (0 <= internships <= 10):
            errors.append("Internships must be between 0 and 10.")
        if not (0 <= projects <= 20):
            errors.append("Projects must be between 0 and 20.")
        if not (1 <= coding_skills <= 10):
            errors.append("Coding Skills must be between 1 and 10.")
        if not (1 <= communication_skills <= 10):
            errors.append("Communication Skills must be between 1 and 10.")
        if not (0 <= aptitude_score <= 100):
            errors.append("Aptitude Test Score must be between 0 and 100.")
        if not (1 <= soft_skills <= 10):
            errors.append("Soft Skills Rating must be between 1 and 10.")
        if not (0 <= certifications <= 20):
            errors.append("Certifications must be between 0 and 20.")
        if not (0 <= backlogs <= 20):
            errors.append("Backlogs must be between 0 and 20.")

        if errors:
            return render_template("index.html", prediction=None, error="; ".join(errors))

        # ── Build input DataFrame (column order matters) ───────────────────
        input_data = pd.DataFrame([{
            "Age":                  age,
            "CGPA":                 cgpa,
            "Internships":          internships,
            "Projects":             projects,
            "Coding_Skills":        coding_skills,
            "Communication_Skills": communication_skills,
            "Aptitude_Test_Score":  aptitude_score,
            "Soft_Skills_Rating":   soft_skills,
            "Certifications":       certifications,
            "Backlogs":             backlogs,
            "Gender":               gender,
            "Degree":               degree,
            "Branch":               branch,
        }])

        # ── Predict ────────────────────────────────────────────────────────
        prediction  = pipeline.predict(input_data)[0]          # "Placed" / "Not Placed"
        proba       = pipeline.predict_proba(input_data)[0]    # [prob_class_0, prob_class_1]
        classes     = pipeline.classes_                        # e.g. ['Not Placed', 'Placed']

        placed_idx      = list(classes).index("Placed")
        placed_prob     = round(proba[placed_idx] * 100, 2)
        not_placed_prob = round(100 - placed_prob, 2)

        result = {
            "label":          prediction,
            "placed_prob":    placed_prob,
            "not_placed_prob": not_placed_prob,
            "is_placed":      prediction == "Placed",
        }

        return render_template("index.html", prediction=result, error=None)

    except (ValueError, KeyError) as e:
        return render_template("index.html", prediction=None,
                               error=f"Invalid input: {e}. Please check all fields.")


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
