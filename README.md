# Student Placement Prediction

## Project Overview
This project predicts whether a student will be **Placed** or **Not Placed** based on their academic performance, skills, and other attributes. It is built as part of the **AICTE Machine Learning Internship** capstone project.

---

## Problem Statement
Campus placement is a critical milestone for students. Predicting placement outcomes early helps students identify areas for improvement and allows institutions to provide targeted guidance. This project uses machine learning to predict placement status from student profiles.

---

## Objectives
- Build a binary classification model to predict student placement.
- Use real student data (train.csv / test.csv) for training and evaluation.
- Deploy the model via a simple Flask web application.
- Create a clean, responsive web UI for making predictions.

---

## Dataset Description
| Property | Value |
|---|---|
| Training samples | 45,000 |
| Test samples | 5,000 |
| Total features | 15 columns |
| Target column | `Placement_Status` (`Placed` / `Not Placed`) |
| Missing values | None |
| Class distribution | Not Placed ≈ 63.8%, Placed ≈ 36.2% |

---

## Features Used (13 input features)

| Feature | Type | Description |
|---|---|---|
| Age | Numerical | Student age (18–24) |
| Gender | Categorical | Male / Female |
| Degree | Categorical | B.Tech / B.Sc / BCA / MCA |
| Branch | Categorical | CSE / IT / ECE / ME / Civil |
| CGPA | Numerical | Cumulative GPA (4.50–9.80) |
| Internships | Numerical | Number of internships (0–3) |
| Projects | Numerical | Number of projects (1–6) |
| Coding_Skills | Numerical | Coding rating (1–10) |
| Communication_Skills | Numerical | Communication rating (1–10) |
| Aptitude_Test_Score | Numerical | Aptitude score (35–100) |
| Soft_Skills_Rating | Numerical | Soft skills rating (1–10) |
| Certifications | Numerical | Number of certifications (0–3) |
| Backlogs | Numerical | Number of backlogs (0–3) |

> **Excluded**: `Student_ID` — unique identifier, no predictive value.

## Target Variable
`Placement_Status`: `Placed` or `Not Placed`

---

## Technologies Used
| Category | Technology |
|---|---|
| Language | Python 3.x |
| Data handling | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Model persistence | Joblib |
| Web framework | Flask |
| Frontend | HTML5, CSS3, Bootstrap 5 |

---

## Project Structure
```
student_placement_prediction/
│
├── data/
│   ├── train.csv              # Training dataset (45,000 rows)
│   └── test.csv               # Test dataset (5,000 rows)
│
├── model/
│   └── placement_model.pkl    # Saved preprocessing + model pipeline
│
├── templates/
│   └── index.html             # Flask HTML template
│
├── static/
│   └── style.css              # Custom CSS styles
│
├── train_model.py             # Model training script
├── app.py                     # Flask web application
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── PROJECT_REPORT.md          # Internship project report
```

---

## Installation Instructions

### Prerequisites
- Python 3.8 or higher installed
- pip package manager

### Step 1 — Clone or download the project
Place the project folder on your machine. The `data/` folder must contain `train.csv` and `test.csv`.

### Step 2 — Create a virtual environment (Windows)
Open Command Prompt or PowerShell in the project folder:

```powershell
python -m venv venv
venv\Scripts\activate
```

### Step 3 — Install required packages
```powershell
pip install -r requirements.txt
```

---

## How to Train the Model
```powershell
python train_model.py
```
This will:
- Load `data/train.csv` for training and `data/test.csv` for evaluation.
- Train a Random Forest Classifier pipeline.
- Print evaluation metrics in the terminal.
- Save the pipeline to `model/placement_model.pkl`.

**Expected output:**
```
Loading data...
  Train set : 45,000 rows, 15 columns
  Test  set :  5,000 rows, 15 columns
Training Random Forest Classifier...
Training complete.
-- Evaluation on test.csv --
  Accuracy  : 1.0000  (100.00%)
  Precision : 1.0000
  Recall    : 1.0000
  F1-Score  : 1.0000
Model saved to: model\placement_model.pkl
```

---

## How to Run the Flask Application

> Make sure you have trained the model first (`model/placement_model.pkl` must exist).

```powershell
python app.py
```

---

## How to Access the Application
Open your browser and go to:
```
http://127.0.0.1:5000
```

---

## Example Usage
1. Open the browser at `http://127.0.0.1:5000`.
2. Fill in the student details form (Age, CGPA, Skills, etc.).
3. Click **Predict Placement**.
4. The result will show **Placed** or **Not Placed** with a probability percentage.

---

## Model Evaluation Metrics (on test.csv — 5,000 samples)

| Metric | Value |
|---|---|
| Accuracy | 100.00% |
| Precision (Placed) | 1.0000 |
| Recall (Placed) | 1.0000 |
| F1-Score (Placed) | 1.0000 |

**Confusion Matrix:**
|  | Predicted: Placed | Predicted: Not Placed |
|---|---|---|
| Actual: Placed | 1812 | 0 |
| Actual: Not Placed | 0 | 3188 |

---

## Limitations
- The model is trained on a structured, clean dataset. Real-world data may have missing values, noise, or different feature distributions.
- 100% accuracy on this particular test set may indicate the test set was sampled from the same distribution as the training data. Always validate on truly unseen, independent data.
- The model does not capture subjective human factors (personality, interview performance, etc.).

---

## Future Enhancements
- Try additional models: Logistic Regression, XGBoost, SVM for comparison.
- Add cross-validation for more robust evaluation.
- Deploy on a cloud platform (Heroku, Render, etc.).
- Add a model explainability section using SHAP or feature importance charts.
- Extend the UI to show feature importance and guidance on how to improve.
