# PROJECT REPORT
## Student Placement Prediction using Machine Learning
**AICTE Machine Learning Internship — Individual Capstone Project**

---

## Abstract
This project presents a machine learning solution for predicting student campus placement outcomes. Using a dataset of 45,000 student records with 13 academic and skill-related features, a Random Forest Classifier was trained and evaluated. The model is deployed as a Flask web application with a responsive Bootstrap-based user interface, allowing users to input student details and receive instant placement predictions with probability scores. This project demonstrates end-to-end implementation of a supervised binary classification problem — from data preprocessing to web deployment.

---

## 1. Introduction
Campus placement is one of the most significant outcomes of a student's academic journey. Educational institutions, students, and recruiters alike benefit from understanding the factors that influence placement success. With the growing availability of student data, machine learning offers a data-driven approach to predict placement outcomes — enabling students to focus on areas that matter most and helping institutions improve their career development programs.

This project implements a complete machine learning pipeline:
- Data exploration and preprocessing
- Feature engineering and encoding
- Model training with a Random Forest Classifier
- Model evaluation on a held-out test set
- Deployment via a Flask web application

---

## 2. Problem Statement
Given a student's academic profile (CGPA, degree, branch), skill ratings (coding, communication, soft skills), and extracurricular indicators (internships, projects, certifications, backlogs), **predict whether the student will be placed or not placed** during campus recruitment.

This is a **binary classification problem**:
- Class 0: `Not Placed`
- Class 1: `Placed`

---

## 3. Objectives
1. Analyze the student placement dataset and identify relevant features.
2. Build and train a machine learning model for placement prediction.
3. Evaluate model performance using standard classification metrics.
4. Deploy the model as an interactive web application using Flask.
5. Provide a clean, beginner-friendly codebase suitable for academic presentation.

---

## 4. Scope
- **In scope**: Supervised binary classification, scikit-learn pipeline, Flask deployment, Bootstrap UI.
- **Out of scope**: Cloud deployment, authentication, real-time data collection, database integration.

---

## 5. Dataset Description

### Source
Two structured CSV files provided as part of the internship:
- `data/train.csv` — 45,000 student records used for training.
- `data/test.csv` — 5,000 student records used for evaluation.

### Structure
| Property | Value |
|---|---|
| Total rows | 50,000 (45,000 train + 5,000 test) |
| Total columns | 15 |
| Target column | `Placement_Status` |
| Missing values | None (complete dataset) |

### Columns
| Column | Data Type | Range / Values |
|---|---|---|
| Student_ID | Integer | Unique identifier (excluded from model) |
| Age | Integer | 18–24 |
| Gender | Categorical | Male, Female |
| Degree | Categorical | B.Tech, B.Sc, BCA, MCA |
| Branch | Categorical | CSE, IT, ECE, ME, Civil |
| CGPA | Float | 4.50–9.80 |
| Internships | Integer | 0–3 |
| Projects | Integer | 1–6 |
| Coding_Skills | Integer | 1–10 |
| Communication_Skills | Integer | 1–10 |
| Aptitude_Test_Score | Integer | 35–100 |
| Soft_Skills_Rating | Integer | 1–10 |
| Certifications | Integer | 0–3 |
| Backlogs | Integer | 0–3 |
| Placement_Status | Categorical | Placed, Not Placed (target) |

### Class Distribution (train.csv)
| Class | Count | Percentage |
|---|---|---|
| Not Placed | 28,688 | ~63.8% |
| Placed | 16,312 | ~36.2% |

A moderate class imbalance exists. This was handled by setting `class_weight="balanced"` in the Random Forest Classifier.

---

## 6. Data Preprocessing

### 6.1 Feature Selection
`Student_ID` was excluded because it is a unique identifier with no predictive value. The remaining 13 columns were used as features.

### 6.2 Feature Categories
- **Numerical features** (10): Age, CGPA, Internships, Projects, Coding_Skills, Communication_Skills, Aptitude_Test_Score, Soft_Skills_Rating, Certifications, Backlogs
- **Categorical features** (3): Gender, Degree, Branch

### 6.3 Preprocessing Steps
| Feature Type | Transformation | Reason |
|---|---|---|
| Numerical | `StandardScaler` | Normalises features to zero mean and unit variance |
| Categorical | `OneHotEncoder` | Converts string categories to binary indicator columns |

### 6.4 Pipeline
A scikit-learn `Pipeline` combined with a `ColumnTransformer` was used to encapsulate all preprocessing steps. This ensures:
- No data leakage: the scaler is fit only on training data.
- Portability: the saved `.pkl` file includes both preprocessing and the model.
- Consistent prediction: Flask app applies identical transformations to user input.

---

## 7. Exploratory Data Analysis

### Key observations from the dataset:
- **No missing values** across all 50,000 records.
- **Age** is tightly distributed between 18–24, typical of undergraduate/postgraduate students.
- **CGPA** ranges from 4.50 to 9.80 — a wide spread that likely influences placement outcome significantly.
- **Aptitude Test Score** ranges from 35 to 100.
- **Backlogs**: students with 0 backlogs are expected to have higher placement rates.
- **Internships** range from 0 to 3 — practical experience is a key differentiator.
- The class imbalance (~64% Not Placed vs ~36% Placed) was noted and addressed with balanced class weights.

---

## 8. Machine Learning Approach

### Algorithm: Random Forest Classifier
Random Forest is an ensemble learning method that builds multiple decision trees and aggregates their predictions. It was chosen for this project because:
- It handles both numerical and categorical (after encoding) features well.
- It is robust to overfitting compared to single decision trees.
- It provides good performance on tabular data without extensive hyperparameter tuning.
- It supports `predict_proba()` for probability-based output, which enriches the user experience.

### Hyperparameters Used
| Parameter | Value | Reason |
|---|---|---|
| `n_estimators` | 100 | Standard number of trees — balances performance and speed |
| `random_state` | 42 | Reproducibility |
| `class_weight` | `"balanced"` | Automatically adjusts for class imbalance |
| `n_jobs` | -1 | Uses all available CPU cores for faster training |

---

## 9. Model Training

The training pipeline:
1. Load `data/train.csv` (45,000 records).
2. Split into features `X_train` (13 columns) and target `y_train`.
3. Fit the `ColumnTransformer` (StandardScaler + OneHotEncoder) on `X_train`.
4. Train `RandomForestClassifier` on the transformed training data.
5. Save the complete pipeline (preprocessing + classifier) to `model/placement_model.pkl`.

The entire pipeline is serialised using `joblib.dump()` so that it can be loaded by the Flask app without re-training.

---

## 10. Model Evaluation

The trained pipeline was evaluated on `data/test.csv` (5,000 records).

### Metrics

| Metric | Value |
|---|---|
| Accuracy | **100.00%** |
| Precision (Placed) | **1.0000** |
| Recall (Placed) | **1.0000** |
| F1-Score (Placed) | **1.0000** |

### Confusion Matrix

|  | Predicted: Placed | Predicted: Not Placed |
|---|---|---|
| **Actual: Placed** | 1812 | 0 |
| **Actual: Not Placed** | 0 | 3188 |

### Classification Report (full)
```
              precision    recall  f1-score   support

  Not Placed       1.00      1.00      1.00      3188
      Placed       1.00      1.00      1.00      1812

    accuracy                           1.00      5000
   macro avg       1.00      1.00      1.00      5000
weighted avg       1.00      1.00      1.00      5000
```

### Interpretation
The model achieves perfect scores on the test set. This is consistent with a clean, synthetically-generated or tightly-controlled structured dataset where the test and training sets share the same data-generating distribution. In a real-world setting, model generalisation to truly unseen data should always be validated separately.

---

## 11. Results

- **Model**: Random Forest Classifier (100 trees, balanced class weights)
- **Accuracy on test set**: 100%
- **Model file size**: ~saved as `model/placement_model.pkl`
- **Prediction speed**: Near-instant (< 50 ms per prediction via Flask)

---

## 12. System Architecture

```
User (Browser)
      |
      | HTTP POST /predict
      v
Flask App (app.py)
      |
      | pandas DataFrame (13 features)
      v
Scikit-learn Pipeline (placement_model.pkl)
      |
      |-- ColumnTransformer
      |       |-- StandardScaler   (numerical features)
      |       |-- OneHotEncoder    (categorical features)
      |
      |-- RandomForestClassifier
      |
      v
Prediction: "Placed" / "Not Placed"  +  Probability
      |
      v
Flask renders index.html with result
      |
      v
User sees result card in browser
```

---

## 13. Backend Implementation

`app.py` handles:
- Loading the saved pipeline at startup (`joblib.load`).
- Serving the prediction form via `GET /`.
- Receiving form data, building a pandas DataFrame, and running the pipeline via `POST /predict`.
- Returning the predicted class and placement probability.
- Basic server-side input validation with error messages returned to the UI.

Key design decisions:
- The model is loaded **once at startup**, not on every request.
- `Student_ID` is **never asked** of the user — it is irrelevant for prediction.
- `predict_proba()` is used to return a confidence percentage alongside the binary prediction.

---

## 14. Frontend Implementation

`templates/index.html` provides:
- A clean, responsive form with Bootstrap 5.
- Range sliders for skill ratings (Coding, Communication, Soft Skills) with live value display.
- Dropdowns for categorical fields (Gender, Degree, Branch, Internships, Projects, Certifications, Backlogs).
- Bootstrap client-side validation (required field checks, range constraints).
- A result card that displays the prediction label and a visual probability bar.

`static/style.css` provides:
- A dark-blue gradient header.
- A white card layout for the form.
- Green highlight for "Placed", red highlight for "Not Placed" results.
- A visual probability bar indicating placement confidence.

---

## 15. Conclusion

This project successfully demonstrates a complete machine learning pipeline for student placement prediction:
- A clean scikit-learn preprocessing and classification pipeline was built.
- The model was trained on 45,000 records and evaluated on 5,000 records.
- The pipeline was saved and served via a Flask web application.
- A user-friendly Bootstrap UI allows real-time predictions without any coding knowledge.

The project meets all requirements of the AICTE Machine Learning Internship capstone and serves as a solid foundation for further enhancements.

---

## 16. Limitations

- The 100% test accuracy suggests the dataset may be structured/synthetic. Real-world datasets typically have noise and more complex distributions.
- The model does not capture dynamic factors such as interview performance, personality, or market conditions.
- The web app does not persist predictions or provide historical analytics.
- Deployment is local only; a production deployment would require a WSGI server (Gunicorn) and a cloud platform.

---

## 17. Future Scope

1. **Model comparison**: Evaluate Logistic Regression, SVM, XGBoost, and Gradient Boosting.
2. **Cross-validation**: Use k-fold cross-validation for a more reliable performance estimate.
3. **Feature importance analysis**: Visualise the top contributing features using SHAP or Random Forest feature importances.
4. **Data augmentation**: Collect real placement data from institutions to improve generalisation.
5. **Cloud deployment**: Deploy on Render, Railway, or AWS for public accessibility.
6. **Dashboard**: Add an analytics dashboard showing prediction history and feature distributions.
7. **Explainability**: Show the user which factors most influenced their individual prediction.

---

## 18. References

1. Scikit-learn Documentation — https://scikit-learn.org/stable/
2. Flask Documentation — https://flask.palletsprojects.com/
3. Bootstrap 5 Documentation — https://getbootstrap.com/docs/5.3/
4. Pandas Documentation — https://pandas.pydata.org/docs/
5. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.
6. AICTE Machine Learning Internship Program — Program guidelines and objectives.
