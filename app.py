from flask import Flask, render_template, request
from pathlib import Path
import pandas as pd
import json
import joblib

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "cvd_logreg_pipeline.joblib"
META_PATH = BASE_DIR / "model" / "model_metadata.json"

model = None
meta = None

if META_PATH.exists():
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))

if MODEL_PATH.exists():
    try:
        model = joblib.load(MODEL_PATH)
    except Exception:
        model = None

if meta is None:
    meta = {
        "categorical_cols": [
            "General_Health", "Checkup", "Exercise", "Skin_Cancer", "Other_Cancer",
            "Depression", "Diabetes", "Arthritis", "Sex", "Age_Category", "Smoking_History"
        ],
        "numeric_cols": [
            "Height_(cm)", "Weight_(kg)", "BMI", "Alcohol_Consumption",
            "Fruit_Consumption", "Green_Vegetables_Consumption", "FriedPotato_Consumption"
        ],
        "category_options": {
            "General_Health": ["Excellent", "Very good", "Good", "Fair", "Poor"],
            "Checkup": ["Within the past year", "Within the past 2 years", "Within the past 5 years", "5 or more years ago"],
            "Exercise": ["Yes", "No"],
            "Skin_Cancer": ["Yes", "No"],
            "Other_Cancer": ["Yes", "No"],
            "Depression": ["Yes", "No"],
            "Diabetes": ["No", "No, pre-diabetes or borderline diabetes", "Yes"],
            "Arthritis": ["Yes", "No"],
            "Sex": ["Female", "Male"],
            "Age_Category": ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"],
            "Smoking_History": ["Yes", "No"]
        },
        "defaults": {
            "General_Health": "Good",
            "Checkup": "Within the past year",
            "Exercise": "Yes",
            "Skin_Cancer": "No",
            "Other_Cancer": "No",
            "Depression": "No",
            "Diabetes": "No",
            "Arthritis": "No",
            "Sex": "Female",
            "Age_Category": "45-49",
            "Smoking_History": "No",
            "Height_(cm)": 168,
            "Weight_(kg)": 75,
            "BMI": 26.5,
            "Alcohol_Consumption": 2,
            "Fruit_Consumption": 30,
            "Green_Vegetables_Consumption": 20,
            "FriedPotato_Consumption": 2
        },
        "metrics": {
            "accuracy": 0.9191368117725146,
            "precision": 0.4992,
            "recall": 0.06107328794553464,
            "f1": 0.1088,
            "auroc": 0.8390080495982254,
            "confusion_matrix": [[56471, 306], [4689, 305]]
        }
    }

categorical_cols = meta["categorical_cols"]
numeric_cols = meta["numeric_cols"]
category_options = meta["category_options"]
defaults = meta["defaults"]
metrics = meta["metrics"]

def build_input_dataframe(form_data):
    row = {}
    for col in categorical_cols:
        row[col] = form_data.get(col, defaults[col])
    for col in numeric_cols:
        try:
            row[col] = float(form_data.get(col, defaults[col]))
        except (TypeError, ValueError):
            row[col] = float(defaults[col])
    return pd.DataFrame([row])

def fallback_probability(form_data):
    bmi = float(form_data.get("BMI", defaults["BMI"]))
    age = form_data.get("Age_Category", defaults["Age_Category"])
    smoke = form_data.get("Smoking_History", defaults["Smoking_History"])
    diabetes = form_data.get("Diabetes", defaults["Diabetes"])
    exercise = form_data.get("Exercise", defaults["Exercise"])
    general_health = form_data.get("General_Health", defaults["General_Health"])

    score = 0.12
    if bmi >= 30:
        score += 0.18
    elif bmi >= 25:
        score += 0.08

    if age in ["60-64", "65-69", "70-74", "75-79", "80+"]:
        score += 0.18
    elif age in ["50-54", "55-59"]:
        score += 0.10

    if smoke == "Yes":
        score += 0.10

    if diabetes != "No":
        score += 0.22

    if exercise == "No":
        score += 0.08

    if general_health in ["Fair", "Poor"]:
        score += 0.10

    return min(max(score, 0.03), 0.95)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    risk_band = None
    submitted = defaults.copy()

    if request.method == "POST":
        submitted.update(request.form.to_dict())
        if model is not None:
            X = build_input_dataframe(request.form)
            probability = float(model.predict_proba(X)[:, 1][0])
        else:
            probability = fallback_probability(request.form)

        if probability >= 0.70:
            risk_band = "High Risk"
        elif probability >= 0.40:
            risk_band = "Moderate Risk"
        else:
            risk_band = "Lower Risk"

        prediction = "Positive screening alert" if probability >= 0.5 else "No immediate screening alert"

    return render_template(
        "index.html",
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        category_options=category_options,
        defaults=defaults,
        submitted=submitted,
        prediction=prediction,
        probability=probability,
        risk_band=risk_band,
        metrics=metrics
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
