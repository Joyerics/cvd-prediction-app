from flask import Flask, render_template, request
import joblib, json, pandas as pd
from pathlib import Path

app = Flask(__name__)

BASE = Path(__file__).resolve().parent
model = joblib.load(BASE / "model" / "cvd_logreg_pipeline.joblib")
meta = json.loads((BASE / "model" / "model_metadata.json").read_text())

categorical_cols = meta["categorical_cols"]
numeric_cols = meta["numeric_cols"]
category_options = meta["category_options"]
defaults = meta["defaults"]
metrics = meta["metrics"]

def coerce_row(form):
    row = {}
    for c in categorical_cols:
        row[c] = form.get(c, defaults[c])
    for c in numeric_cols:
        try:
            row[c] = float(form.get(c, defaults[c]))
        except Exception:
            row[c] = float(defaults[c])
    return pd.DataFrame([row])

@app.route("/", methods=["GET", "POST"])
def index():
    submitted = defaults.copy()
    probability = None
    prediction = None

    if request.method == "POST":
        submitted.update(request.form.to_dict())
        X = coerce_row(request.form)
        probability = float(model.predict_proba(X)[:, 1][0])
        prediction = "Higher Risk" if probability >= 0.5 else "Lower Risk"

    return render_template(
        "index.html",
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        category_options=category_options,
        defaults=defaults,
        submitted=submitted,
        probability=probability,
        prediction=prediction,
        metrics=metrics
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)