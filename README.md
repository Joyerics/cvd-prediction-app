# Hospital CVD Risk Screening App

A GitHub-ready Flask project for **direct Render deployment**. It loads a saved Logistic Regression pipeline, accepts patient profile inputs, and returns the estimated probability of heart disease.

## Model summary
- Accuracy: 0.9191
- Precision: 0.4992
- Recall: 0.0611
- F1 score: 0.1088
- AUROC: 0.8390
- Confusion Matrix: [[56471, 306], [4689, 305]]

## Repository structure
```text
.
├── app.py
├── requirements.txt
├── render.yaml
├── .python-version
├── .gitignore
├── model/
│   ├── cvd_logreg_pipeline.joblib
│   └── model_metadata.json
├── templates/
│   └── index.html
├── static/
│   └── styles.css
├── scripts/
│   ├── train_and_save_model.py
│   └── anova_test_bmi_by_age.py
├── notebooks/
│   └── CVD_project_extended.ipynb
└── data/
    └── .gitkeep
```

## Local run
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## Render deployment
This repo is already set up for Render:
- `render.yaml` included
- build command: `pip install -r requirements.txt`
- start command: `gunicorn app:app`

Push this folder to GitHub, then create a new Render web service from the repo.

## Full ANOVA test included
See:
- `scripts/anova_test_bmi_by_age.py`

This script:
- states **H0** and **H1**
- runs the ANOVA test
- prints group means
- makes a decision automatically:
  - **Reject H0** if `p < 0.05`
  - **Fail to reject H0** otherwise

## Notes
- The UI is designed with a clean navy-teal clinical palette, different from your capstone layout.
- It is suitable for a hospital-style demo or screening workflow.
