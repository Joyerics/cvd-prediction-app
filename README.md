#  Cardiovascular Disease Prediction System

A full-stack machine learning project that predicts the likelihood of cardiovascular disease (CVD) using real-world health data. This project integrates **data analysis, statistical testing, machine learning modeling, and a deployed web application** for real-time risk prediction.

---

## Project Overview

This project aims to:

* Predict cardiovascular disease risk using patient health indicators
* Validate relationships using statistical hypothesis testing
* Provide a **professional healthcare-style web interface**
* Deploy a working ML model for real-world usage

---

## Dataset

* Source: CDC BRFSS 2021 (via Kaggle)
* Size: **308,854 rows × 19 features**
* Type: Public health survey data

### Key Features:

* Age Category
* Sex
* BMI
* Smoking History
* Physical Activity
* Alcohol Consumption
* General Health

---

## Methodology

### 1. Data Preprocessing

* Cleaned categorical and numeric features
* Verified no missing values
* Encoded categorical variables
* Scaled numeric features

---

### 2. Exploratory Data Analysis (EDA)

* Identified **severe class imbalance (91.91% vs 8.09%)**
* Found strong relationships with:

  * Age
  * BMI
  * Smoking
  * Sex

---

### 3. Hypothesis Testing

| Test       | Purpose                   | Result      |
| ---------- | ------------------------- | ----------- |
| Chi-square | Smoking vs Heart Disease  | Significant |
| Chi-square | Exercise vs Heart Disease | Significant |
| t-test     | BMI vs Heart Disease      | Significant |
| ANOVA      | BMI across Age Groups     | Significant |
| Chi-square | Sex vs Heart Disease      | Significant |

---

### 4. Machine Learning Model

* Model: **Logistic Regression**
* Pipeline:

  * OneHotEncoder (categorical)
  * StandardScaler (numeric)

---

## Model Performance

| Metric   | Value  |
| -------- | ------ |
| Accuracy | 91.91% |
| AUROC    | 0.8390 |
| Recall   | 6.11%  |

### Note:

High accuracy is due to class imbalance. Recall is low, meaning the model misses many positive cases.

---

## Fairness & Bias Analysis

* Male Recall: **9.06%**
* Female Recall: **1.74%**
* Performance varies across demographic groups

👉 Indicates need for:

* Threshold tuning
* Class balancing
* Fairness-aware modeling

---

## Web Application (UI)

The project includes a **fully deployed healthcare-style web application**.

### Features:

* Landing page
* Patient input form
* Prediction result page
* Real-time probability output
* Clean, professional UI (hospital-style)

### Live App:

👉 https://healthcare-prediction-model.onrender.com

---

## Project Structure

```
├── app.py                 # Flask application
├── model/
│   └── model.pkl         # Trained ML model
├── templates/
│   ├── index.html
│   ├── form.html
│   └── result.html
├── static/
│   └── styles.css
├── notebooks/
│   └── analysis.ipynb
├── data/
│   └── dataset.csv
├── requirements.txt
├── README.md
```

---

## Installation

```bash
git clone https://github.com/your-username/cvd-prediction.git
cd cvd-prediction

pip install -r requirements.txt
python app.py
```

---

## Key Findings

* Age is the strongest predictor of heart disease
* Smoking and lack of exercise significantly increase risk
* Higher BMI is associated with increased CVD risk
* Significant gender-based differences observed

---

## Limitations

* Severe class imbalance affects recall
* Model misses many positive cases
* Fairness disparities across groups

---

## Future Work

* Apply SMOTE (class balancing)
* Use ensemble models (Random Forest, XGBoost)
* Improve recall through threshold tuning
* Implement fairness-aware machine learning

---

## Contributors

* **Chinenye Onyedika**

  * UI Development
  * Model Integration
  * Deployment

* **Nene Tenabe**

  * Data Cleaning
  * Hypothesis Testing
  * Analysis

---

## License

This project is for academic purposes.

---

## Acknowledgements

* CDC BRFSS Dataset
* Kaggle Community
* Course: DAB304

---
Author

Chinenye Joy Onyedika Data Analyst |Heathcare Analyst
Windsor, Ontario, Canada 
 LinkedIn | GitHub