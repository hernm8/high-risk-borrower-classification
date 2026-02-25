
# Predicting High-Risk Borrowers Using Machine Learning

Developed a classification model to identify borrowers likely to default, improving risk assessment and supporting data-driven lending decisions.

## Project Overview
This project builds and evaluates supervised machine learning models to predict loan default risk using borrower and loan attributes. The workflow includes data cleaning, feature engineering, model training, and threshold tuning to support practical decision-making in lending.

**Primary goal:** predict high-risk borrowers (default = 1) and compare model performance across metrics relevant to credit risk.

## Dataset
- Source: LendingClub-style loan performance data (public historical loan data)
- Target: `default` (1 = default, 0 = non-default)
- Note: Raw dataset is not included in this repo due to size/licensing.  
  Use `data/raw/` for your local copy and see instructions below.

## Methods
- Data cleaning: missing value handling, outlier checks, type fixes
- Feature engineering: encoding categorical variables, scaling where needed
- Models tested (examples):
  - Logistic Regression (baseline)
  - Random Forest / Gradient Boosting
  - KNN (baseline comparison)
- Evaluation:
  - Confusion Matrix, Precision/Recall/F1
  - ROC-AUC
  - Threshold tuning to optimize recall/precision tradeoff for defaults

## Results (Example Summary)
- Best ROC-AUC: **[X.XX]**
- Default-class recall at chosen threshold: **[X.XX]**
- Key takeaway: tuned thresholds improved detection of high-risk borrowers, enabling more conservative lending decisions when desired.

> Replace bracketed values with your final numbers.

## Repository Structure
- `notebooks/` – end-to-end analysis and modeling workflow
- `src/` – reusable Python modules for training/evaluation
- `reports/` – saved metrics and figures

## How to Run
### 1) Setup environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
