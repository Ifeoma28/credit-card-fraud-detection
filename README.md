# Credit Card Fraud Detection

## Author
Ifeoma Mary-Ann James

## Overview
Detecting fraudulent credit card transactions in a highly imbalanced dataset
(fraud = 0.054% of all transactions). Three models were built and compared,
moving from a linear baseline to a gradient-boosted tree model capable of
capturing non-linear behavioural signals.

## Dataset
Sourced from the [Kaggle Credit Card Transactions Fraud Detection dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
(`fraudTrain.csv`, `fraudTest.csv`). The raw CSVs are not included in this repo
(see `.gitignore`) — download them from Kaggle and place them in a local
`data/` folder before running the notebook.

## Feature Engineering
- Extracted `hour`, `day_of_week`, `month`, `year` from the transaction timestamp
- Calculated customer `age` from date of birth
- Calculated `distance_km` between customer and merchant using the Haversine formula
- Dropped non-predictive identifiers (card number, name, street, transaction number)

## Models
| Model | Recall (fraud) | Precision (fraud) | Notes |
|---|---|---|---|
| Logistic Regression | 76% | 8% | Linear baseline on 7 numeric features |
| PCA + Logistic Regression | 76% | 8% | 10 components, 100% explained variance — confirms linearity, not dimensionality, was the limiting factor |
| **LightGBM (Champion)** | **85%** | **31%** | F1 0.45, ROC-AUC 0.92 on held-out test set (555,719 transactions, 2,145 fraud cases). Natively handles class imbalance via `scale_pos_weight`; captures non-linear interactions between time, distance and merchant category |

## Tools Used
- Python — Pandas, NumPy, Scikit-learn, LightGBM, plotnine
- SQL-style feature joins via `pyjanitor`
- Matplotlib for feature importance and precision/recall threshold analysis

## Key Insight
Precision/recall trade-off matters more than ROC-AUC on imbalanced fraud data,
since ROC-AUC can overstate performance by rewarding correct ranking of the
(overwhelming) majority class. A threshold search confirmed the default 0.5
cutoff was already close to optimal for this model.

## Next Steps
- Add stratified k-fold cross-validation to confirm result stability across folds
- Report PR-AUC alongside ROC-AUC
- Try XGBoost / CatBoost as further comparison points
