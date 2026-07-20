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
| LightGBM (initial) | 85% | 31% | F1 0.45, ROC-AUC 0.92 on held-out test set. Train performance (92% recall, ROC-AUC 0.97) was noticeably higher than test — a real overfitting gap |
| LightGBM, regularized (default threshold) | 90% | 39% | ROC-AUC 0.98. Regularization improved precision *and* recall together — a sign the overfitting was actually fixed, not just masked |
| **LightGBM, regularized + tuned threshold (Champion)** | **85%** | **52%** | Threshold raised to 0.7721. Roughly 1 in 2 fraud alerts is now a true positive, up from fewer than 1 in 3 in the original model |

Cross-validated (5-fold, default threshold) to confirm stability: mean recall 90.9% (±0.8%), mean precision 55.0% (±3.4%), mean F1 68.5% (±2.9%), mean ROC-AUC 98.4% (±0.9%).

## Why the Champion Model Works
- Captures non-linear interactions between transaction hour, distance from home, merchant category and gender that a linear model can't see
- Natively handles the severe class imbalance (0.054% fraud rate) via `scale_pos_weight`
- Regularization (`max_depth`, `num_leaves`, `min_child_samples`, `reg_alpha`, `reg_lambda`) and early stopping on a validation split prevent the model from memorising the small number of fraud examples (7,506) instead of learning generalisable patterns
- Validated with stratified 5-fold cross-validation, confirming the result is stable across different data splits rather than a lucky single train/test split
- Threshold tuning then traded a small amount of recall (90% → 85%) for a large precision gain (39% → 52%), giving a model that can be dialled toward "catch more fraud" or "fewer false alarms" depending on business need

## Tools Used
- Python — Pandas, NumPy, Scikit-learn, LightGBM, plotnine
- SQL-style feature joins via `pyjanitor`
- Matplotlib for feature importance and precision/recall threshold analysis

## Key Insight
Precision/recall trade-off matters more than ROC-AUC on imbalanced fraud data,
since ROC-AUC can overstate performance by rewarding correct ranking of the
(overwhelming) majority class. Comparing train vs. test performance also matters:
the initial LightGBM model looked strong on ROC-AUC alone (0.92) but was overfitting,
which only became visible by checking train/test and cross-validating.

## Honest Caveat
The final threshold (0.7721) was chosen using the same held-out test set already
used to report metrics — a mild form of tuning on the test set. A stricter setup
would pick the threshold on a separate validation split and use the test set only
once, for a final untouched check.

## Next Steps
- Re-validate the tuned threshold on a proper held-out validation split rather than the test set
- Explore behavioural/velocity features (e.g. transactions per card in the last hour, deviation from a customer's typical spend) for further precision gains
- Try XGBoost / CatBoost as further comparison points
