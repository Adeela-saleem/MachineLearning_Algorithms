
# 📊 Regression Models — Complete Reference & Project Repository

This repository contains hands-on implementations of various **Regression algorithms** in Machine Learning, covering everything from basic Linear Regression to advanced ensemble methods, along with proper evaluation, comparison, and deployment where applicable.

---

## 📌 What is Regression?

Regression is a **supervised learning** technique used when the target variable (output) is **continuous/numerical** — e.g., predicting price, salary, temperature, marks, etc. (as opposed to Classification, where the output is categorical).

---

## 🗂️ Repository Structure

```
Regression-Models/
│
├── 01_Linear_Regression/
├── 02_Ridge_Lasso_ElasticNet/
├── 03_Polynomial_Regression/
├── 04_Decision_Tree_Regressor/
├── 05_Random_Forest_Regressor/
├── 06_SVR/
├── 07_KNN_Regressor/
├── 08_Gradient_Boosting_XGBoost/
├── datasets/
├── utils/
└── README.md
```

Each folder contains:
- `notebook.ipynb` — EDA + Model building + Evaluation
- `model.pkl` (if deployed)
- `app.py` (if Streamlit deployed)
- `README.md` (mini explanation specific to that model)

---

## 🧠 Regression Models Covered (When to Use Which)

### 1. **Linear Regression**
- **What it is:** Fits a straight line to the data: `y = mx + c` (or with multiple features: `y = b0 + b1x1 + b2x2 + ...`)
- **When to use:** When the relationship between features and target is **linear**, and the dataset isn't too complex.
- **Limitation:** Doesn't perform well with outliers or non-linear data.

### 2. **Ridge Regression (L2 Regularization)**
- **What it is:** Linear Regression + penalty on large coefficients (squared magnitude).
- **When to use:** When there's **multicollinearity** (correlated features) or risk of overfitting.
- **Effect:** Shrinks coefficients but doesn't make them exactly zero.

### 3. **Lasso Regression (L1 Regularization)**
- **What it is:** Linear Regression + penalty on the absolute value of coefficients.
- **When to use:** When you also need **feature selection** — Lasso can shrink irrelevant feature coefficients to exactly zero.

### 4. **ElasticNet Regression**
- **What it is:** A combination of Ridge + Lasso penalties.
- **When to use:** When you're unsure whether to use Ridge or Lasso — ElasticNet balances both, especially useful with many correlated features.

### 5. **Polynomial Regression**
- **What it is:** An extension of Linear Regression that models non-linear/curved relationships by adding polynomial terms (x², x³, etc.)
- **When to use:** When the scatter plot shows a clear **curved/non-linear pattern**.
- **Caution:** High-degree polynomials can easily overfit.

### 6. **Decision Tree Regressor**
- **What it is:** Splits data into branches using rules/conditions (like an if-else tree).
- **When to use:** When the data has non-linear relationships and interpretability is also needed.
- **Limitation:** Prone to overfitting on its own.

### 7. **Random Forest Regressor**
- **What it is:** An ensemble (bagging) of multiple Decision Trees — averages predictions across all trees.
- **When to use:** When you need strong, stable performance and want to reduce overfitting (e.g., as used in the Laptop Price Predictor).
- **Advantage:** Robust to outliers and handles non-linear data well.

### 8. **Support Vector Regressor (SVR)**
- **What it is:** The regression version of SVM — tries to fit as many points as possible within a "margin."
- **When to use:** When the dataset is small and has high-dimensional features.
- **Limitation:** Slow on large datasets.

### 9. **K-Nearest Neighbors (KNN) Regressor**
- **What it is:** Predicts a point's value based on the average value of its "k" nearest neighbors.
- **When to use:** When the data has clear local patterns and the dataset isn't too large.
- **Limitation:** Slow on high-dimensional/large datasets and sensitive to feature scaling.

### 10. **Gradient Boosting / XGBoost / LightGBM**
- **What it is:** A boosting technique — trees are built sequentially, where each new tree corrects the errors of the previous one.
- **When to use:** When you need the **best possible accuracy** (competitions, production-level models). Very commonly used in industry.
- **Limitation:** Tuning is more complex, and training takes longer.

---

## ⚙️ Preprocessing Steps (Common for All Models)

1. **Handle Missing Values** — mean/median/mode imputation or drop
2. **Handle Outliers** — IQR method, capping, log transform
3. **Encoding Categorical Features**
   - Ordinal Encoding (when order matters)
   - One-Hot Encoding (when order doesn't matter)
4. **Feature Scaling**
   - StandardScaler (mean=0, std=1) — needed for linear models, SVR, KNN
   - Tree-based models (Random Forest, XGBoost) don't need scaling
5. **Train-Test Split** — usually 80/20 or 70/30
6. **⚠️ Important Rule:** Fit the scaler/encoder only on **training data**, then use that fitted object to transform the test data (to avoid data leakage)

---

## 📏 Evaluation Metrics for Regression (Complete Explanation)

| Metric | Formula (concept) | What it tells you | When to prefer it |
|---|---|---|---|
| **MAE** (Mean Absolute Error) | Average of `\|actual - predicted\|` | Average error, easy to interpret (same unit as target) | When you want to reduce the effect of outliers |
| **MSE** (Mean Squared Error) | Average of `(actual - predicted)²` | Penalizes large errors more heavily | When large errors are especially costly |
| **RMSE** (Root Mean Squared Error) | `√MSE` | Same as MSE but in the original unit (interpretable) | The most commonly used metric |
| **R² Score (Coefficient of Determination)** | `1 - (SS_residual / SS_total)` | How much variance the model explains (0 to 1, closer to 1 is better) | Overall model performance summary |
| **Adjusted R²** | R² adjusted for number of features | Checks that R² doesn't artificially increase just from adding more features | When comparing models with different numbers of features |
| **MAPE** (Mean Absolute Percentage Error) | Average of `\|(actual-predicted)/actual\|` × 100 | Percentage error — business-friendly | When explaining results as % to stakeholders |

### 🔑 Quick Rule of Thumb:
- **R² close to 1** = good model
- **Lower RMSE/MAE** = better (but only compare within the same dataset/scale)
- Always look at **multiple metrics together** — don't decide based on just one

---

## 📈 Model Selection Workflow (General Approach)

1. Do EDA — understand the data pattern (linear or non-linear)
2. Build a baseline model (simple Linear Regression)
3. Try more complex models (Random Forest, XGBoost)
4. Compare using cross-validation (don't rely on a single train-test split)
5. Optimize the best model with hyperparameter tuning (GridSearchCV/RandomizedSearchCV)
6. Save the final model using pickle/joblib
7. Deploy it (Streamlit/Flask) if it's a portfolio project

---

## 🛠️ Tech Stack Used

- **Python**, **Pandas**, **NumPy**
- **Scikit-learn** (models, pipelines, metrics)
- **Matplotlib / Seaborn** (EDA & visualization)
- **Streamlit** (deployment)

---

## 📬 Connect

Feel free to explore each folder for detailed notebooks. Suggestions/feedback welcome!

---
