# =============================================================================
# Linear_Regression.py
# Course  : Data Mining
# Degree  : M.Sc. High Integrity Systems
# Author  : Mobeen Anwar
# =============================================================================

# ─── TOPIC OVERVIEW ───────────────────────────────────────────────────────────
#
#  Linear Regression is a Supervised Learning algorithm used to predict
#  a continuous numerical value — not a category.
#
#  Interview Question → What is the difference between Classification and Regression?
#  Answer: Classification predicts categories (e.g. "setosa" or "virginica").
#          Regression predicts numbers (e.g. how many minutes a tool will last).
#
#  In this exercise we use the AI4I 2020 Predictive Maintenance dataset.
#  Goal: predict "Tool wear [min]" based on machine sensor readings.
#
# ─── DATASET ──────────────────────────────────────────────────────────────────
#   ai4i2020.csv
#   Features (X):
#       - Air temperature [K]
#       - Process temperature [K]
#       - Rotational speed [rpm]
#       - Torque [Nm]
#   Target (y):
#       - Tool wear [min]
#
# ─── METRICS COVERED ──────────────────────────────────────────────────────────
#   - Mean Absolute Error (MAE)
#   - R² Score
#   - Correlation Matrix + Heatmap (bonus)
#
# =============================================================================


# ─── STEP 1: LOAD THE DATASET ─────────────────────────────────────────────────

import pandas as pd

df = pd.read_csv('ai4i2020.csv')

print(df.head())


# ─── STEP 2: DEFINE FEATURES (X) AND TARGET (y) ───────────────────────────────

# X = the inputs the model will learn from (machine sensor readings)
X = df[
    [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]"
    ]
]

# y = what we want to predict (how worn the tool is after running)
y = df["Tool wear [min]"]


# ─── STEP 3: SPLIT DATA INTO TRAINING AND TEST SETS ───────────────────────────

from sklearn.model_selection import train_test_split

# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ─── STEP 4: TRAIN THE LINEAR REGRESSION MODEL ────────────────────────────────

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)


# ─── STEP 5: MAKE PREDICTIONS ON THE TEST SET ─────────────────────────────────

predictions = model.predict(X_test)

# Preview first 5 predictions vs actual values
print("First 5 predictions:", predictions[:5])
print("First 5 actual values:\n", y_test.head())


# ─── STEP 6: COMPARE ACTUAL vs PREDICTED IN A DATAFRAME ───────────────────────

comparison = pd.DataFrame({
    "Actual":    y_test.values,
    "Predicted": predictions
})

print(comparison.head(10))


# ─── STEP 7: TARGET VARIABLE STATS ────────────────────────────────────────────

# Good to know the range of the target before judging the error
print("Min tool wear :", df["Tool wear [min]"].min())
print("Max tool wear :", df["Tool wear [min]"].max())
print("Mean tool wear:", df["Tool wear [min]"].mean())


# ─── STEP 8: EVALUATE – MEAN ABSOLUTE ERROR (MAE) ────────────────────────────

# MAE = average of how far off the predictions are (in same units as y)
# e.g. MAE of 20 means predictions are off by ~20 minutes on average

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error (MAE):", mae)


# ─── STEP 9: EVALUATE – R² SCORE ─────────────────────────────────────────────

# R² measures how much of the variance in y the model explains.
# R² = 1.0  → perfect model
# R² = 0.0  → model is no better than just guessing the mean
# R² < 0    → model is worse than guessing the mean

from sklearn.metrics import r2_score

r2 = r2_score(y_test, predictions)
print("R² Score:", r2)


# ─── STEP 10: CORRELATION WITH TARGET ─────────────────────────────────────────

# Which features are most correlated with "Tool wear [min]"?
# High correlation → stronger influence on the target
print("\nCorrelation with Tool wear [min]:\n")
print(df.corr(numeric_only=True)["Tool wear [min]"])


# ─── STEP 11: MACHINE FAILURE CLASS DISTRIBUTION ──────────────────────────────

# Just an extra look at the dataset — how many failures vs non-failures
print("\nMachine failure counts:\n", df["Machine failure"].value_counts())


# ─── BONUS: CORRELATION HEATMAP ───────────────────────────────────────────────

# A heatmap makes it easy to spot strong relationships between all features visually

import seaborn as sns
import matplotlib.pyplot as plt

corr_matrix = df.corr(numeric_only=True)

plt.figure(figsize=(10, 8))

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Feature Correlation Heatmap")
plt.show()

# Top features correlated with machine failure
print("\nTop correlations with Machine failure:\n")
print(corr_matrix["Machine failure"].sort_values(ascending=False))
