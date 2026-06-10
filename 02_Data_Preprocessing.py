# =============================================================================
# Data_Preprocessing.py
# Course  : Data Mining
# Degree  : M.Sc. High Integrity Systems
# Author  : Mobeen Anwar
# =============================================================================

# ─── TOPIC OVERVIEW ───────────────────────────────────────────────────────────
#
#  Before feeding data into any machine learning model, it almost always
#  needs to be cleaned and prepared. This step is called Data Preprocessing.
#
#  Real-world datasets have problems:
#       - Missing values (nulls)
#       - Wrong data types
#       - Duplicate columns
#       - Outliers
#       - Columns that need renaming or transforming
#
#  In this exercise we work with the AI4I 2020 Predictive Maintenance dataset
#  and practice all of the above.
#
# ─── TOPICS COVERED ───────────────────────────────────────────────────────────
#   - Loading and inspecting a dataset
#   - Handling missing values (dropna vs fillna)
#   - Renaming columns
#   - Changing data types
#   - Removing outliers with filtering
#   - Feature engineering (creating a new column from existing ones)
#   - Correlation analysis + heatmap
#   - GroupBy analysis
#
# =============================================================================


# ─── STEP 1: LOAD AND INSPECT THE DATASET ─────────────────────────────────────

import pandas as pd

df = pd.read_csv('ai4i2020.csv')

print(df.head())

# How many rows and columns?
print("Shape:", df.shape)

# Column names
print("Columns:", df.columns.tolist())

# Data types and non-null counts
print(df.info())

# Descriptive statistics
print(df.describe())

# Check for missing values in every column
print("Null counts:\n", df.isnull().sum())


# ─── STEP 2: SIMULATE AND HANDLE MISSING VALUES ───────────────────────────────

# Introducing two null values manually so we can practice handling them
df.loc[10, "Torque [Nm]"] = None
df.loc[20, "Torque [Nm]"] = None

print("Null counts after introducing nulls:\n", df.isnull().sum())

# Option A: Drop all rows that contain any null value
dropVal = df.dropna()
# Note: dropVal is a separate variable – df itself is unchanged here

# Option B: Fill nulls with the median of the column
# Median is preferred over mean when there are outliers
df["Torque [Nm]"] = df["Torque [Nm]"].fillna(
    df["Torque [Nm]"].median()
)

print(df.head(21))


# ─── STEP 3: RENAME A COLUMN ──────────────────────────────────────────────────

# Renaming to a shorter name for easier access throughout the code
df.rename(columns={"Torque [Nm]": "Torque"}, inplace=True)

print("Columns after rename:", df.columns.tolist())
print(df.head())


# ─── STEP 4: CHANGE DATA TYPE ─────────────────────────────────────────────────

# Converting Torque from float to int (removes decimal places)
df["Torque"] = df["Torque"].astype(int)

print(df.head())


# ─── STEP 5: REMOVE DUPLICATE COLUMNS ────────────────────────────────────────

# If any column names are duplicated, this removes the extra copies
df = df.loc[:, ~df.columns.duplicated()]

print(df.head())
print(df.info())


# ─── STEP 6: HANDLE MISSING VALUES AGAIN (PRACTICE ROUND) ────────────────────

# Introducing nulls again – this time filling with the mean instead of median
df.loc[3,  "Torque"] = None
df.loc[10, "Torque"] = None

print("Null counts:\n", df.isnull().sum())

df["Torque"] = df["Torque"].fillna(
    df["Torque"].mean()
)

print(df.head(21))


# ─── STEP 7: CHECK MAX VALUE AND FILTER OUTLIERS ──────────────────────────────

print("Max Torque:", df["Torque"].max())

# Keep only rows where Torque is below 50 to remove extreme outliers
df = df[df["Torque"] < 50]

MinValue  = df["Torque"].min()
MaxValue  = df["Torque"].max()

print("Min Torque after filtering:", MinValue)
print("Max Torque after filtering:", MaxValue)


# ─── STEP 8: CORRELATION MATRIX ───────────────────────────────────────────────

# A correlation matrix shows how strongly each pair of features is related.
# Values close to 1 or -1 = strong relationship.
# Values close to 0 = weak or no relationship.

corr = df.corr(numeric_only=True)
print("Correlation matrix:\n", corr)


# ─── STEP 9: FEATURE ENGINEERING – CREATE A NEW COLUMN ───────────────────────

# Instead of using two temperature columns separately, we create one column
# that captures the difference between them.
# This is called Feature Engineering — creating more meaningful inputs.

df["Temp_Diff"] = (
    df["Process temperature [K]"]
    -
    df["Air temperature [K]"]
)

print(df.head())


# ─── STEP 10: CORRELATION HEATMAP ─────────────────────────────────────────────

import seaborn as sns
import matplotlib.pyplot as plt

corr = df.corr(numeric_only=True)

sns.heatmap(corr)
plt.title("Correlation Heatmap")
plt.show()


# ─── STEP 11: ANALYSE MACHINE FAILURES ────────────────────────────────────────

# Look at all rows where a machine failure actually occurred
print("Failed machine records:\n")
print(df[df["Machine failure"] == 1])


# ─── STEP 12: GROUPBY ANALYSIS ────────────────────────────────────────────────

# What is the average Torque for machines that failed vs those that didn't?
# This is useful for understanding which conditions lead to failure.

print("Average Torque by failure status:\n")
print(df.groupby("Machine failure")["Torque"].mean())
