# =============================================================================
# Exercise 2 – Classification
# Course  : Data Mining
# Degree  : M.Sc. High Integrity Systems
# Author  : Mobeen Anwar
# =============================================================================

# ─── TOPIC OVERVIEW ───────────────────────────────────────────────────────────
#
#  Classification is a supervised machine learning task.
#  We train a model on labelled data, then use it to predict the label of
#  unseen data.
#
#  In this exercise we use the classic Iris dataset, which contains
#  measurements of 150 flowers from three species:
#       Iris setosa  |  Iris versicolor  |  Iris virginica
#
#  Goal: given four flower measurements, correctly predict the species.
#
# ─── MODELS COVERED ───────────────────────────────────────────────────────────
#   1. Decision Tree Classifier
#   2. Random Forest Classifier
#
# ─── EVALUATION METRICS COVERED ───────────────────────────────────────────────
#   - Accuracy Score
#   - Confusion Matrix
#   - Classification Report (Precision, Recall, F1-Score)
#
# ─── BONUS ────────────────────────────────────────────────────────────────────
#   - PCA (Principal Component Analysis) for dimensionality reduction
#
# ==============================================================================


# ─── STEP 1: IMPORT LIBRARIES AND LOAD THE DATASET ────────────────────────────

from sklearn.datasets import load_iris
import pandas as pd

# Load the built-in Iris dataset from scikit-learn
iris = load_iris()


# ─── STEP 2: BUILD A DATAFRAME ────────────────────────────────────────────────

# Create a DataFrame from the feature measurements (sepal/petal length & width)
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names     # column names from the dataset itself
)

# Add the target column – initially stored as integers (0, 1, 2)
df["species"] = iris.target

# Check the shape: should be (150, 5) → 150 rows, 5 columns
print("Dataset shape:", df.shape)

# Preview the first five rows
print(df.head())

# See what the three class labels look like in raw numeric form
print("Target class integers:", iris.target_names)


# ─── STEP 3: REPLACE NUMERIC LABELS WITH ACTUAL FLOWER NAMES ─────────────────

# Mapping 0 → 'setosa', 1 → 'versicolor', 2 → 'virginica'
# This makes predictions and reports much easier to read
df["species"] = iris.target_names[iris.target]

print(df.head(10))


# ─── STEP 4: DEFINE FEATURES (X) AND TARGET (y) ──────────────────────────────

# X contains all input columns (the four flower measurements)
X = df.drop("species", axis=1)

# y contains only what we want to predict
y = df["species"]


# ─── STEP 5: SPLIT DATA INTO TRAINING SET AND TEST SET ────────────────────────

from sklearn.model_selection import train_test_split

# 80 % of the data goes to training, 20 % is kept for testing
# random_state=42 ensures the same split every time we run the code
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Quick sanity check – shuffle the full dataframe and inspect
df_shuffled = df.sample(frac=1, random_state=42)
print(df_shuffled.head())


# ─── STEP 6: TRAIN A DECISION TREE CLASSIFIER ────────────────────────────────

from sklearn.tree import DecisionTreeClassifier

# Instantiate the model (no hyperparameters set → uses scikit-learn defaults)
model = DecisionTreeClassifier()

# Train the model on the training data
model.fit(X_train, y_train)

# Peek at the last five rows of the original dataset
print(df.tail())

# Generate predictions on the test set
predictions = model.predict(X_test)
print("Decision Tree predictions:", predictions)


# ─── STEP 7: PREDICT A COMPLETELY NEW FLOWER ─────────────────────────────────

# Imagine someone hands you a flower with these measurements:
#   sepal length = 5.7 cm
#   sepal width  = 3.8 cm
#   petal length = 1.7 cm
#   petal width  = 0.3 cm
new_flower = [[5.7, 3.8, 1.7, 0.3]]

prediction = model.predict(new_flower)
print("Predicted species for new flower:", prediction)


# ─── STEP 8: MEASURE ACCURACY ─────────────────────────────────────────────────

from sklearn.metrics import accuracy_score

# Compare model predictions to the actual labels in the test set
accuracy = accuracy_score(y_test, predictions)
print("Decision Tree accuracy:", accuracy)


# ─── STEP 9: TRAIN A RANDOM FOREST CLASSIFIER ────────────────────────────────

# A Random Forest builds many decision trees and combines their votes.
# It is generally more robust and accurate than a single tree.

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

pred_rf = rf.predict(X_test)

print("Random Forest accuracy:", accuracy_score(y_test, pred_rf))


# ─── STEP 10: CONFUSION MATRIX ───────────────────────────────────────────────

# A confusion matrix shows exactly where the model made mistakes.
# Rows = actual class  |  Columns = predicted class

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predictions)
print("\nConfusion Matrix (raw):\n", cm)

# Display it as a labelled DataFrame so species names appear
cm_df = pd.DataFrame(
    cm,
    index=iris.target_names,      # actual labels (rows)
    columns=iris.target_names     # predicted labels (columns)
)
print("\nConfusion Matrix (labelled):\n", cm_df)


# ─── STEP 11: CLASSIFICATION REPORT ──────────────────────────────────────────

# Precision  – of all predictions for class X, how many were correct?
# Recall     – of all real examples of class X, how many did we catch?
# F1-Score   – harmonic mean of Precision and Recall (balanced metric)
# Support    – number of real examples for each class in the test set

from sklearn.metrics import classification_report

print("\nClassification Report:\n")
print(classification_report(y_test, predictions))


# ─── BONUS: PCA – PRINCIPAL COMPONENT ANALYSIS ───────────────────────────────

# PCA reduces many features down to fewer "principal components"
# while keeping as much variance (information) as possible.
# Here we compress the 4 features into 2 components.

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

# fit_transform learns the PCA axes and projects the data at the same time
X_pca = pca.fit_transform(X)

# How much of the original variance each component captures
print("\nExplained variance ratio per component:", pca.explained_variance_ratio_)

# The actual directions (loadings) in original feature space
print("\nPCA components (loadings):\n", pca.components_)

# Column names of the original features (for reference)
print("\nOriginal feature names:", list(X.columns))
