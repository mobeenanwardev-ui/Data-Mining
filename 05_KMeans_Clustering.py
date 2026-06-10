# =============================================================================
# KMeans_Clustering.py
# Course  : Data Mining
# Degree  : M.Sc. High Integrity Systems
# Author  : Mobeen Anwar
# =============================================================================

# ─── TOPIC OVERVIEW ───────────────────────────────────────────────────────────
#
#  Clustering is an Unsupervised Learning technique.
#  Unlike Classification, there are NO labels in the training data.
#  The algorithm finds patterns and groups the data on its own.
#
#  In this exercise we use the Mall Customers dataset.
#  The goal is to group customers based on their income and spending behaviour
#  so a business can target each group with a different strategy.
#
# ─── ALGORITHM COVERED ────────────────────────────────────────────────────────
#   K-Means Clustering
#
# ─── DATASET ──────────────────────────────────────────────────────────────────
#   Mall_Customers.csv
#   Features used:
#       - Annual Income (k$)
#       - Spending Score (1-100)
#
# =============================================================================


# ─── STEP 1: LOAD THE DATASET ─────────────────────────────────────────────────

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Mall_Customers.csv')

# Quick preview of the first 10 rows
print(df.head(10))

# Data types and null check
print(df.info)

# Statistical summary (mean, std, min, max, etc.)
print(df.describe())

# Dataset dimensions (rows, columns)
print("Shape:", df.shape)


# ─── STEP 2: SELECT FEATURES FOR CLUSTERING ───────────────────────────────────

# We only use two features so we can visualise the clusters on a 2D scatter plot
X = df[
    [
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]
]


# ─── STEP 3: VISUALISE THE RAW DATA BEFORE CLUSTERING ─────────────────────────

# This scatter plot shows the data as-is before any grouping
plt.scatter(
    X["Annual Income (k$)"],
    X["Spending Score (1-100)"]
)

plt.xlabel("Income")
plt.ylabel("Spending")
plt.title("Customer Data – Before Clustering")
plt.show()


# ─── STEP 4: APPLY K-MEANS CLUSTERING ─────────────────────────────────────────

from sklearn.cluster import KMeans

# n_clusters=5 means we ask the algorithm to find 5 groups
# random_state=42 ensures we get the same result every time we run this
kmeans = KMeans(
    n_clusters=5,
    random_state=42
)

kmeans.fit(X)

# Each customer is now assigned a cluster label: 0, 1, 2, 3, or 4
print("Cluster labels:", kmeans.labels_)


# ─── STEP 5: ADD CLUSTER LABELS BACK TO THE DATAFRAME ─────────────────────────

df["Cluster"] = kmeans.labels_

print(df.head(10))


# ─── STEP 6: VISUALISE THE CLUSTERS ───────────────────────────────────────────

# Now the scatter plot is colour-coded by cluster — much more informative
plt.scatter(
    X["Annual Income (k$)"],
    X["Spending Score (1-100)"],
    c=df["Cluster"]
)

plt.xlabel("Income")
plt.ylabel("Spending")
plt.title("Customer Segments – After K-Means Clustering")
plt.show()


# ─── STEP 7: EXPLORE A SINGLE CLUSTER ─────────────────────────────────────────

# Let's isolate Cluster 0 and inspect the customers in it
cluster0 = df[df["Cluster"] == 0]
print("Customers in Cluster 0:\n", cluster0)


# ─── STEP 8: HOW MANY CUSTOMERS ARE IN EACH CLUSTER? ──────────────────────────

print("Cluster sizes:\n", df["Cluster"].value_counts())


# ─── STEP 9: AVERAGE STATISTICS PER CLUSTER ───────────────────────────────────

# This tells us the profile of each cluster:
# e.g. Cluster 2 might have high income but low spending → potential target group
print("Cluster profiles (mean values):\n")
print(df.groupby("Cluster").mean(numeric_only=True))
