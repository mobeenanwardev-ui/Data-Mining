# Data Mining – Exercise Solutions

**Author:** Mobeen Anwar  
**Degree:** M.Sc. High Integrity Systems  
**University:** Frankfurt University of Applied Sciences  
**Semester:** 2nd Semester (Summer 2026)

---

## What is this repo?

This repository contains my personal solution files for the Data Mining course.
Each file is written by me after working through the concepts in class.
Well-commented and structured so that anyone reading can follow step by step.

---

## Files

```
Data-Mining/
│
├── Exercise1_Python_Basics_Solutions.py
│   Python fundamentals for data mining.
│   Topics: shell commands, numpy, pandas, mutability, list comprehension,
│            assert statements, custom functions (add_values, cumulative_sum,
│            is_palindrome, find_the_a).
│
├── Exercise2_Classification_Solutions.py
│   Supervised Learning – Classification on the Iris dataset.
│   Models: Decision Tree, Random Forest.
│   Metrics: Accuracy, Confusion Matrix, Classification Report.
│   Bonus: PCA (dimensionality reduction).
│
├── KMeans_Clustering.py
│   Unsupervised Learning – Clustering on the Mall Customers dataset.
│   Algorithm: K-Means.
│   Topics: scatter plots, cluster labelling, cluster profiling with groupby.
│
├── Linear_Regression.py
│   Supervised Learning – Regression on the AI4I Predictive Maintenance dataset.
│   Model: Linear Regression.
│   Target: predict Tool wear [min] from machine sensor readings.
│   Metrics: MAE, R² Score.
│   Bonus: correlation heatmap with seaborn.
│
└── Data_Preprocessing.py
    Data cleaning and preparation on the AI4I dataset.
    Topics: null handling (dropna / fillna), renaming columns, type casting,
            outlier filtering, feature engineering, correlation analysis,
            groupby analysis.
```

---

## Quick Concept Reference

| Topic | Type | Algorithm / Method |
|---|---|---|
| Classification | Supervised | Decision Tree, Random Forest |
| Regression | Supervised | Linear Regression |
| Clustering | Unsupervised | K-Means |
| Preprocessing | - | fillna, dropna, astype, groupby, corr |

**Classification vs Regression** — Classification predicts a category (e.g. flower species). Regression predicts a number (e.g. how many minutes a tool will last).

**Supervised vs Unsupervised** — Supervised means the training data has labels. Unsupervised means there are no labels; the algorithm finds structure on its own.

---

## Datasets Used

| Dataset | Source | Used In |
|---|---|---|
| Iris | `sklearn.datasets` | Classification |
| Mall Customers | Kaggle | KMeans Clustering |
| AI4I 2020 Predictive Maintenance | UCI / Kaggle | Linear Regression, Preprocessing |

---

## How to Run

```bash
git clone https://github.com/mobeenanwardev-ui/Data-Mining.git
cd Data-Mining
pip install scikit-learn pandas numpy matplotlib seaborn
python KMeans_Clustering.py
```

---

## Contact

GitHub: [@mobeenanwardev-ui](https://github.com/mobeenanwardev-ui)
