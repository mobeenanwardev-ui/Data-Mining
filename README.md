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

## Structure

### Foundations

```
01_Python_Basics.py
```
Python fundamentals needed before jumping into machine learning.  
Topics: shell commands, numpy, pandas, mutability, list comprehension, assert statements, custom functions (add_values, cumulative_sum, is_palindrome, find_the_a).

```
02_Data_Preprocessing.py
```
Cleaning and preparing real-world data before feeding it into any model.  
Topics: null handling (dropna / fillna), renaming columns, type casting, outlier filtering, feature engineering, correlation analysis, groupby analysis.  
Dataset: AI4I 2020 Predictive Maintenance

---

### Supervised Learning

> The training data has labels. The model learns the relationship between inputs and outputs.

```
03_Decision_Tree_RandomForest_Classification.py
```
Predicting a **category** from labelled data.  
Models: Decision Tree, Random Forest.  
Metrics: Accuracy, Confusion Matrix, Classification Report (Precision, Recall, F1).  
Bonus: PCA (dimensionality reduction).  
Dataset: Iris

```
04_Linear_Regression.py
```
Predicting a **continuous number** from labelled data.  
Model: Linear Regression.  
Target: Tool wear [min] predicted from machine sensor readings.  
Metrics: MAE (Mean Absolute Error), R² Score.  
Bonus: Correlation heatmap with seaborn.  
Dataset: AI4I 2020 Predictive Maintenance

---

### Unsupervised Learning

> No labels. The algorithm finds hidden patterns and groups in the data on its own.

```
05_KMeans_Clustering.py
```
Grouping customers into segments based on income and spending behaviour.  
Algorithm: K-Means Clustering.  
Topics: scatter plots, cluster labelling, cluster profiling with groupby.  
Dataset: Mall Customers

---

## Quick Concept Reference

| # | File | Type | Algorithm |
|---|---|---|---|
| 01 | Python_Basics | Foundations | — |
| 02 | Data_Preprocessing | Foundations | — |
| 03 | Decision_Tree_RandomForest_Classification | Supervised | Decision Tree, Random Forest |
| 04 | Linear_Regression | Supervised | Linear Regression |
| 05 | KMeans_Clustering | Unsupervised | K-Means |

**Classification vs Regression** — Classification predicts a category (e.g. flower species). Regression predicts a number (e.g. how many minutes a tool will last).  
**Supervised vs Unsupervised** — Supervised means the training data has labels. Unsupervised means no labels; the algorithm finds structure on its own.

---

## Datasets Used

| Dataset | Source | Used In |
|---|---|---|
| Iris | `sklearn.datasets` | 03 – Classification |
| AI4I 2020 Predictive Maintenance | UCI / Kaggle | 02 – Preprocessing, 04 – Regression |
| Mall Customers | Kaggle | 05 – Clustering |

---

## How to Run

```bash
git clone https://github.com/mobeenanwardev-ui/Data-Mining.git
cd Data-Mining
pip install scikit-learn pandas numpy matplotlib seaborn
python 03_Decision_Tree_RandomForest_Classification.py
```

---

## Contact

GitHub: [@mobeenanwardev-ui](https://github.com/mobeenanwardev-ui)
