# Data Mining – Exercise Solutions

**Author:** Mobeen Anwar
**Degree:** M.Sc. High Integrity Systems
**University:** Frankfurt University of Applied Sciences
**Semester:** 2nd Semester (Summer 2026)

---

## What is this repo?

This repository contains my personal solution files for the Data Mining course.
Each exercise is written by me after working through the concepts in class.
The files are clean, well-commented, and structured so that anyone reading them
can follow the logic step by step — no prior context needed.

---

## Structure

```
Data-Mining/
│
├── Exercise1_Python_Basics_Solutions.py
│   Python fundamentals needed for data mining work.
│   Topics: shell commands, numpy, pandas, mutability, list comprehension,
│            assert statements, custom functions.
│
└── Exercise2_Classification_Solutions.py
    Supervised machine learning – Classification.
    Topics: Iris dataset, Decision Tree, Random Forest,
            accuracy score, confusion matrix, classification report, PCA.
```

---

## Exercise 2 – Classification

### What is Classification?

Classification is a type of supervised learning.
You have labelled training data, you train a model on it, and then you ask
the model to predict the label of data it has never seen before.

Think of it like teaching a friend to sort fruit by showing them 100 labelled
examples — then handing them a new fruit and asking what it is.

### Dataset: Iris

The Iris dataset is a standard benchmark dataset in machine learning.
It contains measurements (in cm) for 150 flowers from three species:

| Feature          | Description                  |
|------------------|------------------------------|
| sepal length     | Length of the outer petals   |
| sepal width      | Width of the outer petals    |
| petal length     | Length of the inner petals   |
| petal width      | Width of the inner petals    |
| species (target) | setosa / versicolor / virginica |

### Models Used

| Model                   | Idea                                                           |
|-------------------------|----------------------------------------------------------------|
| Decision Tree           | Splits data by asking yes/no questions on features            |
| Random Forest           | Builds many trees, takes majority vote for final prediction   |

### Evaluation Metrics

| Metric               | What it tells you                                             |
|----------------------|---------------------------------------------------------------|
| Accuracy Score       | Percentage of correct predictions overall                    |
| Confusion Matrix     | Exact breakdown of correct and incorrect predictions by class|
| Precision            | Of all predictions for class X, how many were actually X?    |
| Recall               | Of all real X examples, how many did the model find?         |
| F1-Score             | Balance between Precision and Recall                         |

### Bonus – PCA

PCA (Principal Component Analysis) is a dimensionality reduction technique.
It compresses multiple features into fewer "components" while keeping as much
information as possible. In this exercise, four features are reduced to two.

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/mobeenanwardev-ui/Data-Mining.git
   ```

2. Install dependencies:
   ```bash
   pip install scikit-learn pandas numpy
   ```

3. Run any exercise file:
   ```bash
   python Exercise2_Classification_Solutions.py
   ```

---

## Contact

Feel free to connect with me on GitHub: [@mobeenanwardev-ui](https://github.com/mobeenanwardev-ui)
