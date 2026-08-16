# Project 2: Data Classification Using AI

## Overview

This project implements a basic AI classification model using a small dataset.

The project follows the required workflow:

1. Load and understand a dataset
2. Split the data into training and testing sets
3. Apply a simple classification algorithm
4. Train the model
5. Evaluate the model
6. Classify a new data sample

## Dataset

The project uses the **Iris dataset**, which contains measurements of iris flowers.

The model classifies samples into three classes:

- setosa
- versicolor
- virginica

## Algorithm

The project uses **Logistic Regression**, a supervised machine-learning classification algorithm.

Feature scaling is performed using `StandardScaler`.

## Technologies

- Python
- Scikit-learn
- Supervised Learning
- Logistic Regression

## Installation

Install the required dependency:

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Expected Output

The program displays:

- Dataset information
- Training/testing sample counts
- Model accuracy
- Classification report
- Prediction for a new sample

## Learning Objectives

This project demonstrates:

- Basic data handling
- Training and testing data
- Supervised learning
- Model training
- Classification
- Model evaluation
- Making predictions with a trained model

## Project Structure

```text
data_classification_project/
├── main.py
├── requirements.txt
└── README.md
```
