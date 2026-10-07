# Project Summary

## Title
**Wine Quality Prediction Using Decision Tree, Random Forest, and Gradient Boosting**

## Problem statement
The project predicts wine quality from measurable physicochemical properties. The supplied abstract describes the 11 features and the comparison of three supervised algorithms. The supplied presentation further specifies binary classification and model configurations.

## Input features
fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol

## Target
- `0`: Standard Quality (`quality < 6.5`)
- `1`: Premium Quality (`quality >= 6.5`)

## Dataset and preprocessing
The supplied CSV has 1,599 rows, 11 input features, and one quality target. It contains no missing values and 240 duplicate rows. The project removes duplicate rows before splitting, leaving 1,359 unique rows.

The train/test split is 80/20 with stratification and random state 42. StandardScaler is included in each model pipeline to match the supplied project materials.

## Models
1. Decision Tree — Gini criterion, max depth 6, min samples split 5.
2. Random Forest — 100 trees, sqrt feature subsampling, OOB scoring.
3. Gradient Boosting — 150 estimators, learning rate 0.1, subsample 0.8.

## Actual reproducible test-set results
See `outputs/model_comparison.csv`.

            Model  Accuracy  Precision  Recall  F1-Score
    Decision Tree    0.8787     0.5833  0.3784    0.4590
    Random Forest    0.8971     0.7368  0.3784    0.5000
Gradient Boosting    0.8934     0.6818  0.4054    0.5085

The supplied presentation reports different headline values. Those values are retained as presentation material, but this project records its own generated results rather than hard-coding reported numbers.

## Course alignment
The supplied Machine Learning course handout emphasizes tree-based models, train/test discipline, classification metrics, feature importance, and model evaluation. This project directly exercises those topics.
