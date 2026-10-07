# Literature Review Summary

This summary is derived from the supplied `Literature_Review_Wine_Quality_Prediction.xlsx` (26 entries).

## Themes found in the supplied review
- The UCI red/white wine quality datasets are repeatedly used for physicochemical wine-quality prediction.
- Decision Trees, Random Forests, Gradient Boosting, XGBoost and related ensemble methods appear frequently.
- Several reviewed works use binary quality classes, while others retain multiclass or regression formulations.
- Class imbalance is a recurring limitation, particularly for the highest quality scores.
- Some studies explore feature importance, SHAP, cross-validation, hyperparameter search, or ensemble methods.
- Several sources identify limitations around small datasets, restricted wine types/regions, generalisation, and loss of the original quality granularity when binary labels are used.

## Selected entries relevant to this project
1. **Jain, P., Kaushik, A., et al. (2023)** — uses Decision Tree, Random Forest, AdaBoost, Gradient Boost and XGBoost on the UCI red-wine dataset; the review notes limitations around dataset size/feature selection and validation on white wine.
2. **Anonymous Springer authors (2025)** — compares Random Forest, XGBoost, LightGBM, SVC, ensembles and deep learning; the review notes computation cost and dataset-size limitations for deep learning.
3. **Anonymous arXiv preprint (2023)** — studies wine-quality prediction with unbalanced data using Decision Tree, Random Forest, SVM, Gradient Boosting and KNN; the review highlights possible bias from class imbalance.
4. **Dahal et al. (2021)** — compares regression and ensemble methods on the UCI red-wine dataset; the review notes the red-wine-only scope and regression framing.
5. **Liu, J. (2024)** — applies Random Forest with GridSearchCV, stratified cross-validation and balanced class weights; the review notes continued minority-class underrepresentation.

## Gap addressed by this project
The project provides a compact, reproducible comparison of three tree-based classifiers—Decision Tree, Random Forest and Gradient Boosting—using the same preprocessing, split, and evaluation metrics. It also exposes feature importance and confusion matrices for interpretability.
