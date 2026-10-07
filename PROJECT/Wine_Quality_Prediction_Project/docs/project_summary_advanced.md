# Advanced Project Summary

## Title
Wine Quality Prediction Using Decision Tree, Random Forest, and Gradient Boosting

## Scope
The upgraded system remains faithful to the original project. It does not replace the three core algorithms with XGBoost, neural networks, SVM or other unrelated models. Instead, it improves the three requested algorithms through validation, hyperparameter tuning, evaluation and a weighted consensus layer.

## Target
- Standard Quality: quality < 6.5
- Premium Quality: quality >= 6.5

## Advanced methodology
1. Remove duplicate records.
2. Preserve the 11 physicochemical input features.
3. Use a stratified 80/20 train-test split.
4. Preserve the original baseline Decision Tree, Random Forest and Gradient Boosting configurations.
5. Tune the same three algorithms using randomized hyperparameter search and stratified 3-fold cross-validation.
6. Optimize model selection using F1-score.
7. Evaluate on the untouched test set with Accuracy, Precision, Recall, F1, Balanced Accuracy, ROC-AUC and PR-AUC.
8. Calculate native tree feature importance and permutation importance.
9. Combine the three tuned model probabilities using CV-F1-derived weights.
10. Select the consensus threshold from training-only out-of-fold predictions.
11. Present model agreement, probabilities and explainability in Streamlit.

## Why this is still the same project
The predictive core is still exactly the three algorithms named in the project title. The additional work improves how those models are trained, compared, interpreted and combined rather than introducing a different model family.

## Current generated results
See `outputs/production_model_metrics.csv`, `outputs/consensus_metrics.csv`, and `outputs/run_summary_advanced.json` for the reproducible results generated from the supplied red-wine CSV.
