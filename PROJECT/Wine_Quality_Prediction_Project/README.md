# Wine Quality Prediction Using Decision Tree, Random Forest, and Gradient Boosting

## Advanced version

This upgraded project pushes the original Wine Quality Prediction system further **without changing its core algorithm family**. The production system remains centered on:

1. Decision Tree
2. Random Forest
3. Gradient Boosting

The original-style baseline models are retained, while tuned versions of the same three algorithms are used for the production predictor.

## What was upgraded

- 3-fold stratified cross-validation during model selection
- Randomized hyperparameter search for each of the same three models (5 trials per model)
- F1-focused optimization for the binary Premium/Standard target
- Held-out test-set evaluation
- Accuracy, Precision, Recall, F1, Balanced Accuracy, ROC-AUC and PR-AUC
- Random Forest OOB capability retained where supported by the selected estimator
- Native tree feature importance
- Permutation feature importance
- Training-only out-of-fold probability analysis
- Weighted three-model consensus prediction
- Training-only threshold selection for the consensus decision
- Model agreement and probability/confidence display in Streamlit
- Baseline-vs-tuned comparison for academic transparency
- Machine-readable training metadata

## Target

The target is kept exactly in line with the supplied project material:

- Standard Quality: quality < 6.5
- Premium Quality: quality >= 6.5

## Run on Windows

```cmd
cd /d "%USERPROFILE%\Documents\ML sem-4\Wine_Quality_Prediction_Project"
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
python train_model.py
python -m streamlit run app.py
```

Then open `http://localhost:8501`.

## Important academic point

The project does **not** add XGBoost, LightGBM, neural networks, SVM, KNN or other unrelated predictive algorithms. The upgrade is achieved by improving training, validation, tuning, evaluation, explainability and consensus around the three algorithms specified in the project title.

## Output folders

- `models/` — baseline, tuned and consensus models
- `outputs/` — metrics, plots, feature importance and training metadata
- `notebooks/` — notebook version
- `docs/` — report/viva/project documentation
- `reference_materials/` — supplied project materials


## VS Code / Windows

1. Open the project folder in VS Code.
2. Create/activate the virtual environment: `python -m venv venv` then `venv\Scripts\activate`.
3. Install dependencies: `python -m pip install -r requirements.txt`.
4. Run the EDA notebook: `notebooks/Wine_Quality_EDA.ipynb` and select the project venv kernel.
5. Train the three models: `python train_model.py`.
6. Start the website: `python -m streamlit run app.py`.
7. Open `http://localhost:8501`.

The project remains centered on Decision Tree, Random Forest, and Gradient Boosting; the advanced layer adds tuning, cross-validation, evaluation, feature importance, and three-model consensus without introducing a different primary algorithm.
