from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_predict
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report, precision_recall_curve
)
from sklearn.inspection import permutation_importance

from src.model_utils import (
    load_and_prepare, build_baseline_models, build_tuning_base_models,
    build_tuning_spaces, split_data, FEATURES, QUALITY_THRESHOLD, RANDOM_STATE
)

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "winequality-red.csv"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

MODEL_KEYS = {
    "Decision Tree": "decision_tree",
    "Random Forest": "random_forest",
    "Gradient Boosting": "gradient_boosting",
}


def evaluate_model(name, model, X_test, y_test):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1-Score": f1_score(y_test, pred, zero_division=0),
        "Balanced Accuracy": balanced_accuracy_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, proba),
        "PR-AUC": average_precision_score(y_test, proba),
    }


def save_confusion_matrix(name, model, X_test, y_test, suffix=""):
    pred = model.predict(X_test)
    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Standard", "Premium"],
                yticklabels=["Standard", "Premium"])
    plt.title(f"{name}{suffix} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    filename = f"{MODEL_KEYS[name]}_confusion_matrix{suffix.lower().replace(' ', '_')}.png"
    plt.savefig(OUT_DIR / filename, dpi=180)
    plt.close()


def feature_importance_frame(models, X_test, y_test):
    rows = []
    for name, model in models.items():
        estimator = model.named_steps["model"]
        for feature, importance in zip(FEATURES, estimator.feature_importances_):
            rows.append({"Model": name, "Feature": feature, "Importance": float(importance), "Type": "Native"})

        perm = permutation_importance(
            model, X_test, y_test, scoring="f1", n_repeats=5,
            random_state=RANDOM_STATE, n_jobs=-1
        )
        for feature, importance, std in zip(FEATURES, perm.importances_mean, perm.importances_std):
            rows.append({"Model": name, "Feature": feature, "Importance": float(importance),
                         "Type": "Permutation", "Std": float(std)})
    return pd.DataFrame(rows)


def main():
    print("\n=== WINE QUALITY PREDICTION - ADVANCED TRAINING ===\n")
    df, X, y, data_info = load_and_prepare(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(X, y)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    # 1) Preserve the original three models as baselines.
    baseline_models = build_baseline_models()
    baseline_results = []
    for name, model in baseline_models.items():
        print(f"Training baseline: {name}")
        model.fit(X_train, y_train)
        row = evaluate_model(name, model, X_test, y_test)
        row["Version"] = "Baseline"
        baseline_results.append(row)
        joblib.dump(model, MODEL_DIR / f"{MODEL_KEYS[name]}_baseline.joblib")

    # 2) Tune only the same three model families.
    spaces = build_tuning_spaces()
    bases = build_tuning_base_models()
    tuned_models = {}
    tuned_results = []
    search_summary = {}

    for name in MODEL_KEYS:
        print(f"\nTuning: {name}")
        search = RandomizedSearchCV(
            estimator=bases[name],
            param_distributions=spaces[name],
            n_iter=5,
            scoring={"f1": "f1", "accuracy": "accuracy", "roc_auc": "roc_auc"},
            refit="f1",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=1,
            return_train_score=False,
            verbose=0,
        )
        search.fit(X_train, y_train)
        tuned_models[name] = search.best_estimator_
        row = evaluate_model(name, search.best_estimator_, X_test, y_test)
        row["Version"] = "Tuned"
        row["CV F1 Mean"] = search.best_score_
        row["CV F1 Std"] = search.cv_results_["std_test_f1"][search.best_index_]
        tuned_results.append(row)
        search_summary[name] = {
            "best_params": search.best_params_,
            "best_cv_f1": float(search.best_score_),
            "cv_f1_std": float(search.cv_results_["std_test_f1"][search.best_index_]),
            "search_iterations": 5,
            "cv_folds": 3,
        }
        joblib.dump(search.best_estimator_, MODEL_DIR / f"{MODEL_KEYS[name]}.joblib")
        save_confusion_matrix(name, search.best_estimator_, X_test, y_test, " Tuned")

    baseline_df = pd.DataFrame(baseline_results)
    tuned_df = pd.DataFrame(tuned_results)
    comparison = pd.concat([baseline_df, tuned_df], ignore_index=True)
    comparison.to_csv(OUT_DIR / "model_comparison_advanced.csv", index=False)
    tuned_df.to_csv(OUT_DIR / "production_model_metrics.csv", index=False)

    # Baseline confusion matrices retain the original project outputs too.
    for name, model in baseline_models.items():
        save_confusion_matrix(name, model, X_test, y_test)

    # 3) Out-of-fold probabilities on TRAIN ONLY -> choose a robust consensus threshold.
    weights = np.array([max(search_summary[name]["best_cv_f1"], 0.001) for name in MODEL_KEYS])
    weights = weights / weights.sum()
    oof_probs = []
    for name in MODEL_KEYS:
        p = cross_val_predict(
            tuned_models[name], X_train, y_train, cv=cv,
            method="predict_proba", n_jobs=-1
        )[:, 1]
        oof_probs.append(p)
    oof_probs = np.column_stack(oof_probs)
    weighted_oof = oof_probs @ weights

    thresholds = np.arange(0.25, 0.751, 0.01)
    f1s = [f1_score(y_train, weighted_oof >= t, zero_division=0) for t in thresholds]
    best_threshold = float(thresholds[int(np.argmax(f1s))])

    # Test-set consensus evaluation uses the threshold selected only from training OOF predictions.
    test_probs = np.column_stack([
        tuned_models[name].predict_proba(X_test)[:, 1] for name in MODEL_KEYS
    ])
    weighted_test = test_probs @ weights
    consensus_pred = (weighted_test >= best_threshold).astype(int)
    consensus = {
        "Model": "Weighted 3-Model Consensus",
        "Accuracy": accuracy_score(y_test, consensus_pred),
        "Precision": precision_score(y_test, consensus_pred, zero_division=0),
        "Recall": recall_score(y_test, consensus_pred, zero_division=0),
        "F1-Score": f1_score(y_test, consensus_pred, zero_division=0),
        "Balanced Accuracy": balanced_accuracy_score(y_test, consensus_pred),
        "ROC-AUC": roc_auc_score(y_test, weighted_test),
        "PR-AUC": average_precision_score(y_test, weighted_test),
        "Version": "Consensus",
    }
    pd.DataFrame([consensus]).to_csv(OUT_DIR / "consensus_metrics.csv", index=False)
    joblib.dump({"models": tuned_models, "weights": weights.tolist(), "threshold": best_threshold},
                MODEL_DIR / "consensus.joblib")

    print("Computing feature importance...", flush=True)
    # 4) Feature importance: native + permutation.
    fi_df = feature_importance_frame(tuned_models, X_test, y_test)
    fi_df.to_csv(OUT_DIR / "feature_importance_advanced.csv", index=False)
    native = fi_df[fi_df["Type"] == "Native"].pivot(index="Feature", columns="Model", values="Importance")
    plt.figure(figsize=(10, 7))
    native.sort_values(by="Gradient Boosting", ascending=True).plot(kind="barh", ax=plt.gca())
    plt.title("Native Feature Importance - Tuned Models")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "feature_importance_tuned.png", dpi=180)
    plt.close()

    print("Generating plots...", flush=True)
    # 5) Quality distribution and correlation.
    plt.figure(figsize=(10, 7))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "correlation_heatmap.png", dpi=180)
    plt.close()

    q = df["quality"].value_counts().sort_index()
    plt.figure(figsize=(7, 4))
    plt.bar(q.index.astype(str), q.values)
    plt.title("Original Wine Quality Distribution")
    plt.xlabel("Quality")
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "quality_distribution.png", dpi=180)
    plt.close()

    binary = y.value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    plt.bar(["Standard Quality", "Premium Quality"], binary.values)
    plt.title("Binary Wine Quality Class Distribution")
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "class_distribution.png", dpi=180)
    plt.close()

    # 6) Threshold/F1 curve from train OOF only.
    plt.figure(figsize=(8, 4))
    plt.plot(thresholds, f1s)
    plt.axvline(best_threshold, linestyle="--", label=f"Selected threshold = {best_threshold:.2f}")
    plt.xlabel("Consensus probability threshold")
    plt.ylabel("OOF F1-score")
    plt.title("Training-Only Threshold Selection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "consensus_threshold_selection.png", dpi=180)
    plt.close()

    print("Saving metadata...", flush=True)
    # 7) Save compact machine-readable metadata for the app.
    metadata = {
        **data_info,
        "features": FEATURES,
        "quality_threshold": QUALITY_THRESHOLD,
        "positive_class": "Premium Quality",
        "negative_class": "Standard Quality",
        "test_size": 0.20,
        "random_state": RANDOM_STATE,
        "cv_folds": 3,
        "tuning_iterations_per_model": 30,
        "optimization_metric": "F1",
        "weights": {name: float(w) for name, w in zip(MODEL_KEYS, weights)},
        "consensus_threshold": best_threshold,
        "baseline_results": baseline_results,
        "tuned_results": tuned_results,
        "consensus_result": consensus,
        "search_summary": search_summary,
    }
    (OUT_DIR / "run_summary_advanced.json").write_text(json.dumps(metadata, indent=2))

    print("\n=== BASELINE ===")
    print(baseline_df[["Model", "Accuracy", "Precision", "Recall", "F1-Score"]].round(4).to_string(index=False))
    print("\n=== TUNED PRODUCTION MODELS ===")
    print(tuned_df[["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]].round(4).to_string(index=False))
    print("\n=== 3-MODEL CONSENSUS ===")
    print(pd.DataFrame([consensus])[["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]].round(4).to_string(index=False))
    print(f"\nConsensus threshold selected from training-only OOF predictions: {best_threshold:.2f}")
    print("Training complete. Tuned models, baseline models, consensus model and advanced outputs were saved.")


if __name__ == "__main__":
    main()
