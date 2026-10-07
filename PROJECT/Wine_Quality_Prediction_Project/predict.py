from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.model_utils import FEATURES

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"


def main():
    print("\nAdvanced Wine Quality Prediction")
    print("Core algorithms: Decision Tree, Random Forest, Gradient Boosting")
    print("Enter the 11 features in this order:\n")
    print(", ".join(FEATURES))
    values = [float(x.strip()) for x in input("\nValues: ").split(",")]
    if len(values) != len(FEATURES):
        raise ValueError(f"Expected {len(FEATURES)} values.")
    X = pd.DataFrame([values], columns=FEATURES)

    bundle = joblib.load(MODEL_DIR / "consensus.joblib")
    models = bundle["models"]
    weights = np.array(bundle["weights"])
    threshold = float(bundle["threshold"])

    probs = []
    for name, model in models.items():
        p = float(model.predict_proba(X)[0, 1])
        probs.append(p)
        print(f"{name}: {'Premium Quality' if p >= 0.5 else 'Standard Quality'} | Premium probability = {p:.3f}")

    consensus_prob = float(np.dot(probs, weights))
    consensus = "Premium Quality" if consensus_prob >= threshold else "Standard Quality"
    print(f"\nWeighted 3-model consensus: {consensus}")
    print(f"Consensus probability: {consensus_prob:.3f}")
    print(f"Decision threshold: {threshold:.2f}")


if __name__ == "__main__":
    main()
