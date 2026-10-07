from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.model_utils import FEATURES, QUALITY_THRESHOLD

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"

st.set_page_config(page_title="Wine Quality AI Lab", page_icon="🍷", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

/* Main App styling */
.stApp {
    background-color: #0d0e15;
    background-image: 
        radial-gradient(circle at 15% 50%, rgba(139, 0, 0, 0.08), transparent 25%),
        radial-gradient(circle at 85% 30%, rgba(100, 30, 215, 0.08), transparent 25%);
    font-family: 'Outfit', sans-serif;
    color: #e2e8f0;
}

/* Typography Overrides */
h1, h2, h3, h4, h5, h6, p, span, div {
    font-family: 'Outfit', sans-serif !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: rgba(18, 20, 30, 0.6) !important;
    backdrop-filter: blur(12px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

/* Glassmorphism containers */
.block-container {
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* Custom Hero Section */
.hero {
    padding: 3rem 2.5rem;
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(30, 15, 45, 0.8), rgba(65, 20, 45, 0.8));
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255,255,255,0.1);
    backdrop-filter: blur(20px);
    color: white;
    margin-bottom: 2.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    animation: fadeInDown 0.8s ease-out;
}

.hero::before {
    content: "";
    position: absolute;
    top: -50%; left: -50%; width: 200%; height: 200%;
    background: radial-gradient(circle, rgba(255,50,100,0.15) 0%, transparent 50%);
    animation: rotate 20s linear infinite;
    pointer-events: none;
}

.hero h1 {
    font-size: 3.5rem !important;
    font-weight: 800 !important;
    margin-bottom: 0.5rem !important;
    background: linear-gradient(to right, #fff, #ffd700);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0px 4px 20px rgba(255, 215, 0, 0.2);
    letter-spacing: -1px;
}

.hero p {
    font-size: 1.2rem !important;
    font-weight: 300 !important;
    color: #cbd5e1 !important;
    max-width: 600px;
    margin: 0 auto !important;
}

/* Animations */
@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-20px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes rotate {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(255, 51, 102, 0.4); }
    70% { box-shadow: 0 0 0 15px rgba(255, 51, 102, 0); }
    100% { box-shadow: 0 0 0 0 rgba(255, 51, 102, 0); }
}

/* Inputs & Metrics Card Styling */
div[data-testid="metric-container"] {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 1.2rem;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    transition: transform 0.3s ease, background 0.3s ease;
}
div[data-testid="metric-container"]:hover {
    transform: translateY(-5px);
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 215, 0, 0.3);
}

/* Inputs styling */
.stNumberInput > div > div > input {
    background-color: rgba(15, 20, 25, 0.7) !important;
    color: #fff !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    transition: all 0.3s ease;
}
.stNumberInput > div > div > input:focus {
    border-color: #ff3366 !important;
    box-shadow: 0 0 0 2px rgba(255, 51, 102, 0.2) !important;
}

/* Predict Button */
.stButton > button {
    background: linear-gradient(135deg, #ff3366, #990033) !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 1.2rem !important;
    padding: 0.8rem 2rem !important;
    border-radius: 16px !important;
    border: none !important;
    box-shadow: 0 8px 25px rgba(255, 51, 102, 0.3) !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
    animation: pulse 2s infinite;
}
.stButton > button:hover {
    transform: translateY(-3px) scale(1.02) !important;
    box-shadow: 0 12px 30px rgba(255, 51, 102, 0.5) !important;
    background: linear-gradient(135deg, #ff4d79, #b3003b) !important;
}

/* Dataframe & Tables */
.stDataFrame {
    background: rgba(255, 255, 255, 0.02);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    padding: 0.5rem;
}

/* Headers */
h2, h3 {
    color: #fff !important;
    font-weight: 600 !important;
    letter-spacing: -0.5px;
    margin-top: 1.5rem !important;
}

/* Custom Alert/Info boxes */
.stAlert {
    border-radius: 16px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    background-color: rgba(30, 40, 60, 0.5) !important;
    backdrop-filter: blur(10px) !important;
    color: #e2e8f0 !important;
}

/* Custom Progress Bar */
.stProgress > div > div > div > div {
    background-image: linear-gradient(to right, #ff3366, #ffd700) !important;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class='hero'>
    <h1>🍷 Grand Cru AI Predictor</h1>
    <p>Experience the ultimate fusion of data science and viticulture. Powered by an advanced ensemble of Decision Trees, Random Forests, and Gradient Boosting.</p>
</div>
""", unsafe_allow_html=True)

metadata_path = OUT_DIR / "run_summary_advanced.json"
if not metadata_path.exists() or not (MODEL_DIR / "consensus.joblib").exists():
    st.error("Advanced models have not been trained yet. Run: python train_model.py")
    st.stop()

metadata = json.loads(metadata_path.read_text())
production = {
    "Decision Tree": joblib.load(MODEL_DIR / "decision_tree.joblib"),
    "Random Forest": joblib.load(MODEL_DIR / "random_forest.joblib"),
    "Gradient Boosting": joblib.load(MODEL_DIR / "gradient_boosting.joblib"),
}
consensus_bundle = joblib.load(MODEL_DIR / "consensus.joblib")
weights = dict(zip(production.keys(), consensus_bundle["weights"]))
threshold = float(consensus_bundle["threshold"])

st.info(f"Premium Quality = quality ≥ {QUALITY_THRESHOLD}; the system predicts the same binary target as the original project. The production layer remains strictly limited to the three requested tree-based algorithms.")

with st.sidebar:
    st.header("System")
    st.metric("Training rows", metadata["rows_after_duplicate_removal"])
    st.metric("Test rows", metadata["test_size"] and round(metadata["rows_after_duplicate_removal"] * metadata["test_size"]))
    st.metric("CV folds", metadata["cv_folds"])
    st.metric("Tuning trials / model", metadata["tuning_iterations_per_model"])
    st.caption("Optimization target: F1-score")

# Reasonable ranges are broad enough for real inputs while catching obvious typos.
ranges = {
    "fixed acidity": (0.0, 20.0), "volatile acidity": (0.0, 3.0), "citric acid": (0.0, 2.0),
    "residual sugar": (0.0, 70.0), "chlorides": (0.0, 1.0), "free sulfur dioxide": (0.0, 300.0),
    "total sulfur dioxide": (0.0, 500.0), "density": (0.9, 1.1), "pH": (0.0, 14.0),
    "sulphates": (0.0, 3.0), "alcohol": (0.0, 25.0)
}
defaults = {
    "fixed acidity": 7.4, "volatile acidity": 0.70, "citric acid": 0.00,
    "residual sugar": 1.9, "chlorides": 0.076, "free sulfur dioxide": 11.0,
    "total sulfur dioxide": 34.0, "density": 0.9978, "pH": 3.51,
    "sulphates": 0.56, "alcohol": 9.4
}

st.subheader("1. Enter physicochemical properties")
cols = st.columns(3)
values = {}
for i, feature in enumerate(FEATURES):
    lo, hi = ranges[feature]
    with cols[i % 3]:
        values[feature] = st.number_input(
            feature.title(), min_value=float(lo), max_value=float(hi),
            value=float(defaults[feature]), format="%.4f", key=feature
        )

if st.button("🔬 Predict with all 3 models", type="primary", use_container_width=True):
    X = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)
    rows = []
    probs = []
    for name, model in production.items():
        prob = float(model.predict_proba(X)[0, 1])
        pred = int(prob >= 0.5)
        probs.append(prob)
        rows.append({
            "Model": name,
            "Premium probability": f"{prob*100:.1f}%",
            "Prediction": "Premium Quality" if pred else "Standard Quality",
            "CV F1 weight": f"{weights[name]:.3f}"
        })

    weighted_prob = float(np.dot(probs, [weights[n] for n in production]))
    consensus_pred = int(weighted_prob >= threshold)
    agreement = sum(int((p >= 0.5) == bool(consensus_pred)) for p in probs)
    confidence = abs(weighted_prob - 0.5) * 2

    st.subheader("2. Model predictions")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("3-model consensus", "Premium Quality" if consensus_pred else "Standard Quality")
    c2.metric("Consensus probability", f"{weighted_prob*100:.1f}%")
    c3.metric("Model agreement", f"{agreement}/3")

    st.progress(min(max(confidence, 0.0), 1.0), text=f"Decision confidence: {confidence*100:.1f}%")
    if agreement < 3:
        st.warning("The three models disagree. Treat this case as borderline and inspect the individual probabilities rather than relying only on the consensus label.")
    else:
        st.success("All three core models agree on the predicted class.")

    st.subheader("3. What influenced the models?")
    fi_path = OUT_DIR / "feature_importance_advanced.csv"
    if fi_path.exists():
        fi = pd.read_csv(fi_path)
        native = fi[fi["Type"] == "Native"].groupby("Feature")["Importance"].mean().sort_values(ascending=False).head(8)
        st.bar_chart(native)

    st.caption(f"Consensus threshold: {threshold:.2f}. It was selected using training-set cross-validation predictions, not the held-out test set.")

with st.expander("Model performance on the held-out test set"):
    metrics_path = OUT_DIR / "production_model_metrics.csv"
    consensus_path = OUT_DIR / "consensus_metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        if consensus_path.exists():
            metrics = pd.concat([metrics, pd.read_csv(consensus_path)], ignore_index=True)
        st.dataframe(metrics.round(4), use_container_width=True, hide_index=True)

with st.expander("Advanced methodology"):
    st.markdown("""
- **Same three algorithms:** Decision Tree, Random Forest and Gradient Boosting.
- **Baseline + tuning:** the original-style configurations are retained for academic comparison; production models are tuned versions of the same algorithms.
- **5-fold stratified CV:** used during tuning to reduce dependence on one split.
- **F1 optimization:** useful when the Premium class is less frequent than the Standard class.
- **Additional evaluation:** Accuracy, Precision, Recall, F1, Balanced Accuracy, ROC-AUC and PR-AUC.
- **Consensus layer:** weighted probability combination of the three tuned models; no fourth ML algorithm is introduced.
- **Threshold selection:** chosen only from training out-of-fold predictions to avoid using the held-out test set for model decisions.
- **Explainability:** native tree feature importance and permutation importance.
""")
