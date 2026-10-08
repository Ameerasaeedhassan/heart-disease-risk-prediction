
"""Generate SHAP explanations for the trained HeartAI XGBoost model."""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from preprocessing import load_data, prepare_features, split_data


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "raw" / "train.csv"
MODEL_PATH = ROOT / "models" / "heartai_xgboost_pipeline.joblib"
FIGURE_DIR = ROOT / "reports" / "figures"
REPORT_DIR = ROOT / "reports"

SAMPLE_SIZE = 1000
RANDOM_STATE = 42

BURGUNDY = "#9B2335"
NAVY = "#17365D"


def save_figure(filename):
    """Save the current figure using consistent export settings."""
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / filename,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


def main():
    """Explain the final model using a reproducible training sample."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Final trained model not found: {MODEL_PATH}"
        )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    pipeline = joblib.load(MODEL_PATH)

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    df = load_data(DATA_PATH)
    X, y = prepare_features(df)

    X_train, _, _, _, _, _ = split_data(X, y)

    sample = X_train.sample(
        n=min(SAMPLE_SIZE, len(X_train)),
        random_state=RANDOM_STATE,
    )

    transformed = preprocessor.transform(sample)
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    transformed = np.asarray(transformed, dtype=np.float64)

    transformed_df = pd.DataFrame(
        transformed,
        columns=feature_names,
        index=sample.index,
    )

    print(f"Records explained: {len(transformed_df):,}")
    print(f"Encoded features: {len(feature_names)}")

    print("\nCalculating SHAP values...")
    explainer = shap.TreeExplainer(classifier)
    explanation = explainer(transformed_df)

    values = np.asarray(explanation.values)

    if values.ndim == 3:
        values = values[:, :, 1]
        base_values = np.asarray(explanation.base_values)
        if base_values.ndim == 2:
            base_values = base_values[:, 1]
    else:
        base_values = explanation.base_values

    explanation = shap.Explanation(
        values=values,
        base_values=base_values,
        data=transformed_df.to_numpy(),
        feature_names=feature_names.tolist(),
    )

    # Mean absolute SHAP values measure global feature importance.
    importance = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Absolute_SHAP": np.abs(values).mean(axis=0),
    }).sort_values(
        "Mean_Absolute_SHAP",
        ascending=False,
    )

    importance.to_csv(
        REPORT_DIR / "shap_feature_importance.csv",
        index=False,
    )

    print("\nTOP 10 ENCODED FEATURES")
    print(importance.head(10).to_string(index=False))

    top_features = importance.head(15).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        top_features["Feature"],
        top_features["Mean_Absolute_SHAP"],
        color=BURGUNDY,
    )
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_title("HeartAI — SHAP Feature Importance")
    save_figure("shap_feature_importance.png")

    shap.plots.beeswarm(
        explanation,
        max_display=15,
        show=False,
    )
    plt.title("HeartAI — SHAP Summary")
    save_figure("shap_beeswarm.png")

    shap.plots.waterfall(
        explanation[0],
        max_display=15,
        show=False,
    )
    plt.title("HeartAI — Individual Prediction Explanation")
    save_figure("shap_waterfall.png")

    print("\nSaved:")
    print("  reports/shap_feature_importance.csv")
    print("  reports/figures/shap_feature_importance.png")
    print("  reports/figures/shap_beeswarm.png")
    print("  reports/figures/shap_waterfall.png")


if __name__ == "__main__":
    main()
