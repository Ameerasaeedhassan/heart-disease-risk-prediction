
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)

from preprocessing import (
    load_data,
    prepare_features,
    split_data,
)


ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "raw" / "train.csv"
MODEL_PATH = ROOT / "models" / "heartai_xgboost_pipeline.joblib"

REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.50


def main():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "The previously trained final model was not found:\n"
            f"{MODEL_PATH}\n"
            "Do not retrain yet. Check your models folder."
        )

    model = joblib.load(MODEL_PATH)

    df = load_data(DATA_PATH)
    X, y = prepare_features(df)

    _, _, X_test, _, _, y_test = split_data(X, y)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "average_precision": average_precision_score(
            y_test, probabilities
        ),
    }

    expected_accuracy = 0.888455

    if not np.isclose(
        metrics["accuracy"],
        expected_accuracy,
        atol=0.000001,
    ):
        raise RuntimeError(
            "Accuracy differs from the original final "
            "test result. Stop and investigate."
        )

    print("\nFINAL TEST RESULTS")

    for metric, value in metrics.items():
        print(f"{metric.upper():20s}: {value:.4%}")

    print("\nCLASSIFICATION REPORT")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Absence", "Presence"],
            digits=4,
        )
    )

    cm = confusion_matrix(y_test, predictions)

    fig, ax = plt.subplots(figsize=(7, 6))

    image = ax.imshow(cm, cmap="Reds")

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color="black",
            )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["Absence", "Presence"])
    ax.set_yticklabels(["Absence", "Presence"])

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ax.set_title(
        "HeartAI — Final Test Confusion Matrix",
        fontweight="bold",
    )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "final_confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_test, probabilities)

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(
        fpr,
        tpr,
        color="#9B2335",
        linewidth=2,
        label=f"XGBoost (AUC={metrics['roc_auc']:.4f})",
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="#17365D",
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("HeartAI — Final Test ROC Curve")
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "final_roc_curve.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    precision_values, recall_values, _ = (
        precision_recall_curve(y_test, probabilities)
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(
        recall_values,
        precision_values,
        color="#9B2335",
        linewidth=2,
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(
        "HeartAI — Final Test Precision-Recall Curve"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "final_precision_recall_curve.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    report = {
        "model": "XGBoost",
        "threshold": THRESHOLD,
        "training_rows": 535500,
        "test_rows": int(len(y_test)),
        "metrics": {
            key: float(value)
            for key, value in metrics.items()
        },
        "confusion_matrix": cm.tolist(),
        "synthetic_dataset": True,
        "clinical_use": False,
    }

    with open(
        REPORT_DIR / "final_test_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, indent=4)

    pd.DataFrame({
        "Actual": y_test.to_numpy(),
        "Predicted": predictions,
        "Probability": probabilities,
    }).to_csv(
        REPORT_DIR / "final_test_predictions.csv",
        index=False,
    )

    print("\nCONFUSION MATRIX")
    print(cm)

    print("\nSaved all reports and figures.")
    print("The model was not retrained.")


if __name__ == "__main__":
    main()
