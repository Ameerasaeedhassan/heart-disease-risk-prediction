
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from preprocessing import (
    load_data,
    prepare_features,
    split_data,
    build_preprocessor
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
FIGURES_PATH = PROJECT_ROOT / "reports" / "figures"

FIGURES_PATH.mkdir(parents=True, exist_ok=True)


def main():

    df = load_data(DATA_PATH)

    X, y = prepare_features(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y
    )

    preprocessor = build_preprocessor()

    model = Pipeline([
        ("preprocessor", preprocessor),

        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_prob)

    print("\n--- Logistic Regression: Validation Results ---")

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    cm = confusion_matrix(y_val, y_pred)

    print("\n--- Confusion Matrix ---")
    print(cm)

    print("\n--- Classification Report ---")

    print(
        classification_report(
            y_val,
            y_pred,
            target_names=["Absence", "Presence"]
        )
    )

    metrics_names = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]

    metrics_values = [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [
        "#D98C8C",
        "#C45A5A",
        "#A52A2A",
        "#8B0000",
        "#600000"
    ]

    bars = ax.bar(
        metrics_names,
        metrics_values,
        color=colors
    )

    ax.set_title(
        "Logistic Regression — Validation Performance",
        fontsize=16,
        fontweight="bold",
        pad=15
    )

    ax.set_ylabel(
        "Score",
        fontsize=11
    )

    ax.set_ylim(0, 1)

    for bar, value in zip(bars, metrics_values):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value * 100:.2f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold"
        )

    ax.grid(
        axis="y",
        alpha=0.2
    )

    plt.tight_layout()

    metrics_path = FIGURES_PATH / "baseline_metrics.png"

    plt.savefig(
        metrics_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(f"Saved: {metrics_path}")

    fig, ax = plt.subplots(figsize=(7, 6))

    image = ax.imshow(
        cm,
        cmap="Reds"
    )

    ax.set_title(
        "Logistic Regression — Confusion Matrix",
        fontsize=16,
        fontweight="bold",
        pad=15
    )

    ax.set_xlabel(
        "Predicted Label",
        fontsize=11
    )

    ax.set_ylabel(
        "Actual Label",
        fontsize=11
    )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels([
        "Absence",
        "Presence"
    ])

    ax.set_yticklabels([
        "Absence",
        "Presence"
    ])

    cell_labels = [
        ["True Negative", "False Positive"],
        ["False Negative", "True Positive"]
    ]

    threshold = cm.max() / 2

    for i in range(2):
        for j in range(2):

            text_color = (
                "white"
                if cm[i, j] > threshold
                else "#0B1F3A"
            )

            ax.text(
                j,
                i,
                f"{cell_labels[i][j]}\n{cm[i, j]:,}",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color=text_color
            )

    colorbar = plt.colorbar(
        image,
        ax=ax
    )

    colorbar.set_label(
        "Number of Predictions"
    )

    plt.tight_layout()

    confusion_path = (
        FIGURES_PATH /
        "baseline_confusion_matrix.png"
    )

    plt.savefig(
        confusion_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(f"Saved: {confusion_path}")


if __name__ == "__main__":
    main()
