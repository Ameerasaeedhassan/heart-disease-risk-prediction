
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_PATH = PROJECT_ROOT / "reports" / "figures"


def add_labels(bars, ax):
    for bar in bars:
        value = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.15,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )


def main():
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)

    models = [
        "Logistic\nRegression",
        "Random\nForest",
        "HistGradient\nBoosting",
        "XGBoost",
        "LightGBM",
        "CatBoost",
    ]

    accuracy = np.array([
        0.8874,
        0.8812,
        0.8897,
        0.8901,
        0.8899,
        0.8900,
    ])

    f1 = np.array([
        0.8727,
        0.8666,
        0.8759,
        0.8764,
        0.8763,
        0.8762,
    ])

    roc_auc = np.array([
        0.9534,
        0.9472,
        0.9554,
        0.9557,
        0.9557,
        0.9557,
    ])

    fig = plt.figure(figsize=(14, 11))

    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.4, 1],
        hspace=0.35,
    )

    ax1 = fig.add_subplot(grid[0])

    x = np.arange(len(models))
    width = 0.23

    bars_accuracy = ax1.bar(
        x - width,
        accuracy * 100,
        width,
        label="Accuracy",
        color="#0B1F3A",
    )

    bars_f1 = ax1.bar(
        x,
        f1 * 100,
        width,
        label="F1 Score",
        color="#B85C5C",
    )

    bars_auc = ax1.bar(
        x + width,
        roc_auc * 100,
        width,
        label="ROC-AUC",
        color="#700000",
    )

    ax1.set_title(
        "Model Performance Comparison",
        fontsize=17,
        fontweight="bold",
        pad=15,
    )

    ax1.set_ylabel("Score (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.set_ylim(84, 98)

    ax1.legend(
        ncol=3,
        loc="upper center",
    )

    ax1.grid(
        axis="y",
        alpha=0.2,
    )

    add_labels(bars_accuracy, ax1)
    add_labels(bars_f1, ax1)
    add_labels(bars_auc, ax1)

    ax2 = fig.add_subplot(grid[1])

    ranking_names = [
        "XGBoost",
        "CatBoost",
        "LightGBM",
        "HistGradientBoosting",
        "Logistic Regression",
        "Random Forest",
    ]

    ranking_accuracy = [
        89.01,
        89.00,
        88.99,
        88.97,
        88.74,
        88.12,
    ]

    colors = [
        "#700000",
        "#963333",
        "#AD5555",
        "#C47777",
        "#0B1F3A",
        "#40546C",
    ]

    bars = ax2.barh(
        ranking_names,
        ranking_accuracy,
        color=colors,
    )

    ax2.invert_yaxis()

    ax2.set_title(
        "Validation Accuracy Ranking",
        fontsize=15,
        fontweight="bold",
        pad=12,
    )

    ax2.set_xlabel("Accuracy (%)")
    ax2.set_xlim(87.5, 89.5)

    ax2.grid(
        axis="x",
        alpha=0.2,
    )

    for bar, value in zip(bars, ranking_accuracy):
        ax2.text(
            value + 0.03,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}%",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax2.text(
        89.01,
        -0.55,
        "Best Accuracy",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color="#700000",
    )

    plt.subplots_adjust(
        top=0.92,
        bottom=0.08,
        left=0.12,
        right=0.96,
    )

    output_path = (
        FIGURES_PATH / "model_performance_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
