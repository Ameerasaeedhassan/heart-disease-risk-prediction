
from pathlib import Path
import time

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from preprocessing import (
    load_data,
    prepare_features,
    split_data,
    build_preprocessor,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def make_xgboost():
    return XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def make_lightgbm():
    return LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )


def make_pipeline(classifier):
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", classifier),
    ])


def make_stacking_model():
    estimators = [
        (
            "xgboost",
            make_pipeline(make_xgboost()),
        ),
        (
            "lightgbm",
            make_pipeline(make_lightgbm()),
        ),
        (
            "logistic",
            make_pipeline(
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                )
            ),
        ),
    ]

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        stack_method="predict_proba",
        cv=cv,
        n_jobs=1,
        passthrough=False,
    )


def evaluate(y_true, probabilities, threshold=0.5):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "Accuracy": accuracy_score(
            y_true, predictions
        ),
        "Precision": precision_score(
            y_true, predictions,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_true, predictions,
            zero_division=0,
        ),
        "F1": f1_score(
            y_true, predictions,
            zero_division=0,
        ),
        "ROC-AUC": roc_auc_score(
            y_true, probabilities
        ),
    }


def print_metrics(name, metrics):
    print(f"\n{name}")

    for metric, value in metrics.items():
        print(f"{metric:12s}: {value:.4%}")


def main():
    df = load_data(DATA_PATH)
    X, y = prepare_features(df)

    X_train, X_val, X_test, y_train, y_val, y_test = (
        split_data(X, y)
    )

    print(f"Training:   {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Test:       {X_test.shape}")

    print("\nTraining baseline XGBoost...")

    baseline = make_pipeline(
        make_xgboost()
    )

    start = time.time()

    baseline.fit(X_train, y_train)

    baseline_time = time.time() - start

    baseline_prob = baseline.predict_proba(
        X_val
    )[:, 1]

    baseline_metrics = evaluate(
        y_val, baseline_prob
    )

    print_metrics(
        "XGBoost Baseline",
        baseline_metrics,
    )

    print("\nTraining stacking ensemble...")

    stacking = make_stacking_model()

    start = time.time()

    stacking.fit(X_train, y_train)

    stacking_time = time.time() - start

    stacking_prob = stacking.predict_proba(
        X_val
    )[:, 1]

    stacking_metrics = evaluate(
        y_val, stacking_prob
    )

    print_metrics(
        "Stacking Ensemble",
        stacking_metrics,
    )

    comparison = pd.DataFrame([
        {
            "Model": "XGBoost",
            **baseline_metrics,
        },
        {
            "Model": "Stacking Ensemble",
            **stacking_metrics,
        },
    ])

    display = comparison.copy()

    for col in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
    ]:
        display[col] *= 100

    print("\nMODEL COMPARISON — THRESHOLD 0.50")
    print(
        display.round(4).to_string(
            index=False
        )
    )

    comparison.to_csv(
        REPORTS_DIR / "stacking_comparison.csv",
        index=False,
    )

    threshold_results = []

    for model_name, probabilities in [
        ("XGBoost", baseline_prob),
        ("Stacking Ensemble", stacking_prob),
    ]:
        for threshold in np.arange(
            0.30, 0.701, 0.01
        ):
            metrics = evaluate(
                y_val,
                probabilities,
                threshold=float(threshold),
            )

            threshold_results.append({
                "Model": model_name,
                "Threshold": round(
                    float(threshold), 2
                ),
                **metrics,
            })

    thresholds_df = pd.DataFrame(
        threshold_results
    )

    thresholds_df.to_csv(
        REPORTS_DIR / "threshold_analysis.csv",
        index=False,
    )

    print("\nVALIDATION THRESHOLD ANALYSIS")

    for model_name in [
        "XGBoost",
        "Stacking Ensemble",
    ]:
        subset = thresholds_df[
            thresholds_df["Model"] == model_name
        ]

        best_accuracy = subset.loc[
            subset["Accuracy"].idxmax()
        ]

        best_f1 = subset.loc[
            subset["F1"].idxmax()
        ]

        print(f"\n{model_name}")

        print(
            "Best accuracy threshold: "
            f"{best_accuracy['Threshold']:.2f}"
        )
        print(
            "Accuracy at that threshold: "
            f"{best_accuracy['Accuracy']:.4%}"
        )
        print(
            "Recall at that threshold: "
            f"{best_accuracy['Recall']:.4%}"
        )

        print(
            "Best F1 threshold: "
            f"{best_f1['Threshold']:.2f}"
        )
        print(
            "F1 at that threshold: "
            f"{best_f1['F1']:.4%}"
        )

        default_predictions = (
            (
                baseline_prob
                if model_name == "XGBoost"
                else stacking_prob
            ) >= 0.5
        ).astype(int)

        print(
            "Confusion matrix at 0.50:\n",
            confusion_matrix(
                y_val, default_predictions
            ),
        )

    print("\nTRAINING TIME")
    print(
        f"XGBoost: {baseline_time / 60:.2f} min"
    )
    print(
        f"Stacking: {stacking_time / 60:.2f} min"
    )

    print("\nSaved:")
    print("  reports/stacking_comparison.csv")
    print("  reports/threshold_analysis.csv")

    print(
        "\nThresholds were explored using validation data only."
    )
    print("The final test set was not used for threshold selection.")


if __name__ == "__main__":
    main()
