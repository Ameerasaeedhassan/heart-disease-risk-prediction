
from pathlib import Path
import time

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression

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
N_SPLITS = 3


def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=-1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        ),
        "XGBoost": XGBClassifier(
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
        ),
    }


def main():
    df = load_data(DATA_PATH)
    X, y = prepare_features(df)

    X_train, X_val, X_test, y_train, y_val, y_test = (
        split_data(X, y)
    )

    print(f"Training records:   {len(X_train):,}")
    print(f"Validation records: {len(X_val):,}")
    print(f"Test records:       {len(X_test):,}")
    print(f"CV folds:           {N_SPLITS}")

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    all_results = []
    fold_results = []

    for model_name, classifier in get_models().items():
        print(f"\nRunning cross-validation: {model_name}")

        # Fit preprocessing independently within each fold to prevent data leakage.
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ])

        start = time.time()

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
            error_score="raise",
        )

        elapsed = time.time() - start

        for fold_idx in range(N_SPLITS):
            row = {
                "Model": model_name,
                "Fold": fold_idx + 1,
            }

            for metric in scoring:
                value = scores[f"test_{metric}"][fold_idx]
                row[metric] = value
                print(
                    f"Fold {fold_idx + 1} | "
                    f"{metric}: {value:.4%}"
                )

            fold_results.append(row)

        summary = {
            "Model": model_name,
            "Training Time (min)": elapsed / 60,
        }

        for metric in scoring:
            values = scores[f"test_{metric}"]

            summary[f"{metric} Mean"] = np.mean(values)
            summary[f"{metric} Std"] = np.std(
                values, ddof=1
            )

        all_results.append(summary)

        print(
            f"\n{model_name} completed in "
            f"{elapsed / 60:.2f} minutes"
        )

    results_df = pd.DataFrame(all_results)
    folds_df = pd.DataFrame(fold_results)

    results_df = results_df.sort_values(
        "accuracy Mean",
        ascending=False,
    )

    summary_path = (
        REPORTS_DIR / "cross_validation_summary.csv"
    )
    folds_path = (
        REPORTS_DIR / "cross_validation_folds.csv"
    )

    results_df.to_csv(summary_path, index=False)
    folds_df.to_csv(folds_path, index=False)

    display_df = results_df.copy()

    for metric in scoring:
        display_df[f"{metric} Mean"] *= 100
        display_df[f"{metric} Std"] *= 100

    columns = [
        "Model",
        "accuracy Mean",
        "accuracy Std",
        "f1 Mean",
        "roc_auc Mean",
        "Training Time (min)",
    ]

    print("\nCROSS-VALIDATION SUMMARY")

    print(
        display_df[columns].round(4).to_string(
            index=False
        )
    )

    print(f"\nSaved summary: {summary_path}")
    print(f"Saved fold results: {folds_path}")


if __name__ == "__main__":
    main()
