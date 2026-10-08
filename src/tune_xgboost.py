
from pathlib import Path
import time

import optuna
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

from preprocessing import (
    load_data,
    prepare_features,
    split_data,
    build_preprocessor,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"

RANDOM_STATE = 42
N_TRIALS = 30
ORIGINAL_ACCURACY = 0.8901


def objective(trial, X_train_processed, y_train, X_val_processed, y_val):
    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators",
            300,
            1000,
        ),
        "max_depth": trial.suggest_int(
            "max_depth",
            3,
            10,
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.15,
            log=True,
        ),
        "min_child_weight": trial.suggest_int(
            "min_child_weight",
            1,
            10,
        ),
        "subsample": trial.suggest_float(
            "subsample",
            0.70,
            1.0,
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.70,
            1.0,
        ),
        "gamma": trial.suggest_float(
            "gamma",
            0.0,
            0.5,
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha",
            1e-4,
            1.0,
            log=True,
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            0.5,
            5.0,
            log=True,
        ),
    }

    model = XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X_train_processed,
        y_train,
    )

    predictions = model.predict(
        X_val_processed
    )

    accuracy = accuracy_score(
        y_val,
        predictions,
    )

    print(
        f"Trial {trial.number:02d} | "
        f"Accuracy: {accuracy:.4%}"
    )

    return accuracy


def main():
    df = load_data(DATA_PATH)
    X, y = prepare_features(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X,
        y,
    )

    print(f"Training samples:   {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")
    print(f"Test samples:       {len(X_test):,}")

    preprocessor = build_preprocessor()

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    print(f"\nProcessed training shape: {X_train_processed.shape}")
    print(f"Processed validation shape: {X_val_processed.shape}")

    study = optuna.create_study(
        direction="maximize"
    )

    print(f"\nStarting Optuna optimization ({N_TRIALS} trials)...")

    start_time = time.time()

    study.optimize(
        lambda trial: objective(
            trial,
            X_train_processed,
            y_train,
            X_val_processed,
            y_val,
        ),
        n_trials=N_TRIALS,
    )

    elapsed_time = time.time() - start_time

    print("\nXGBOOST TUNING RESULTS")
    print(f"Best validation accuracy: {study.best_value:.4%}")

    print("\nBest parameters:")
    for parameter, value in study.best_params.items():
        print(f"  {parameter}: {value}")

    print(f"\nTotal tuning time: {elapsed_time / 60:.2f} minutes")

    improvement = (
        study.best_value - ORIGINAL_ACCURACY
    ) * 100

    print("\nACCURACY COMPARISON")
    print(f"Original XGBoost: {ORIGINAL_ACCURACY:.2%}")
    print(f"Tuned XGBoost:    {study.best_value:.2%}")
    print(f"Improvement:      {improvement:+.3f} percentage points")

    if study.best_value >= 0.90:
        print("\nValidation accuracy target of 90% achieved.")
    else:
        remaining = (
            0.90 - study.best_value
        ) * 100

        print(
            f"\nBest validation accuracy is "
            f"{remaining:.3f} percentage points below 90%."
        )


if __name__ == "__main__":
    main()
