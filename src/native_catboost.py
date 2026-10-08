
from pathlib import Path
import time

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from preprocessing import (
    load_data,
    prepare_features,
    split_data,
    CATEGORICAL_FEATURES,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"


def main():
    df = load_data(DATA_PATH)
    X, y = prepare_features(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y
    )

    print(f"Training:   {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Test:       {X_test.shape}")

    X_train = X_train.copy()
    X_val = X_val.copy()

    for column in CATEGORICAL_FEATURES:
        X_train[column] = X_train[column].astype(str)
        X_val[column] = X_val[column].astype(str)

    model = CatBoostClassifier(
        iterations=700,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=False,
        thread_count=-1,
    )

    print("\nTraining native CatBoost...")

    start_time = time.time()

    model.fit(
        X_train,
        y_train,
        cat_features=CATEGORICAL_FEATURES,
        eval_set=(X_val, y_val),
        early_stopping_rounds=80,
        verbose=False,
    )

    elapsed = time.time() - start_time

    predictions = model.predict(X_val).astype(int).ravel()
    probabilities = model.predict_proba(X_val)[:, 1]

    accuracy = accuracy_score(y_val, predictions)
    precision = precision_score(y_val, predictions)
    recall = recall_score(y_val, predictions)
    f1 = f1_score(y_val, predictions)
    auc = roc_auc_score(y_val, probabilities)

    print("\nNATIVE CATBOOST VALIDATION RESULTS")

    print(f"Accuracy:  {accuracy:.4%}")
    print(f"Precision: {precision:.4%}")
    print(f"Recall:    {recall:.4%}")
    print(f"F1 Score:  {f1:.4%}")
    print(f"ROC-AUC:   {auc:.4%}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, predictions))

    print(f"\nBest iteration: {model.best_iteration_}")
    print(f"Training time:  {elapsed / 60:.2f} minutes")

    previous_results = {
        "Logistic Regression": 0.8874,
        "HistGradientBoosting": 0.8897,
        "LightGBM": 0.8899,
        "CatBoost (One-Hot)": 0.8900,
        "XGBoost": 0.8901,
        "CatBoost (Native)": accuracy,
    }

    comparison = pd.DataFrame(
        previous_results.items(),
        columns=["Model", "Accuracy"],
    )

    comparison = comparison.sort_values(
        "Accuracy",
        ascending=False,
    )

    comparison["Accuracy (%)"] = (
        comparison["Accuracy"] * 100
    ).round(4)

    print("\nACCURACY COMPARISON")
    print(
        comparison[["Model", "Accuracy (%)"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
