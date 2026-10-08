
from pathlib import Path
import time
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from preprocessing import (
    load_data,
    prepare_features,
    split_data,
    build_preprocessor
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"


def evaluate_model(
    name,
    model,
    X_train,
    y_train,
    X_val,
    y_val
):

    print(f"\nTraining {name}...")

    start_time = time.time()

    model.fit(
        X_train,
        y_train
    )

    training_time = time.time() - start_time

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    results = {
        "Model": name,
        "Accuracy": accuracy_score(y_val, y_pred),
        "Precision": precision_score(y_val, y_pred),
        "Recall": recall_score(y_val, y_pred),
        "F1": f1_score(y_val, y_pred),
        "ROC-AUC": roc_auc_score(y_val, y_prob),
        "Training Time (s)": training_time
    }

    return results


def main():

    df = load_data(DATA_PATH)

    X, y = prepare_features(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X,
        y
    )

    models = {

        "XGBoost": XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        ),

        "LightGBM": LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        ),

        "CatBoost": CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            random_seed=42,
            verbose=False
        )
    }

    results = []

    for name, classifier in models.items():

        pipeline = Pipeline([
            (
                "preprocessor",
                build_preprocessor()
            ),
            (
                "classifier",
                classifier
            )
        ])

        result = evaluate_model(
            name,
            pipeline,
            X_train,
            y_train,
            X_val,
            y_val
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="ROC-AUC",
        ascending=False
    )

    print("\n" + "=" * 80)
    print("ADVANCED MODEL COMPARISON — VALIDATION SET")
    print("=" * 80)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )


if __name__ == "__main__":
    main()
