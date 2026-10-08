
from pathlib import Path
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from preprocessing import (
    load_data,
    prepare_features,
    split_data,
    CONTINUOUS_FEATURES,
    CATEGORICAL_FEATURES,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"

NEW_NUMERIC_FEATURES = [
    "Age_x_MaxHR",
    "STDep_x_ExerciseAngina",
]

NEW_CATEGORICAL_FEATURES = [
    "ChestPain_x_Thallium",
    "Age_Group",
]


def create_features(X):
    X = X.copy()

    X["Age_x_MaxHR"] = X["Age"] * X["Max HR"]

    X["STDep_x_ExerciseAngina"] = (
        X["ST depression"] * X["Exercise angina"]
    )

    X["ChestPain_x_Thallium"] = (
        X["Chest pain type"].astype(str)
        + "_"
        + X["Thallium"].astype(str)
    )

    X["Age_Group"] = pd.cut(
        X["Age"],
        bins=[-float("inf"), 35, 45, 55, 65, float("inf")],
        labels=["Under36", "36to45", "46to55", "56to65", "Over65"],
    ).astype(str)

    return X


def build_feature_preprocessor():
    numeric_features = (
        CONTINUOUS_FEATURES + NEW_NUMERIC_FEATURES
    )

    categorical_features = (
        CATEGORICAL_FEATURES + NEW_CATEGORICAL_FEATURES
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    ("scaler", StandardScaler())
                ]),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline([
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=True,
                        ),
                    )
                ]),
                categorical_features,
            ),
        ]
    )


def evaluate_model(name, model, X_val, y_val):
    predictions = model.predict(X_val)
    probabilities = model.predict_proba(X_val)[:, 1]

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_val, predictions),
        "Precision": precision_score(y_val, predictions),
        "Recall": recall_score(y_val, predictions),
        "F1": f1_score(y_val, predictions),
        "ROC-AUC": roc_auc_score(y_val, probabilities),
    }


def train_experiment(name, X_train, X_val, y_train, y_val,
                     preprocessor):
    print(f"\nTraining {name}...")

    model = Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])

    model.fit(X_train, y_train)

    return evaluate_model(
        name, model, X_val, y_val
    )


def main():
    df = load_data(DATA_PATH)
    X, y = prepare_features(df)

    X_train, X_val, X_test, y_train, y_val, y_test = (
        split_data(X, y)
    )

    print(f"Training:   {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Test:       {X_test.shape}")

    original_preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                CONTINUOUS_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    original_result = train_experiment(
        "Original XGBoost",
        X_train,
        X_val,
        y_train,
        y_val,
        original_preprocessor,
    )

    X_train_fe = create_features(X_train)
    X_val_fe = create_features(X_val)

    print("\nEngineered features:")
    for feature in (
        NEW_NUMERIC_FEATURES + NEW_CATEGORICAL_FEATURES
    ):
        print(f"  + {feature}")

    engineered_result = train_experiment(
        "Feature-Engineered XGBoost",
        X_train_fe,
        X_val_fe,
        y_train,
        y_val,
        build_feature_preprocessor(),
    )

    results = pd.DataFrame(
        [original_result, engineered_result]
    )

    for column in [
        "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"
    ]:
        results[column] = (
            results[column] * 100
        ).round(4)

    print("\nFEATURE ENGINEERING RESULTS")
    print(results.to_string(index=False))

    improvement = (
        engineered_result["Accuracy"]
        - original_result["Accuracy"]
    ) * 100

    print(
        f"\nAccuracy difference: "
        f"{improvement:+.4f} percentage points"
    )

    if improvement > 0:
        print("Engineered features improved accuracy.")
    elif improvement < 0:
        print("Original features performed better.")
    else:
        print("Both versions achieved equal accuracy.")


if __name__ == "__main__":
    main()
