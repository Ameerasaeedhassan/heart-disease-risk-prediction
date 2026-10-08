
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


CONTINUOUS_FEATURES = [
    "Age",
    "BP",
    "Cholesterol",
    "Max HR",
    "ST depression",
]

CATEGORICAL_FEATURES = [
    "Sex",
    "Chest pain type",
    "FBS over 120",
    "EKG results",
    "Exercise angina",
    "Slope of ST",
    "Number of vessels fluro",
    "Thallium",
]


def load_data(filepath):
    """Load the raw heart disease dataset."""
    return pd.read_csv(filepath)


def prepare_features(df):
    """Prepare the feature matrix and binary target."""

    df = df.copy()

    df = df.drop(columns=["id"])

    df["Heart Disease"] = df["Heart Disease"].map({
        "Absence": 0,
        "Presence": 1,
    })

    X = df.drop(columns=["Heart Disease"])
    y = df["Heart Disease"]

    return X, y


def build_preprocessor():
    """Create preprocessing pipelines for numerical and categorical features."""

    numerical_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        )
    ])

    preprocessor = ColumnTransformer([
        (
            "numerical",
            numerical_pipeline,
            CONTINUOUS_FEATURES,
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES,
        ),
    ])

    return preprocessor


def split_data(X, y, random_state=42):
    """Create stratified training (70%), validation (15%), and test (15%) sets."""

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=random_state,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=random_state,
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )
