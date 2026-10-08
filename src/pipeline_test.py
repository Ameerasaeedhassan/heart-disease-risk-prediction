
from pathlib import Path

from preprocessing import (
    load_data,
    prepare_features,
    split_data,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"


def main():
    df = load_data(DATA_PATH)

    print(f"Original shape: {df.shape}")

    X, y = prepare_features(df)

    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape:   {y.shape}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y
    )

    print("\nData Split")
    print(f"Training:   {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Testing:    {X_test.shape}")

    print("\nTarget Distribution")
    print(f"Train positive rate:      {y_train.mean():.2%}")
    print(f"Validation positive rate: {y_val.mean():.2%}")
    print(f"Test positive rate:       {y_test.mean():.2%}")


if __name__ == "__main__":
    main()
