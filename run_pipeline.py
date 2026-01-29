from src.data_loader import load_data
from src.preprocess import clean_data, split_features_target, train_test_scaling
from src.feature_engineering import add_features
from src.train import train_model
from src.evaluate import evaluate_model


def main():
    print("Starting Fraud Detection Pipeline\n")

    # 1. Load data
    df = load_data()
    print("Data loaded")

    # 2. Clean data
    df = clean_data(df)
    print("Data cleaned")

    # 3. Feature engineering
    df = add_features(df)
    print("Features engineered")

    # 4. Split
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_scaling(X, y)
    print("Train-test split done")

    # 5. Train
    model, cv_f1 = train_model(X_train, y_train)
    print(f"Model trained | CV F1 Score: {cv_f1:.3f}")

    # 6. Evaluate
    print("\nModel Evaluation:")
    evaluate_model(model, X_test, y_test)

    print("\nPipeline executed successfully!")


if __name__ == "__main__":
    main()