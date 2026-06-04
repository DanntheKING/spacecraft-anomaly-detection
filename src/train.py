import argparse
import joblib
import mlflow
import mlflow.sklearn

from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score

from preprocess import load_channel_data

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="/Data/raw/")
    parser.add_argument("--channel_id", type=str, default="P-1")
    parser.add_argument("--contamination", type=float, default=0.15)
    parser.add_argument("--model_dir", type=str, default="models")

    args = parser.parse_args()

    X_train, X_test, y_true, scaler = load_channel_data(
        args.data_dir,
        args.channel_id
    )

    model = IsolationForest(
        contamination=args.contamination,
        random_state=42
    )

    model.fit(X_train)

    preds = model.predict(X_test)
    preds = (preds == -1).astype(int)

    precision = precision_score(y_true, preds)
    recall = recall_score(y_true, preds)
    f1 = f1_score(y_true, preds)

    mlflow.log_param("channel_id", args.channel_id)
    mlflow.log_param("model_type", "IsolationForest")
    mlflow.log_param("contamination", args.contamination)

    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)

    joblib.dump(model, f"{args.model_dir}/isolation_forest.joblib")
    joblib.dump(scaler, f"{args.model_dir}/scaler.joblib")

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)


if __name__ == "__main__":
    main()
