import argparse
from pathlib import Path

from tensorflow.keras.models import load_model

from data_loader import load_bearing_data, create_dataset, calculate_time_intervals
from infer import rebuild_preprocessing_artifacts
from predict import evaluate_model, convert_predictions_to_hours, print_prediction_samples
from preprocessing import load_preprocessing_artifacts, split_data


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved bearing RUL model without retraining."
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to the trained Keras model (.keras). Defaults to models/cnn_rul_model.keras",
    )
    parser.add_argument(
        "--artifacts-path",
        default=None,
        help="Path to preprocessing artifact file. Defaults to models/preprocessing_artifacts.pkl",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="IMS dataset directory used for evaluation. Defaults to data/ims/1st_test",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split ratio used during training. Default: 0.2",
    )
    parser.add_argument(
        "--classification-threshold",
        type=float,
        default=10,
        help="Threshold for binary failure-soon evaluation. Default: 10",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    model_path = Path(args.model_path) if args.model_path else project_root / "models" / "cnn_rul_model.keras"
    artifacts_path = (
        Path(args.artifacts_path)
        if args.artifacts_path
        else project_root / "models" / "preprocessing_artifacts.pkl"
    )
    data_path = Path(args.data_path) if args.data_path else project_root / "data" / "ims" / "1st_test"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {data_path}")

    if artifacts_path.exists():
        print(f"Loading preprocessing artifacts from {artifacts_path}")
        artifacts = load_preprocessing_artifacts(artifacts_path)
    else:
        artifacts = rebuild_preprocessing_artifacts(
            artifacts_path,
            data_path,
            test_size=args.test_size,
        )

    print(f"Loading evaluation dataset from {data_path}")
    signals, files = load_bearing_data(str(data_path))
    X, y = create_dataset(
        signals,
        window_size=artifacts["window_size"],
        stride=artifacts["stride"],
    )
    X = ((X - artifacts["signal_mean"]) / (artifacts["signal_std"] + 1e-8)).astype("float32")
    _, X_test, _, y_test = split_data(X, y, test_size=args.test_size)
    y_test_scaled = artifacts["scaler_y"].transform(y_test.reshape(-1, 1)).flatten().astype("float32")

    print(f"Loading model from {model_path}")
    model = load_model(model_path)

    metrics = evaluate_model(
        model,
        X_test,
        y_test_scaled,
        artifacts["scaler_y"],
        classification_threshold=args.classification_threshold,
    )
    print("\n========== SAVED MODEL EVALUATION ==========")
    print(f"Test MAE (original units): {metrics['mae_original']:.4f}")
    print(f"Test RMSE (original units): {metrics['rmse_original']:.4f}")
    print(
        f"Classification threshold (RUL <= {metrics['classification_threshold']})"
        " is treated as failure-soon"
    )
    print(f"Binary accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Actual failure-soon samples: {metrics['true_positive_samples']}")
    print(f"Predicted failure-soon samples: {metrics['predicted_positive_samples']}")

    avg_interval_hours = calculate_time_intervals(files)
    y_pred_hours, y_test_hours = convert_predictions_to_hours(
        metrics["y_pred_original"],
        metrics["y_test_original"],
        avg_interval_hours,
    )
    print_prediction_samples(y_pred_hours, y_test_hours, unit="hours")


if __name__ == "__main__":
    main()
