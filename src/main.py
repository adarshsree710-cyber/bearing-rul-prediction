"""
Main script for Bearing RUL Prediction using CNN
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from data_loader import load_bearing_data, create_dataset, calculate_time_intervals
from preprocessing import (
    normalize_data,
    scale_rul_labels,
    split_data,
    augment_data,
    shuffle_data,
    save_preprocessing_artifacts,
)
from model import create_cnn_model, check_gpu_availability
from train import train_model, analyze_overfitting
from predict import evaluate_model, convert_predictions_to_hours, plot_predictions, print_prediction_samples

def main():
    """Main training and evaluation pipeline"""

    project_root = Path(__file__).resolve().parent.parent

    # Configuration
    DATA_PATH = project_root / "data" / "ims" / "1st_test"
    WINDOW_SIZE = 2048
    STRIDE = 512
    TEST_SIZE = 0.2
    EPOCHS = 50
    BATCH_SIZE = 64
    NUM_AUGMENTATIONS = 1
    MODEL_SAVE_PATH = project_root / "models" / "cnn_rul_model.keras"
    PREPROCESSING_ARTIFACTS_PATH = project_root / "models" / "preprocessing_artifacts.pkl"
    PLOT_SAVE_PATH = project_root / "outputs" / "plots" / "predictions_vs_actual_rul_hours.png"

    total_steps = 12
    print(f"Pipeline started. Total steps: {total_steps}")
    print(f"Project root: {project_root}")
    print("Loading bearing data...")
    signals, files = load_bearing_data(str(DATA_PATH))

    print(f"Step 1/{total_steps} complete: bearing data loaded")
    print("Creating dataset...")
    X, y = create_dataset(signals, WINDOW_SIZE, STRIDE)

    print(f"Step 2/{total_steps} complete: dataset created")
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=TEST_SIZE)

    print(f"Step 3/{total_steps} complete: data split")
    print("Normalizing data...")
    X_train, signal_mean, signal_std = normalize_data(X_train, return_stats=True)
    X_test = normalize_data(X_test, mean=signal_mean, std=signal_std)

    print(f"Step 4/{total_steps} complete: data normalized")
    print("Scaling labels...")
    y_train_scaled, y_test_scaled, scaler_y = scale_rul_labels(y_train, y_test)
    y_max = y_train.max()  # For reference
    print(f"Max RUL: {y_max}")

    print(f"Step 5/{total_steps} complete: labels scaled")
    print("Shuffling training data...")
    X_train, y_train_scaled = shuffle_data(X_train, y_train_scaled)

    print(f"Step 6/{total_steps} complete: training data shuffled")
    print("Augmenting training data...")
    X_train_aug, y_train_aug = augment_data(
        X_train,
        y_train_scaled,
        num_augmentations=NUM_AUGMENTATIONS
    )
    print(f"Augmented dataset size: {X_train_aug.shape[0]} (from {X_train.shape[0]})")

    print(f"Step 7/{total_steps} complete: training data augmented")
    print("Creating model...")
    model = create_cnn_model(input_shape=(WINDOW_SIZE, 1))

    print(f"Step 8/{total_steps} complete: model created")
    print("Checking GPU availability...")
    gpus = check_gpu_availability()
    print(f"Available GPUs: {gpus}")

    print(f"Step 9/{total_steps} complete: gpu check complete")
    print("Training model...")
    history = train_model(
        model,
        X_train_aug,
        y_train_aug,
        X_test,
        y_test_scaled,
        EPOCHS,
        BATCH_SIZE
    )

    print(f"Step 10/{total_steps} complete: model trained")
    print("Analyzing overfitting...")
    analysis = analyze_overfitting(history)
    print(f"Final train loss: {analysis['final_train_loss']:.4f}")
    print(f"Final val loss: {analysis['final_val_loss']:.4f}")
    print(f"Loss gap: {analysis['loss_gap']:.4f}")

    print(f"Step 11/{total_steps} complete: analysis complete")
    print("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test_scaled, scaler_y)
    print(f"Test MAE (original units): {metrics['mae_original']:.4f}")
    print(f"Test RMSE (original units): {metrics['rmse_original']:.4f}")
    print(f"Test RUL range: {metrics['rul_range']:.4f}")
    print(f"Normalized MAE: {metrics['normalized_mae_percent']:.2f}% of test RUL range")
    print(f"Normalized RMSE: {metrics['normalized_rmse_percent']:.2f}% of test RUL range")
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

    print("Converting predictions to hours...")
    avg_interval_hours = calculate_time_intervals(files)
    y_pred_hours, y_test_hours = convert_predictions_to_hours(
        metrics['y_pred_original'], metrics['y_test_original'], avg_interval_hours
    )

    print_prediction_samples(y_pred_hours, y_test_hours, unit="hours")
    print(f"Mean predicted RUL: {y_pred_hours.mean():.2f} hours")
    print(f"Mean actual RUL: {y_test_hours.mean():.2f} hours")

    plot_predictions(y_test_hours, y_pred_hours,
                    title="Model Predictions vs Actual RUL (Hours)",
                    xlabel="Actual RUL (hours)",
                    ylabel="Predicted RUL (hours)",
                    output_path=PLOT_SAVE_PATH)

    print(f"Saving model to {MODEL_SAVE_PATH}...")
    from predict import save_model
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, str(MODEL_SAVE_PATH))
    save_preprocessing_artifacts(
        str(PREPROCESSING_ARTIFACTS_PATH),
        signal_mean,
        signal_std,
        scaler_y,
        WINDOW_SIZE,
        STRIDE,
        avg_interval_hours
    )
    print(f"Preprocessing artifacts saved to {PREPROCESSING_ARTIFACTS_PATH}")

    print(f"Step 12/{total_steps} complete: model saved")
    print("Training complete!")

if __name__ == "__main__":
    main()
