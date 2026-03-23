"""
Main script for Bearing RUL Prediction using CNN
"""

import numpy as np
import matplotlib.pyplot as plt

from data_loader import load_bearing_data, create_dataset, calculate_time_intervals
from preprocessing import normalize_data, scale_rul_labels, split_data, augment_data, shuffle_data
from model import create_cnn_model, check_gpu_availability
from train import train_model, analyze_overfitting
from predict import evaluate_model, convert_predictions_to_hours, plot_predictions, print_prediction_samples

def main():
    """Main training and evaluation pipeline"""

    # Configuration
    DATA_PATH = "../data/ims/1st_test"
    WINDOW_SIZE = 2048
    STRIDE = 512
    TEST_SIZE = 0.2
    EPOCHS = 50
    BATCH_SIZE = 64
    MODEL_SAVE_PATH = "../models/cnn_rul_model.h5"

    print("Loading bearing data...")
    signals, files = load_bearing_data(DATA_PATH)

    print("Creating dataset...")
    X, y = create_dataset(signals, WINDOW_SIZE, STRIDE)

    print("Normalizing data...")
    X = normalize_data(X)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=TEST_SIZE)

    print("Scaling labels...")
    y_train_scaled, y_test_scaled, scaler_y = scale_rul_labels(y_train, y_test)
    y_max = y_train.max()  # For reference
    print(f"Max RUL: {y_max}")

    print("Shuffling training data...")
    X_train, y_train = shuffle_data(X_train, y_train)

    print("Augmenting training data...")
    X_train_aug, y_train_aug = augment_data(X_train, y_train, num_augmentations=2)
    print(f"Augmented dataset size: {X_train_aug.shape[0]} (from {X_train.shape[0]})")

    print("Creating model...")
    model = create_cnn_model(input_shape=(WINDOW_SIZE, 1))

    print("Checking GPU availability...")
    gpus = check_gpu_availability()
    print(f"Available GPUs: {gpus}")

    print("Training model...")
    history = train_model(model, X_train_aug, y_train_aug, X_test, y_test, EPOCHS, BATCH_SIZE)

    print("Analyzing overfitting...")
    analysis = analyze_overfitting(history)
    print(f"Final train loss: {analysis['final_train_loss']:.4f}")
    print(f"Final val loss: {analysis['final_val_loss']:.4f}")
    print(f"Loss gap: {analysis['loss_gap']:.4f}")

    print("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test_scaled, scaler_y)
    print(f"Test MAE (original units): {metrics['mae_original']:.4f}")
    print(f"Test RMSE (original units): {metrics['rmse_original']:.4f}")

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
                    ylabel="Predicted RUL (hours)")

    print(f"Saving model to {MODEL_SAVE_PATH}...")
    from predict import save_model
    save_model(model, MODEL_SAVE_PATH)

    print("Training complete!")

if __name__ == "__main__":
    main()