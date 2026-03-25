from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)

def evaluate_model(model, X_test, y_test_scaled, scaler_y, classification_threshold=10):
    """
    Evaluate model performance.

    Args:
        model: Trained Keras model
        X_test (np.array): Test features
        y_test_scaled (np.array): Scaled test labels
        scaler_y: Scaler for inverse transformation
        classification_threshold (float): RUL threshold used to convert
            regression outputs into binary classes for accuracy, precision,
            recall, and F1 evaluation

    Returns:
        dict: Evaluation metrics
    """
    # Get predictions
    print(f"Evaluating model on {len(X_test)} test samples...")
    test_loss, test_mae = model.evaluate(X_test, y_test_scaled, verbose=0)

    print(f"Generating predictions for {len(X_test)} test samples...")
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    y_pred_original = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_test_original = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

    # Calculate metrics
    mae = mean_absolute_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
    rul_range = float(np.max(y_test_original) - np.min(y_test_original))
    if rul_range > 0:
        normalized_mae_percent = (mae / rul_range) * 100.0
        normalized_rmse_percent = (rmse / rul_range) * 100.0
    else:
        normalized_mae_percent = 0.0
        normalized_rmse_percent = 0.0
    y_true_class = (y_test_original <= classification_threshold).astype(int)
    y_pred_class = (y_pred_original <= classification_threshold).astype(int)

    accuracy = accuracy_score(y_true_class, y_pred_class)
    precision = precision_score(y_true_class, y_pred_class, zero_division=0)
    recall = recall_score(y_true_class, y_pred_class, zero_division=0)
    f1 = f1_score(y_true_class, y_pred_class, zero_division=0)

    return {
        'test_loss': test_loss,
        'test_mae_scaled': test_mae,
        'mae_original': mae,
        'rmse_original': rmse,
        'rul_range': rul_range,
        'normalized_mae_percent': normalized_mae_percent,
        'normalized_rmse_percent': normalized_rmse_percent,
        'classification_threshold': classification_threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positive_samples': int(np.sum(y_true_class)),
        'predicted_positive_samples': int(np.sum(y_pred_class)),
        'y_pred_original': y_pred_original,
        'y_test_original': y_test_original,
        'y_true_class': y_true_class,
        'y_pred_class': y_pred_class,
    }

def convert_predictions_to_hours(y_pred, y_test, avg_interval_hours):
    """
    Convert RUL predictions from file units to hours.

    Args:
        y_pred (np.array): Predicted RUL in files
        y_test (np.array): Actual RUL in files
        avg_interval_hours (float): Average hours per file

    Returns:
        tuple: (y_pred_hours, y_test_hours)
    """
    print(f"Converting predictions to hours using avg_interval_hours={avg_interval_hours:.4f}")
    y_pred_hours = y_pred * avg_interval_hours
    y_test_hours = y_test * avg_interval_hours
    return y_pred_hours, y_test_hours

def plot_predictions(
    y_test,
    y_pred,
    title="Model Predictions vs Actual",
    xlabel="Actual RUL",
    ylabel="Predicted RUL",
    output_path=None
):
    """
    Plot prediction scatter plot.

    Args:
        y_test (np.array): Actual values
        y_pred (np.array): Predicted values
        title (str): Plot title
        xlabel (str): X-axis label
        ylabel (str): Y-axis label
        output_path (str | Path | None): File path to save the plot image
    """
    print(f"Plotting {len(y_test)} prediction points...")
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, s=20)

    # Add perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        print(f"Plot saved to {output_path}")

    plt.close()

def print_prediction_samples(y_pred, y_test, num_samples=5, unit="files"):
    """
    Print sample predictions.

    Args:
        y_pred (np.array): Predicted values
        y_test (np.array): Actual values
        num_samples (int): Number of samples to print
        unit (str): Unit description
    """
    print(f"\n========== PREDICTIONS ({unit}) ==========")
    for i in range(min(num_samples, len(y_pred))):
        print(f"Sample {i+1} - Predicted: {y_pred[i]:.2f} | Actual: {y_test[i]:.2f}")
    print("=" * 50)

def save_model(model, filepath):
    """
    Save trained model.

    Args:
        model: Keras model
        filepath (str): Path to save model
    """
    print(f"Saving model to {filepath}...")
    model.save(filepath)
    print(f"Model saved to {filepath}")
