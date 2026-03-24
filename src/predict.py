import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_model(model, X_test, y_test_scaled, scaler_y):
    """
    Evaluate model performance.

    Args:
        model: Trained Keras model
        X_test (np.array): Test features
        y_test_scaled (np.array): Scaled test labels
        scaler_y: Scaler for inverse transformation

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

    return {
        'test_loss': test_loss,
        'test_mae_scaled': test_mae,
        'mae_original': mae,
        'rmse_original': rmse,
        'y_pred_original': y_pred_original,
        'y_test_original': y_test_original
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

def plot_predictions(y_test, y_pred, title="Model Predictions vs Actual", xlabel="Actual RUL", ylabel="Predicted RUL"):
    """
    Plot prediction scatter plot.

    Args:
        y_test (np.array): Actual values
        y_pred (np.array): Predicted values
        title (str): Plot title
        xlabel (str): X-axis label
        ylabel (str): Y-axis label
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
    plt.show()

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
