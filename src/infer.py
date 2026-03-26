import argparse
from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model

from data_loader import create_windows, load_bearing_data, create_dataset, calculate_time_intervals
from preprocessing import (
    apply_normalization,
    load_preprocessing_artifacts,
    normalize_data,
    save_preprocessing_artifacts,
    scale_rul_labels,
    split_data,
)


def load_signal(file_path, bearing_column=4):
    """
    Load one IMS vibration file and extract the configured bearing column.

    Args:
        file_path (str | Path): Path to the IMS file
        bearing_column (int): Zero-based signal column index

    Returns:
        np.array: 1D vibration signal
    """
    data = np.loadtxt(file_path, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D array from {file_path}, got shape {data.shape}")
    if data.shape[1] <= bearing_column:
        raise ValueError(
            f"File {file_path} has {data.shape[1]} columns, cannot read column index {bearing_column}"
        )
    return data[:, bearing_column].astype(np.float32, copy=False)


def prepare_windows(signal, window_size, stride, mean, std):
    """
    Create normalized CNN windows for inference.

    Args:
        signal (np.array): Raw 1D vibration signal
        window_size (int): Sliding window size
        stride (int): Sliding window stride
        mean (float): Training normalization mean
        std (float): Training normalization std

    Returns:
        np.array: Normalized windows with channel dimension
    """
    windows = create_windows(signal, window_size, stride)
    if not windows:
        raise ValueError(
            f"Signal length {len(signal)} is smaller than window_size={window_size}. "
            "Use a longer file or retrain with a smaller window."
        )

    X = np.array(windows, dtype=np.float32)[..., np.newaxis]
    return apply_normalization(X, mean, std)


def summarize_predictions(predictions_files, avg_interval_hours):
    """
    Aggregate per-window predictions into a file-level summary.

    Args:
        predictions_files (np.array): Per-window RUL predictions in file units
        avg_interval_hours (float): Average hours per file

    Returns:
        dict: Summary statistics
    """
    mean_files = float(np.mean(predictions_files))
    median_files = float(np.median(predictions_files))
    min_files = float(np.min(predictions_files))
    max_files = float(np.max(predictions_files))

    return {
        'mean_files': mean_files,
        'median_files': median_files,
        'min_files': min_files,
        'max_files': max_files,
        'mean_hours': mean_files * avg_interval_hours,
        'median_hours': median_files * avg_interval_hours,
        'min_hours': min_files * avg_interval_hours,
        'max_hours': max_files * avg_interval_hours,
    }


def rebuild_preprocessing_artifacts(artifacts_path, data_path, window_size=2048, stride=512, test_size=0.2):
    """
    Recreate preprocessing artifacts from the training dataset when they are missing.

    Args:
        artifacts_path (str | Path): Destination artifact file
        data_path (str | Path): IMS dataset directory used for training
        window_size (int): Sliding window size
        stride (int): Sliding window stride
        test_size (float): Test split ratio used during training

    Returns:
        dict: Loaded preprocessing artifacts
    """
    print(f"Preprocessing artifacts not found. Rebuilding them from dataset: {data_path}")
    signals, files = load_bearing_data(str(data_path))
    X, y = create_dataset(signals, window_size, stride)
    X_train, _, y_train, y_test = split_data(X, y, test_size=test_size)
    _, signal_mean, signal_std = normalize_data(X_train, return_stats=True)
    _, _, scaler_y = scale_rul_labels(y_train, y_test)
    avg_interval_hours = calculate_time_intervals(files)

    artifacts_path = Path(artifacts_path)
    artifacts_path.parent.mkdir(parents=True, exist_ok=True)
    save_preprocessing_artifacts(
        str(artifacts_path),
        signal_mean,
        signal_std,
        scaler_y,
        window_size,
        stride,
        avg_interval_hours,
    )
    print(f"Rebuilt preprocessing artifacts at {artifacts_path}")
    return load_preprocessing_artifacts(artifacts_path)


def main():
    parser = argparse.ArgumentParser(description="Run inference on a single IMS bearing file.")
    parser.add_argument("input_file", help="Path to the IMS data file to score")
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
        "--bearing-column",
        type=int,
        default=4,
        help="Zero-based column index to read from the IMS file. Default: 4",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Training dataset directory used to rebuild missing preprocessing artifacts if needed",
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
    input_file = Path(args.input_file)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    print(f"Loading model from {model_path}")
    model = load_model(model_path)

    if artifacts_path.exists():
        print(f"Loading preprocessing artifacts from {artifacts_path}")
        artifacts = load_preprocessing_artifacts(artifacts_path)
    else:
        if not data_path.exists():
            raise FileNotFoundError(
                f"Preprocessing artifacts not found: {artifacts_path}. "
                f"Also could not rebuild them because dataset path does not exist: {data_path}"
            )
        artifacts = rebuild_preprocessing_artifacts(artifacts_path, data_path)

    print(f"Loading signal from {input_file}")
    signal = load_signal(input_file, bearing_column=args.bearing_column)
    X = prepare_windows(
        signal,
        artifacts['window_size'],
        artifacts['stride'],
        artifacts['signal_mean'],
        artifacts['signal_std'],
    )
    print(f"Created {len(X)} normalized windows for inference")

    print("Running model inference...")
    y_pred_scaled = model.predict(X, verbose=0).flatten()
    y_pred_files = artifacts['scaler_y'].inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    summary = summarize_predictions(y_pred_files, artifacts['avg_interval_hours'])

    print("\n========== INFERENCE RESULT ==========")
    print("The model predicts one RUL value per window, so the file-level result is aggregated below.")
    print(f"Input file: {input_file}")
    print(f"Window count: {len(y_pred_files)}")
    print(f"Mean predicted RUL: {summary['mean_files']:.2f} files ({summary['mean_hours']:.2f} hours)")
    print(f"Median predicted RUL: {summary['median_files']:.2f} files ({summary['median_hours']:.2f} hours)")
    print(f"Prediction range: {summary['min_files']:.2f} to {summary['max_files']:.2f} files")
    print(f"Prediction range: {summary['min_hours']:.2f} to {summary['max_hours']:.2f} hours")
    print("=" * 36)


if __name__ == "__main__":
    main()
