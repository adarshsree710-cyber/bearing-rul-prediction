import pickle

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def compute_normalization_stats(X):
    """
    Compute normalization statistics for vibration data.

    Args:
        X (np.array): Input data

    Returns:
        tuple: (mean, std)
    """
    X = X.astype(np.float32, copy=False)
    mean = float(np.mean(X, dtype=np.float32))
    std = float(np.std(X, dtype=np.float32))
    return mean, std

def apply_normalization(X, mean, std):
    """
    Normalize data using provided z-score statistics.

    Args:
        X (np.array): Input data
        mean (float): Mean used for normalization
        std (float): Standard deviation used for normalization

    Returns:
        np.array: Normalized data
    """
    X = X.astype(np.float32, copy=False)
    return ((X - mean) / (std + 1e-8)).astype(np.float32, copy=False)

def normalize_data(X, mean=None, std=None, return_stats=False):
    """
    Normalize data using z-score normalization.

    Args:
        X (np.array): Input data
        mean (float | None): Optional precomputed mean
        std (float | None): Optional precomputed std
        return_stats (bool): Whether to return mean and std

    Returns:
        np.array | tuple: Normalized data, optionally with stats
    """
    print(f"Normalizing {len(X)} samples...")
    if mean is None or std is None:
        mean, std = compute_normalization_stats(X)

    normalized = apply_normalization(X, mean, std)

    if return_stats:
        return normalized, mean, std

    return normalized

def scale_labels(y, y_max=None):
    """
    Scale RUL labels to [0, 1] range.

    Args:
        y (np.array): RUL labels
        y_max (float): Maximum RUL value for scaling

    Returns:
        tuple: (scaled_y, y_max, scaler)
    """
    if y_max is None:
        y_max = y.max()

    y_scaled = y / y_max
    return y_scaled, y_max

def create_scaler():
    """
    Create MinMaxScaler for RUL values.

    Returns:
        MinMaxScaler: Configured scaler
    """
    return MinMaxScaler()

def scale_rul_labels(y_train, y_test):
    """
    Scale RUL labels using MinMaxScaler.

    Args:
        y_train (np.array): Training labels
        y_test (np.array): Test labels

    Returns:
        tuple: (y_train_scaled, y_test_scaled, scaler)
    """
    scaler_y = MinMaxScaler()
    print(f"Scaling RUL labels: train={len(y_train)}, test={len(y_test)}")
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten().astype(np.float32)
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten().astype(np.float32)
    return y_train_scaled, y_test_scaled, scaler_y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets.

    Args:
        X (np.array): Features
        y (np.array): Labels
        test_size (float): Test set proportion
        random_state (int): Random state for reproducibility

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print(f"Splitting data: samples={len(X)}, test_size={test_size}")
    return train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True)

def augment_data(X_train, y_train, num_augmentations=3):
    """
    Augment training data by adding noise.

    Args:
        X_train (np.array): Training features
        y_train (np.array): Training labels
        num_augmentations (int): Number of augmentations

    Returns:
        tuple: (X_augmented, y_augmented)
    """
    X_train = X_train.astype(np.float32, copy=False)
    y_train = y_train.astype(np.float32, copy=False)
    X_augmented = [X_train]
    y_augmented = [y_train]
    print(f"Augmenting data with {num_augmentations} noisy copies...")

    for augmentation_index in range(num_augmentations):
        # Add small Gaussian noise
        noise = np.random.normal(
            0,
            np.std(X_train, dtype=np.float32) * 0.05,
            X_train.shape
        ).astype(np.float32)
        X_noisy = (X_train + noise).astype(np.float32, copy=False)
        X_augmented.append(X_noisy)
        y_augmented.append(y_train)
        print(f"Completed augmentation {augmentation_index + 1}/{num_augmentations}")

    return (
        np.concatenate(X_augmented, axis=0).astype(np.float32, copy=False),
        np.concatenate(y_augmented, axis=0).astype(np.float32, copy=False)
    )

def shuffle_data(X, y, random_state=42):
    """
    Shuffle data.

    Args:
        X (np.array): Features
        y (np.array): Labels
        random_state (int): Random state

    Returns:
        tuple: (X_shuffled, y_shuffled)
    """
    print(f"Shuffling {len(X)} samples...")
    return shuffle(X, y, random_state=random_state)

def save_preprocessing_artifacts(filepath, mean, std, scaler_y, window_size, stride, avg_interval_hours):
    """
    Save preprocessing metadata needed for inference.

    Args:
        filepath (str): Destination file path
        mean (float): Training normalization mean
        std (float): Training normalization std
        scaler_y (MinMaxScaler): Fitted label scaler
        window_size (int): Sliding window size
        stride (int): Sliding window stride
        avg_interval_hours (float): Average time interval between files
    """
    artifacts = {
        'signal_mean': float(mean),
        'signal_std': float(std),
        'window_size': int(window_size),
        'stride': int(stride),
        'avg_interval_hours': float(avg_interval_hours),
        'scaler_min': scaler_y.min_.astype(np.float32),
        'scaler_scale': scaler_y.scale_.astype(np.float32),
        'scaler_data_min': scaler_y.data_min_.astype(np.float32),
        'scaler_data_max': scaler_y.data_max_.astype(np.float32),
        'scaler_data_range': scaler_y.data_range_.astype(np.float32),
        'scaler_feature_range': scaler_y.feature_range,
    }

    with open(filepath, 'wb') as artifact_file:
        pickle.dump(artifacts, artifact_file)

def load_preprocessing_artifacts(filepath):
    """
    Load preprocessing metadata needed for inference.

    Args:
        filepath (str): Artifact file path

    Returns:
        dict: Loaded artifacts with a reconstructed label scaler
    """
    with open(filepath, 'rb') as artifact_file:
        artifacts = pickle.load(artifact_file)

    scaler_y = MinMaxScaler(feature_range=artifacts['scaler_feature_range'])
    scaler_y.min_ = np.array(artifacts['scaler_min'], dtype=np.float32)
    scaler_y.scale_ = np.array(artifacts['scaler_scale'], dtype=np.float32)
    scaler_y.data_min_ = np.array(artifacts['scaler_data_min'], dtype=np.float32)
    scaler_y.data_max_ = np.array(artifacts['scaler_data_max'], dtype=np.float32)
    scaler_y.data_range_ = np.array(artifacts['scaler_data_range'], dtype=np.float32)
    scaler_y.n_features_in_ = 1
    scaler_y.n_samples_seen_ = 1

    artifacts['scaler_y'] = scaler_y
    return artifacts
