import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def normalize_data(X):
    """
    Normalize data using z-score normalization.

    Args:
        X (np.array): Input data

    Returns:
        np.array: Normalized data
    """
    print(f"Normalizing {len(X)} samples...")
    return (X - np.mean(X)) / np.std(X)

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
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
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
    X_augmented = [X_train]
    y_augmented = [y_train]
    print(f"Augmenting data with {num_augmentations} noisy copies...")

    for augmentation_index in range(num_augmentations):
        # Add small Gaussian noise
        noise = np.random.normal(0, np.std(X_train) * 0.05, X_train.shape)
        X_noisy = X_train + noise
        X_augmented.append(X_noisy)
        y_augmented.append(y_train)
        print(f"Completed augmentation {augmentation_index + 1}/{num_augmentations}")

    return np.concatenate(X_augmented), np.concatenate(y_augmented)

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
