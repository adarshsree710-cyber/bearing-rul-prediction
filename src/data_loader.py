import os
import numpy as np
from datetime import datetime

def load_bearing_data(data_path="../data/ims/1st_test"):
    """
    Load bearing vibration data from IMS dataset.

    Args:
        data_path (str): Path to the data directory

    Returns:
        tuple: (signals, files) - List of signals and corresponding filenames
    """
    files = sorted(os.listdir(data_path))
    signals = []

    for file in files:
        path = os.path.join(data_path, file)
        data = np.loadtxt(path)
        signals.append(data[:, 4])  # Bearing 3 horizontal

    print(f"Total signals loaded: {len(signals)}")
    return signals, files

def create_windows(signal, window_size=2048, stride=512):
    """
    Create sliding windows from a signal.

    Args:
        signal (np.array): Input signal
        window_size (int): Size of each window
        stride (int): Step size between windows

    Returns:
        list: List of windowed segments
    """
    windows = []

    for start in range(0, len(signal) - window_size + 1, stride):
        window = signal[start:start + window_size]
        windows.append(window)

    return windows

def create_dataset(signals, window_size=2048, stride=512):
    """
    Create X and y datasets from signals.

    Args:
        signals (list): List of signals
        window_size (int): Window size
        stride (int): Stride for windowing

    Returns:
        tuple: (X, y) - Features and labels
    """
    X = []
    y = []

    total_files = len(signals)

    for i, signal in enumerate(signals):
        windows = create_windows(signal, window_size, stride)
        rul = total_files - i

        for w in windows:
            X.append(w)
            y.append(rul)

    X = np.array(X)
    y = np.array(y)

    # Add channel dimension for CNN
    X = X[..., np.newaxis]

    return X, y

def parse_timestamp(filename):
    """
    Parse timestamp from filename format: YYYY.MM.DD.HH.MM.SS

    Args:
        filename (str): Filename with timestamp

    Returns:
        datetime: Parsed datetime object
    """
    parts = filename.split('.')
    if len(parts) >= 6:
        year, month, day, hour, minute, second = parts[:6]
        return datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
    return None

def calculate_time_intervals(files):
    """
    Calculate average time interval between files.

    Args:
        files (list): List of filenames

    Returns:
        float: Average interval in hours
    """
    timestamps = []
    for file in files:
        ts = parse_timestamp(file)
        if ts:
            timestamps.append(ts)

    if len(timestamps) > 1:
        time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() for i in range(len(timestamps)-1)]
        avg_interval_seconds = np.mean(time_diffs)
        avg_interval_hours = avg_interval_seconds / 3600
        return avg_interval_hours
    else:
        return 5/60  # Default 5 minutes