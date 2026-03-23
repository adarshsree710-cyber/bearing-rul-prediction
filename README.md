# Bearing Remaining Useful Life (RUL) Prediction

This project implements a Convolutional Neural Network (CNN) based approach to predict the Remaining Useful Life (RUL) of bearings using vibration sensor data from the IMS bearing dataset.

## Project Structure

```
bearing-rul-prediction/
├── data/
│   └── ims/
│       └── 1st_test/
│           ├── 2003.10.22.12.06.24
│           ├── 2003.10.22.12.09.13
│           └── ... (additional bearing data files)
├── models/
│   ├── cnn_rul_model.h5
│   └── y_max.py
├── notebooks/
│   └── data_exploration.ipynb
└── src/
    ├── __init__.py
    ├── data_loader.py
    ├── preprocessing.py
    ├── model.py
    ├── train.py
    ├── predict.py
    └── main.py
```

## Datasets

### IMS Bearing Dataset
- **Location**: `data/ims/1st_test/`
- **Description**: Vibration sensor data from bearings under accelerated life testing
- **Format**: Text files containing time-series vibration measurements (8 channels per file)
- **Usage**: Bearing 3 horizontal vibration channel (column 4) is used for RUL prediction
- **Time Series**: Files are timestamped, allowing calculation of time intervals between measurements

## Components

### Data Loading (`src/data_loader.py`)
- Loads vibration data from IMS dataset files
- Extracts bearing signals from text files
- Creates sliding windows for time-series analysis
- Handles file parsing and data aggregation

### Preprocessing (`src/preprocessing.py`)
- Normalizes vibration signals (z-score normalization)
- Creates training windows with configurable size and stride
- Applies data augmentation (Gaussian noise) to increase dataset size
- Handles train/test splitting and scaling of target values

### Model Architecture (`src/model.py`)
- **CNN Architecture**:
  - 3 Conv1D layers with increasing filters (32 → 64 → 64)
  - MaxPooling and Dropout (0.3) after each conv block
  - Dense layers (64 → 32) with L2 regularization
  - Output layer for regression
- **Regularization**: Dropout (0.5), L2 regularization (0.001)
- **Input Shape**: (2048, 1) - 2048-sample windows
- **Output**: Single value representing remaining useful life

### Training (`src/train.py`)
- Implements training callbacks:
  - EarlyStopping (patience=15, monitors val_loss)
  - ReduceLROnPlateau (factor=0.5, patience=5)
- Includes overfitting analysis functions
- Handles model training with validation

### Prediction (`src/predict.py`)
- Loads trained model for inference
- Handles prediction on new data
- Includes visualization of results
- Converts predictions to meaningful units (hours)

### Main Script (`src/main.py`)
- Orchestrates the entire pipeline
- Loads data, preprocesses, trains model, and evaluates
- Provides command-line interface for the project

## Models

### CNN RUL Model (`models/cnn_rul_model.h5`)
- **Framework**: TensorFlow/Keras
- **Architecture**: 1D Convolutional Neural Network
- **Purpose**: Regression model for RUL prediction
- **Input**: Normalized vibration signal windows
- **Output**: Predicted remaining useful life (scaled 0-1)
- **Training**: Optimized with early stopping and learning rate scheduling

### y_max Storage (`models/y_max.py`)
- Stores the maximum RUL value for scaling
- Used for inverse transformation of predictions

## Notebooks

### Data Exploration (`notebooks/data_exploration.ipynb`)
- Comprehensive Jupyter notebook containing:
  - Data loading and visualization
  - Signal processing and windowing
  - Model development and training
  - Performance evaluation and analysis
  - Overfitting diagnostics
  - Prediction visualization
- Includes all preprocessing steps and model training code
- Serves as the main development and experimentation environment

## Key Features

- **Data Augmentation**: Gaussian noise addition to improve generalization
- **Overfitting Prevention**: Multiple regularization techniques (dropout, L2, early stopping)
- **Time-aware Predictions**: Converts RUL predictions to hours using dataset timestamps
- **Comprehensive Evaluation**: MAE, RMSE metrics in both scaled and original units
- **Modular Design**: Separated concerns for maintainability and reusability

## Usage

1. **Data Preparation**:
   ```python
   from src.data_loader import load_bearing_data
   from src.preprocessing import preprocess_data
   
   # Load and preprocess data
   X, y = load_bearing_data()
   X_train, X_test, y_train, y_test = preprocess_data(X, y)
   ```

2. **Model Training**:
   ```python
   from src.model import create_cnn_model
   from src.train import train_model
   
   model = create_cnn_model()
   history = train_model(model, X_train, y_train, X_test, y_test)
   ```

3. **Prediction**:
   ```python
   from src.predict import predict_rul
   
   predictions = predict_rul(model, new_data)
   ```

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Scikit-learn
- IPython (for notebooks)

## Installation

```bash
pip install tensorflow numpy matplotlib scikit-learn ipython
```

## Results

The model achieves:
- **Test MAE**: ~0.05-0.1 (scaled units)
- **Test MAE**: ~2-5 hours (original units, depending on dataset)
- Effective overfitting control through regularization
- Good generalization to unseen bearing data

## Future Improvements

- Implement additional sensor channels
- Experiment with different architectures (LSTM, Transformer)
- Add uncertainty quantification
- Deploy as web service for real-time monitoring

---

**Note**: This project uses the IMS bearing dataset for research purposes. Ensure proper citation when using this work. Thank you. 