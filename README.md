# Bearing Remaining Useful Life (RUL) Prediction

This project uses a 1D CNN to predict the Remaining Useful Life (RUL) of bearings from the IMS bearing dataset.

## Project Structure

```text
bearing-rul-prediction/
|- data/
|  `- ims/
|     `- 1st_test/
|        |- 2003.10.22.12.06.24
|        |- 2003.10.22.12.09.13
|        `- ... (additional bearing data files)
|- models/
|  |- cnn_rul_model.keras
|  `- preprocessing_artifacts.pkl
|- outputs/
|  `- plots/
|     `- predictions_vs_actual_rul_hours.png
|- notebooks/
|  `- data_exploration.ipynb
`- src/
   |- __init__.py
   |- data_loader.py
   |- preprocessing.py
   |- model.py
   |- train.py
   |- predict.py
   |- infer.py
   `- main.py
```

## Dataset

### IMS Bearing Dataset
- Location: `data/ims/1st_test/`
- Format: text files containing time-series vibration measurements
- Signal used: bearing 3 horizontal vibration channel, column index `4`
- File names contain timestamps, which are used to estimate hours between measurements

## Components

### `src/data_loader.py`
- Loads IMS vibration files
- Extracts the bearing signal from each file
- Creates sliding windows for CNN input
- Parses timestamps from file names

### `src/preprocessing.py`
- Normalizes vibration signals with z-score normalization
- Splits data into train and test sets
- Scales RUL labels with `MinMaxScaler`
- Saves and loads preprocessing artifacts for inference

### `src/model.py`
- Defines the 1D CNN model
- Compiles the model for regression
- Checks GPU availability

### `src/train.py`
- Trains the model
- Uses early stopping and learning-rate reduction
- Prints training progress in the terminal
- Includes simple overfitting analysis

### `src/predict.py`
- Evaluates the trained model on test data
- Converts predictions to hours
- Saves prediction plots as image files

### `src/infer.py`
- Loads a trained model and preprocessing artifacts
- Accepts one IMS file as input
- Creates windows from that file
- Predicts RUL for each window
- Prints aggregated file-level RUL in files and hours

### `src/main.py`
- Runs the full training and evaluation pipeline
- Saves the model, preprocessing artifacts, and evaluation plot

## Saved Files

### `models/cnn_rul_model.keras`
- Trained TensorFlow/Keras model

### `models/preprocessing_artifacts.pkl`
- Saved normalization statistics
- Saved fitted label scaler information
- Saved `window_size`, `stride`, and average interval in hours

### `outputs/plots/predictions_vs_actual_rul_hours.png`
- Scatter plot of actual vs predicted RUL in hours

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Scikit-learn

## Installation

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

## Usage

### Train the Model

Run this from the project root:

```bash
python .\src\main.py
```

This will:
- train the CNN model
- save the model to `models/cnn_rul_model.keras`
- save preprocessing metadata to `models/preprocessing_artifacts.pkl`
- save the plot to `outputs/plots/predictions_vs_actual_rul_hours.png`

### Run Inference on One File

After training, you can run inference on a single IMS file:

```bash
python .\src\infer.py data/ims/1st_test/2003.10.22.13.54.13
```

This command:
- loads the trained model
- loads the preprocessing artifacts
- reads the input file
- creates sliding windows
- predicts RUL for each window
- prints aggregated RUL in files and approximate hours

If `models/preprocessing_artifacts.pkl` is missing, `infer.py` will try to rebuild it automatically from `data/ims/1st_test`.

## Notes

- The model predicts one RUL value per window, so `infer.py` reports aggregated file-level values.
- For consistent inference, the model and preprocessing artifacts should come from the same training configuration.

## Future Improvements

- Add support for additional sensor channels
- Try sequence models such as LSTM or Transformer variants
- Add uncertainty estimation
- Expose inference through an API or UI
