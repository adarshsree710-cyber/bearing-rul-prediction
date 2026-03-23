import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

def create_cnn_model(input_shape=(2048, 1)):
    """
    Create CNN model for RUL prediction.

    Args:
        input_shape (tuple): Input shape for the model

    Returns:
        Sequential: Compiled CNN model
    """
    model = Sequential([
        # Reduced model complexity with stronger regularization
        Conv1D(32, kernel_size=5, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Conv1D(64, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Conv1D(64, kernel_size=3, activation='relu'),  # Reduced from 128
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Flatten(),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),  # Reduced + stronger L2
        Dropout(0.5),  # Increased from 0.4
        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),  # Reduced + stronger L2
        Dropout(0.5),  # Increased from 0.4

        Dense(1)
    ])

    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse', metrics=['mae'])

    return model

def check_gpu_availability():
    """
    Check if GPU is available for training.

    Returns:
        list: List of available GPU devices
    """
    return tf.config.list_physical_devices('GPU')