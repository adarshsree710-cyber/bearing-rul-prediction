from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def create_callbacks():
    """
    Create training callbacks.

    Returns:
        list: List of callbacks
    """
    # Early stopping with patience
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,  # Increased patience for better convergence
        restore_best_weights=True
    )

    # Reduce learning rate when validation loss plateaus
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )

    return [early_stop, reduce_lr]

def train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=64):
    """
    Train the CNN model.

    Args:
        model: Keras model to train
        X_train (np.array): Training features
        y_train (np.array): Training labels
        X_test (np.array): Test features
        y_test (np.array): Test labels
        epochs (int): Number of epochs
        batch_size (int): Batch size

    Returns:
        History: Training history
    """
    callbacks = create_callbacks()

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    return history

def analyze_overfitting(history):
    """
    Analyze overfitting from training history.

    Args:
        history: Keras training history

    Returns:
        dict: Overfitting analysis metrics
    """
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Calculate gap between training and validation loss
    loss_gap = [val - train for val, train in zip(val_loss, train_loss)]

    analysis = {
        'final_train_loss': train_loss[-1],
        'final_val_loss': val_loss[-1],
        'loss_gap': loss_gap[-1],
        'avg_loss_gap': sum(loss_gap) / len(loss_gap),
        'max_loss_gap': max(loss_gap),
        'total_epochs': len(train_loss)
    }

    return analysis