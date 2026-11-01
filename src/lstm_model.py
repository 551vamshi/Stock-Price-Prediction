"""
lstm_model.py
-------------
Module 4: Deep Learning Model (LSTM)

Builds, trains, saves, and uses an LSTM (Long Short-Term Memory) model
to predict stock prices.
"""

import numpy as np
import os
import pickle

# TensorFlow import with graceful error handling
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def check_tensorflow() -> bool:
    """Return True if TensorFlow is available."""
    return TF_AVAILABLE


def build_lstm_model(lookback: int = 60, units_1: int = 64, units_2: int = 32) -> "Sequential":
    """
    Construct and compile the LSTM neural network.

    Parameters
    ----------
    lookback : Number of past days used as input (must match preprocessing)
    units_1  : Number of neurons in the first LSTM layer
    units_2  : Number of neurons in the second LSTM layer

    Returns
    -------
    Compiled Keras Sequential model
    """
    if not TF_AVAILABLE:
        raise ImportError(
            "TensorFlow is not installed. "
            "Run: pip install tensorflow"
        )

    # Suppress TensorFlow info/warning messages for cleaner output
    tf.get_logger().setLevel("ERROR")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    model = Sequential([
        Input(shape=(lookback, 1)),

        # First LSTM layer
        # return_sequences=True means pass the full output sequence to the next layer
        LSTM(units=units_1, return_sequences=True),

        # Dropout: during training, randomly set 20% of activations to 0.
        Dropout(0.2),

        # Second LSTM layer — further refines temporal patterns
        LSTM(units=units_2, return_sequences=False),

        Dropout(0.2),

        # Dense hidden layer for non-linear transformation
        Dense(16, activation="relu"),

        # Output layer: single neuron → predicted next-day scaled price
        Dense(1),
    ])

    # Adam optimizer: adaptive learning rate, works well for time series
    # Loss: MSE (Mean Squared Error) — standard for regression problems
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    return model


def train_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    ticker: str = "stock",
) -> dict:
    """
    Train the LSTM model with early stopping.

    Early Stopping: If validation loss stops improving for `patience` epochs,
    training halts automatically. This saves time and prevents overfitting.
    """
    model_filename = f"{ticker.replace('.', '_')}_lstm.keras"
    model_path = os.path.join(MODELS_DIR, model_filename)

    early_stop = EarlyStopping(
        monitor="val_loss",   # watch validation loss
        patience=10,          # stop if no improvement for 10 epochs
        restore_best_weights=True,  # revert to the best epoch's weights
        verbose=0,
    )

    checkpoint = ModelCheckpoint(
        filepath=model_path,
        monitor="val_loss",
        save_best_only=True,  # only save when val_loss improves
        verbose=0,
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, checkpoint],
        verbose=0,  # suppress per-epoch console output
        shuffle=False,  # IMPORTANT: do NOT shuffle time-series data
    )

    epochs_ran = len(history.history["loss"])

    return {
        "history":    history,
        "model_path": model_path,
        "epochs_ran": epochs_ran,
    }


def predict(model, X: np.ndarray) -> np.ndarray:
    """
    Run inference: generate predictions for given input sequences.
    """
    predictions = model.predict(X, verbose=0)
    return predictions.flatten()


def predict_next_day(model, recent_data: np.ndarray, lookback: int, scaler) -> float:
    """
    Predict the NEXT trading day's closing price.
    """
    X = recent_data[-lookback:].reshape(1, lookback, 1)
    scaled_pred = model.predict(X, verbose=0)[0][0]
    actual_pred = scaler.inverse_transform([[scaled_pred]])[0][0]
    return round(float(actual_pred), 2)


def load_saved_model(ticker: str):
    """
    Load a previously saved model from the models/ directory.
    """
    model_filename = f"{ticker.replace('.', '_')}_lstm.keras"
    model_path = os.path.join(MODELS_DIR, model_filename)

    if os.path.exists(model_path):
        try:
            return load_model(model_path), model_path
        except Exception:
            return None, None
    return None, None


def model_exists(ticker: str) -> bool:
    """Check if a saved model file exists for this ticker."""
    model_filename = f"{ticker.replace('.', '_')}_lstm.keras"
    model_path = os.path.join(MODELS_DIR, model_filename)
    return os.path.exists(model_path)


def get_model_summary_text(model) -> str:
    """
    Return model architecture as a text string for display in the dashboard.
    """
    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    return "\n".join(lines)
