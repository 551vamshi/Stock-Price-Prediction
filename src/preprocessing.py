"""
preprocessing.py
----------------
Module 2: Data Preprocessing

Cleans raw stock data and transforms it into sequences that the LSTM
model can learn from.

Key concepts:
- MinMaxScaler: Scales values to the range [0, 1] so the neural network
  trains efficiently. Prices like ₹2,000 and ₹3,500 are mapped to 0.0–1.0.
- Lookback window: How many past trading days the model "looks back" to
  predict the next day. Default is 60 days.
- Train/Test split: We split data chronologically (not randomly!) to avoid
  data leakage — the model must never see future data during training.
- Data leakage: A serious mistake where future information accidentally
  helps train the model, giving fake-good performance metrics.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw OHLCV data.

    Steps performed:
    1. Remove duplicate rows (same date appearing twice).
    2. Sort by date ascending (oldest first).
    3. Forward-fill small gaps in price data (e.g., one missing day).
    4. Drop any remaining rows with missing Close price.
    5. Reset the index.

    Parameters
    ----------
    df : Raw DataFrame from data_collection module

    Returns
    -------
    Clean pd.DataFrame
    """
    # 1. Remove duplicates based on the Date column
    df = df.drop_duplicates(subset=["Date"])

    # 2. Sort chronologically
    df = df.sort_values("Date").reset_index(drop=True)

    # 3. Forward-fill missing numeric values (max 2 consecutive gaps)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].ffill(limit=2)

    # 4. Drop rows where Close is still missing after forward-fill
    df = df.dropna(subset=["Close"]).reset_index(drop=True)

    return df


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract useful calendar features from the Date column.

    These are used in analytics but NOT fed into the LSTM
    (LSTM uses only the price sequence).

    Adds columns: Year, Month, DayOfWeek, Quarter
    """
    df = df.copy()
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.dayofweek   # 0=Monday … 4=Friday
    df["Quarter"] = df["Date"].dt.quarter
    return df


def prepare_lstm_data(
    df: pd.DataFrame,
    lookback: int = 60,
    train_ratio: float = 0.80,
    feature_col: str = "Close",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, int]:
    """
    Transform a price series into supervised (X, y) sequences for LSTM.

    How it works
    ------------
    Given this closing-price series (simplified example with lookback=3):
        [100, 102, 98, 105, 110, 107]

    We create sequences like:
        X[0] = [100, 102, 98]  →  y[0] = 105
        X[1] = [102, 98, 105]  →  y[1] = 110
        X[2] = [98, 105, 110]  →  y[2] = 107

    The model learns: "given these 3 days, predict the next day."

    Parameters
    ----------
    df          : Clean DataFrame with a Date and Close column
    lookback    : Number of past trading days used as input (default 60)
    train_ratio : Fraction of data for training (default 0.80 = 80%)
    feature_col : Column name to predict (default "Close")

    Returns
    -------
    X_train, y_train : Training sequences and labels
    X_test,  y_test  : Testing sequences and labels
    scaler           : Fitted MinMaxScaler (needed to inverse-transform predictions)
    train_size       : Number of training rows (useful for aligning dates)
    """
    # ── 1. Extract the price series as a 2D array ──────────────────────────
    prices = df[feature_col].values.reshape(-1, 1)  # shape: (N, 1)

    # ── 2. Split into train and test BEFORE scaling ──────────────────────────
    # Chronological split: first 80% = training, last 20% = testing
    train_size = int(len(prices) * train_ratio)

    train_prices = prices[:train_size]
    test_prices  = prices[train_size:]

    # ── 3. Fit scaler ONLY on training data ──────────────────────────────────
    # IMPORTANT: If we fit the scaler on all data, the training set would
    # indirectly "see" future prices through the scale — that is data leakage!
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_prices)

    # Transform the test data using the SAME scaler (no re-fitting)
    test_scaled = scaler.transform(test_prices)

    # ── 4. Build sequences from the training data ─────────────────────────────
    X_train, y_train = _build_sequences(train_scaled, lookback)

    # ── 5. Build sequences from training tail + test data ─────────────────────
    # The first test sequence needs the last `lookback` training values as context
    combined_test = np.concatenate([train_scaled[-lookback:], test_scaled], axis=0)
    X_test, y_test = _build_sequences(combined_test, lookback)

    # ── 6. Reshape X to 3D: (samples, timesteps, features) ───────────────────
    # LSTM expects input shape: (batch_size, lookback, number_of_features)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test  = X_test.reshape(X_test.shape[0],  X_test.shape[1],  1)

    return X_train, y_train, X_test, y_test, scaler, train_size


def _build_sequences(data: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Internal helper: convert a 1D scaled series into (X, y) sequence pairs.

    Parameters
    ----------
    data     : 2D array of shape (N, 1) with scaled price values
    lookback : Size of each input window

    Returns
    -------
    X : shape (N - lookback, lookback)
    y : shape (N - lookback,)
    """
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback : i, 0])   # past `lookback` days
        y.append(data[i, 0])                   # next day to predict
    return np.array(X), np.array(y)


def inverse_transform(scaler: MinMaxScaler, values: np.ndarray) -> np.ndarray:
    """
    Convert scaled (0–1) predicted values back to original price scale.

    Example: scaled value 0.65 → ₹2,345 (actual price)

    Parameters
    ----------
    scaler : The same scaler used during preprocessing
    values : 1D array of scaled predictions

    Returns
    -------
    1D array of prices in original scale
    """
    return scaler.inverse_transform(values.reshape(-1, 1)).flatten()


def get_preprocessing_summary(df: pd.DataFrame, lookback: int, train_ratio: float) -> dict:
    """
    Return a human-readable summary of preprocessing parameters.
    Displayed in the dashboard to help users understand the data.
    """
    train_size = int(len(df) * train_ratio)
    test_size  = len(df) - train_size

    return {
        "total_rows": len(df),
        "date_range": f"{df['Date'].min().date()} → {df['Date'].max().date()}",
        "train_rows": train_size,
        "test_rows": test_size,
        "lookback_window": lookback,
        "train_sequences": max(0, train_size - lookback),
        "test_sequences": max(0, test_size),
        "feature": "Close Price",
        "scaler": "MinMaxScaler [0, 1]",
    }
