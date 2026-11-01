"""
evaluation.py
-------------
Module 5: Model Evaluation

Measures how well the LSTM model performs using standard regression metrics:
MAE, RMSE, and MAPE, and compares against a naïve baseline.
"""

import numpy as np
import pandas as pd


def compute_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error in original price units (₹)."""
    return float(np.mean(np.abs(actual - predicted)))


def compute_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error in original price units (₹)."""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def compute_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error (%).

    Zero-safe: excludes data points where actual price is 0
    to avoid division-by-zero errors.
    """
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)

    mask = actual != 0
    if mask.sum() == 0:
        return float("nan")

    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def evaluate_model(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """
    Compute all evaluation metrics for the LSTM model.
    """
    return {
        "mae":          round(compute_mae(actual, predicted), 4),
        "rmse":         round(compute_rmse(actual, predicted), 4),
        "mape":         round(compute_mape(actual, predicted), 4),
        "num_samples":  len(actual),
    }


def evaluate_baseline(actual: np.ndarray) -> dict:
    """
    Evaluate the naïve baseline: predict today's price as yesterday's close.
    """
    baseline_pred = actual[:-1]
    baseline_actual = actual[1:]

    return {
        "mae":  round(compute_mae(baseline_actual, baseline_pred), 4),
        "rmse": round(compute_rmse(baseline_actual, baseline_pred), 4),
        "mape": round(compute_mape(baseline_actual, baseline_pred), 4),
        "label": "Naïve Baseline (previous day's price)",
    }


def compare_models(lstm_metrics: dict, baseline_metrics: dict) -> dict:
    """
    Compare LSTM metrics against the baseline.
    """
    def improvement(lstm_val, base_val):
        if base_val == 0 or base_val != base_val:
            return None
        pct = ((base_val - lstm_val) / base_val) * 100
        return round(pct, 2)

    return {
        "mae_improvement_pct":  improvement(lstm_metrics["mae"],  baseline_metrics["mae"]),
        "rmse_improvement_pct": improvement(lstm_metrics["rmse"], baseline_metrics["rmse"]),
        "mape_improvement_pct": improvement(lstm_metrics["mape"], baseline_metrics["mape"]),
        "lstm_beats_baseline": (
            lstm_metrics["rmse"] < baseline_metrics["rmse"]
        ),
    }


def align_predictions_with_dates(
    df: pd.DataFrame,
    actual: np.ndarray,
    predicted: np.ndarray,
    train_size: int,
    lookback: int,
) -> pd.DataFrame:
    """
    Create a DataFrame aligning dates with actual and predicted prices.
    """
    test_dates = df["Date"].iloc[train_size : train_size + len(actual)].reset_index(drop=True)

    result = pd.DataFrame({
        "Date":      test_dates,
        "Actual":    np.round(actual, 2),
        "Predicted": np.round(predicted, 2),
    })

    return result
