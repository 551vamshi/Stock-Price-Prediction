"""
data_analysis.py
----------------
Module 3: Data Analytics

Computes all the financial metrics and statistics shown in the dashboard.

Key financial concepts explained:
- Moving Average (MA): Average of closing prices over N days. Smooths out
  daily noise and reveals the trend direction.
  * 20-day MA: Short-term trend indicator.
  * 50-day MA: Medium-term trend indicator.
- Daily Return: Percentage change from one day's close to the next.
  Formula: (Today - Yesterday) / Yesterday × 100
  Tells us: "How much did the stock move today?"
- Volatility: Standard deviation of daily returns over a rolling window.
  Higher volatility = higher risk (price swings more unpredictably).
- VWAP (Volume Weighted Average Price): Average price weighted by volume.
  Traders use it as a benchmark for trade execution quality.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def compute_moving_averages(df: pd.DataFrame, windows: list = None) -> pd.DataFrame:
    """
    Add moving average columns to the DataFrame.

    Parameters
    ----------
    df      : DataFrame with a 'Close' column
    windows : List of window sizes, default [20, 50]

    Returns
    -------
    DataFrame with new columns: MA_20, MA_50 (or whatever windows you pass)
    """
    if windows is None:
        windows = [20, 50]

    df = df.copy()
    for w in windows:
        col_name = f"MA_{w}"
        # rolling(w) computes the average of the last `w` rows
        # min_periods=1 ensures we get values even at the start of the series
        df[col_name] = df["Close"].rolling(window=w, min_periods=1).mean()

    return df


def compute_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate percentage daily returns.

    Formula: pct_change() = (today - yesterday) / yesterday
    Multiplied by 100 to express as percentage.

    Returns
    -------
    DataFrame with new column 'Daily_Return' (%)
    """
    df = df.copy()
    df["Daily_Return"] = df["Close"].pct_change() * 100
    return df


def compute_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Rolling standard deviation of daily returns.

    Volatility = std(daily returns over last N days)

    A high value means the stock price is swinging a lot.
    A low value means the stock is relatively stable.

    Returns
    -------
    DataFrame with new column 'Volatility'
    """
    df = df.copy()
    if "Daily_Return" not in df.columns:
        df = compute_daily_returns(df)

    df["Volatility"] = df["Daily_Return"].rolling(window=window, min_periods=1).std()
    return df


def compute_bollinger_bands(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Bollinger Bands: Upper and Lower bands around a moving average.

    Upper Band = MA_20 + (2 × std)
    Lower Band = MA_20 - (2 × std)

    When price touches the upper band → possibly overbought.
    When price touches the lower band → possibly oversold.

    Returns
    -------
    DataFrame with new columns: BB_Upper, BB_Middle, BB_Lower
    """
    df = df.copy()
    rolling = df["Close"].rolling(window=window, min_periods=1)
    df["BB_Middle"] = rolling.mean()
    std = rolling.std()
    df["BB_Upper"] = df["BB_Middle"] + 2 * std
    df["BB_Lower"] = df["BB_Middle"] - 2 * std
    return df


def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Relative Strength Index (RSI).

    RSI measures the speed and magnitude of price changes.
    Range: 0–100
      > 70 → Overbought (price may fall soon)
      < 30 → Oversold  (price may rise soon)

    Returns
    -------
    DataFrame with new column 'RSI'
    """
    df = df.copy()
    delta = df["Close"].diff()

    gain = delta.clip(lower=0)   # positive changes only
    loss = -delta.clip(upper=0)  # negative changes (made positive)

    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()

    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(50)  # neutral value when undefined

    return df


def get_summary_statistics(df: pd.DataFrame) -> dict:
    """
    Compute key summary statistics shown on the dashboard KPI cards.

    Returns a dictionary with:
    - latest_close     : Most recent closing price
    - all_time_high    : Highest closing price in the period
    - all_time_low     : Lowest closing price in the period
    - avg_close        : Mean closing price
    - total_return_pct : Total return from first to last close (%)
    - avg_daily_return : Average daily percentage change
    - std_daily_return : Volatility (std dev of daily returns)
    - avg_volume       : Average trading volume
    - latest_volume    : Most recent day's trading volume
    """
    close = df["Close"]

    latest_close = float(close.iloc[-1])
    first_close  = float(close.iloc[0])
    total_return = ((latest_close - first_close) / first_close) * 100

    daily_returns = close.pct_change().dropna() * 100

    return {
        "latest_close":     round(latest_close, 2),
        "all_time_high":    round(float(close.max()), 2),
        "all_time_low":     round(float(close.min()), 2),
        "avg_close":        round(float(close.mean()), 2),
        "total_return_pct": round(total_return, 2),
        "avg_daily_return": round(float(daily_returns.mean()), 4),
        "std_daily_return": round(float(daily_returns.std()), 4),
        "avg_volume":       int(df["Volume"].mean()) if "Volume" in df.columns else None,
        "latest_volume":    int(df["Volume"].iloc[-1]) if "Volume" in df.columns else None,
        "num_trading_days": len(df),
        "date_range_start": str(df["Date"].min().date()),
        "date_range_end":   str(df["Date"].max().date()),
    }


def get_price_change_today(df: pd.DataFrame) -> Tuple[float, float, str]:
    """
    Return the price change for the latest available trading day.

    Returns
    -------
    (change_amount, change_pct, direction)
    direction is "up", "down", or "flat"
    """
    if len(df) < 2:
        return 0.0, 0.0, "flat"

    latest = float(df["Close"].iloc[-1])
    prev   = float(df["Close"].iloc[-2])
    change = latest - prev
    pct    = (change / prev) * 100 if prev != 0 else 0.0

    direction = "up" if change > 0 else ("down" if change < 0 else "flat")
    return round(change, 2), round(pct, 2), direction


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function: run all technical indicator computations
    on the DataFrame in one call.

    Returns DataFrame enriched with all indicator columns.
    """
    df = compute_moving_averages(df, windows=[20, 50])
    df = compute_daily_returns(df)
    df = compute_volatility(df, window=20)
    df = compute_bollinger_bands(df, window=20)
    df = compute_rsi(df, period=14)
    return df
