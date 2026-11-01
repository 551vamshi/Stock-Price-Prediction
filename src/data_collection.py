"""
data_collection.py
------------------
Module 1: Stock Data Collection

Fetches historical stock data from Yahoo Finance (yfinance).
Also supports loading data from a local CSV file as a fallback
when the internet is unavailable.

Key concepts:
- yfinance: A Python library that downloads stock data from Yahoo Finance.
- Ticker symbol: A unique code for a stock, e.g., "RELIANCE.NS" for
  Reliance Industries listed on the National Stock Exchange (NSE) of India.
- OHLCV data: Open, High, Low, Close, Volume - the standard columns
  in stock market data for each trading day.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os


# ── Popular Indian stock tickers for quick selection ──────────────────────────
POPULAR_STOCKS = {
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Wipro": "WIPRO.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "ITC": "ITC.NS",
    "Asian Paints": "ASIANPAINT.NS",
}


def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download historical OHLCV data from Yahoo Finance.

    Parameters
    ----------
    ticker     : Stock symbol, e.g. "RELIANCE.NS"
    start_date : Start date in "YYYY-MM-DD" format
    end_date   : End date   in "YYYY-MM-DD" format

    Returns
    -------
    pd.DataFrame with columns: Date, Open, High, Low, Close, Volume
    Raises RuntimeError on failure.
    """
    try:
        ticker = ticker.strip().upper()

        # yf.download returns a DataFrame indexed by date
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,   # suppress the download progress bar
            auto_adjust=True, # auto-adjust for splits & dividends
        )

        if df.empty:
            raise RuntimeError(
                f"No data returned for '{ticker}'. "
                "Check the ticker symbol and date range."
            )

        # Flatten MultiIndex columns that yfinance sometimes returns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Reset index so Date becomes a regular column
        df.reset_index(inplace=True)
        df.rename(columns={"index": "Date"}, inplace=True)

        # Ensure the Date column is a proper datetime
        df["Date"] = pd.to_datetime(df["Date"])

        # Keep only standard OHLCV columns that actually exist
        desired_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        existing_cols = [c for c in desired_cols if c in df.columns]
        df = df[existing_cols]

        # Drop rows where Close is completely missing
        df.dropna(subset=["Close"], inplace=True)

        return df

    except Exception as exc:
        raise RuntimeError(
            f"Failed to fetch data for '{ticker}': {exc}\n"
            "Possible reasons:\n"
            "  • No internet connection\n"
            "  • Invalid ticker symbol\n"
            "  • Yahoo Finance is temporarily unavailable\n"
            "Try uploading a CSV file instead."
        ) from exc


def load_csv_data(filepath: str) -> pd.DataFrame:
    """
    Load stock data from a local CSV file.

    Expected columns (case-insensitive):
        Date, Open, High, Low, Close, Volume

    Parameters
    ----------
    filepath : Full path to the CSV file

    Returns
    -------
    pd.DataFrame in standard OHLCV format
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    try:
        df = pd.read_csv(filepath)

        # Normalize column names to title-case (e.g. "date" -> "Date")
        df.columns = [col.strip().title() for col in df.columns]

        # Rename common alternative spellings
        col_map = {
            "Adj Close": "Close",
            "Adjusted Close": "Close",
            "Adj_Close": "Close",
        }
        df.rename(columns=col_map, inplace=True)

        required = ["Date", "Close"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {missing}. "
                "The file must contain at least 'Date' and 'Close' columns."
            )

        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values("Date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.dropna(subset=["Close"], inplace=True)

        return df

    except Exception as exc:
        raise RuntimeError(f"Failed to read CSV file: {exc}") from exc


def get_stock_info(ticker: str) -> dict:
    """
    Fetch basic company information for a given ticker.

    Returns a dictionary with company name, sector, market cap, etc.
    Falls back gracefully if the info cannot be retrieved.
    """
    try:
        stock = yf.Ticker(ticker.strip().upper())
        info = stock.info

        return {
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", None),
            "currency": info.get("currency", "INR"),
            "exchange": info.get("exchange", "NSE"),
            "website": info.get("website", "N/A"),
            "description": info.get("longBusinessSummary", "N/A"),
        }
    except Exception:
        # Return minimal info rather than crashing
        return {
            "name": ticker,
            "sector": "N/A",
            "industry": "N/A",
            "market_cap": None,
            "currency": "INR",
            "exchange": "NSE",
            "website": "N/A",
            "description": "Company information unavailable.",
        }


def validate_ticker(ticker: str) -> bool:
    """
    Quick check: does this ticker return any data?

    Returns True if valid, False otherwise.
    Used for user-input validation before a full download.
    """
    try:
        df = yf.download(ticker.strip().upper(),
                         period="5d", progress=False, auto_adjust=True)
        return not df.empty
    except Exception:
        return False


def get_default_date_range() -> tuple:
    """
    Return a sensible default (start, end) date range.
    Default: last 3 years up to today.
    """
    end = datetime.today()
    start = end - timedelta(days=3 * 365)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
