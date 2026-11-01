"""
tests/test_preprocessing.py
----------------------------
Unit tests for the preprocessing module.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

import pytest
import numpy as np
import pandas as pd

from preprocessing import (
    clean_data,
    prepare_lstm_data,
    inverse_transform,
    get_preprocessing_summary,
)


def make_sample_df(n_rows: int = 300, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    dates = pd.date_range(start="2021-01-01", periods=n_rows, freq="B")
    prices = 2000 + np.cumsum(np.random.randn(n_rows) * 10)
    prices = np.maximum(prices, 100)

    return pd.DataFrame({
        "Date":   dates,
        "Open":   prices * 0.99,
        "High":   prices * 1.01,
        "Low":    prices * 0.98,
        "Close":  prices,
        "Volume": np.random.randint(1_000_000, 5_000_000, n_rows),
    })


class TestCleanData:

    def test_removes_duplicates(self):
        df = make_sample_df(100)
        duplicate = df.iloc[[0]]
        df_with_dup = pd.concat([df, duplicate], ignore_index=True)
        cleaned = clean_data(df_with_dup)
        assert len(cleaned) == len(df)

    def test_sorted_by_date(self):
        df = make_sample_df(100)
        df_shuffled = df.sample(frac=1, random_state=0).reset_index(drop=True)
        cleaned = clean_data(df_shuffled)
        dates = cleaned["Date"].values
        assert all(dates[i] <= dates[i + 1] for i in range(len(dates) - 1))

    def test_drops_rows_with_missing_close(self):
        df = make_sample_df(100)
        df.loc[5,  "Close"] = np.nan
        df.loc[15, "Close"] = np.nan
        cleaned = clean_data(df)
        assert cleaned["Close"].isna().sum() == 0

    def test_preserves_non_null_rows(self):
        df = make_sample_df(200)
        cleaned = clean_data(df)
        assert len(cleaned) == 200

    def test_index_reset(self):
        df = make_sample_df(100)
        df.loc[5, "Close"] = np.nan
        cleaned = clean_data(df)
        assert list(cleaned.index) == list(range(len(cleaned)))


class TestPrepareLstmData:

    @pytest.fixture
    def sample_df(self):
        return make_sample_df(300)

    def test_shapes_correct(self, sample_df):
        lookback = 60
        X_tr, y_tr, X_te, y_te, scaler, train_size = prepare_lstm_data(
            sample_df, lookback=lookback
        )
        assert X_tr.ndim == 3
        assert X_te.ndim == 3
        assert X_tr.shape[1] == lookback
        assert X_tr.shape[2] == 1

    def test_train_test_sizes(self, sample_df):
        lookback = 60
        X_tr, y_tr, X_te, y_te, scaler, train_size = prepare_lstm_data(
            sample_df, lookback=lookback, train_ratio=0.80
        )
        expected_train = int(len(sample_df) * 0.80)
        assert train_size == expected_train

    def test_no_data_leakage_in_scaler(self, sample_df):
        lookback = 60
        X_tr, y_tr, X_te, y_te, scaler, train_size = prepare_lstm_data(
            sample_df, lookback=lookback, train_ratio=0.80
        )
        train_min = float(sample_df["Close"].iloc[:train_size].min())
        train_max = float(sample_df["Close"].iloc[:train_size].max())
        assert abs(float(scaler.data_min_[0]) - train_min) < 1e-3
        assert abs(float(scaler.data_max_[0]) - train_max) < 1e-3

    def test_chronological_split(self, sample_df):
        lookback = 60
        _, _, _, _, _, train_size = prepare_lstm_data(sample_df, lookback=lookback)
        last_train_date = sample_df["Date"].iloc[train_size - 1]
        first_test_date = sample_df["Date"].iloc[train_size]
        assert last_train_date < first_test_date

    def test_scaled_values_in_range(self, sample_df):
        lookback = 60
        X_tr, _, _, _, _, _ = prepare_lstm_data(sample_df, lookback=lookback)
        assert X_tr.min() >= 0.0 - 1e-6
        assert X_tr.max() <= 1.0 + 1e-6


class TestInverseTransform:

    def test_roundtrip(self):
        from sklearn.preprocessing import MinMaxScaler
        original = np.array([2000.0, 2500.0, 3000.0, 2200.0, 2800.0])
        scaler = MinMaxScaler()
        scaler.fit(original.reshape(-1, 1))
        scaled   = scaler.transform(original.reshape(-1, 1)).flatten()
        restored = inverse_transform(scaler, scaled)
        np.testing.assert_allclose(restored, original, atol=1e-3)

    def test_output_is_1d(self):
        from sklearn.preprocessing import MinMaxScaler
        values = np.array([100.0, 200.0, 300.0])
        scaler = MinMaxScaler()
        scaler.fit(values.reshape(-1, 1))
        scaled = scaler.transform(values.reshape(-1, 1)).flatten()
        result = inverse_transform(scaler, scaled)
        assert result.ndim == 1


class TestPreprocessingSummary:

    def test_summary_keys(self):
        df = make_sample_df(200)
        summary = get_preprocessing_summary(df, lookback=60, train_ratio=0.80)
        expected_keys = [
            "total_rows", "date_range", "train_rows", "test_rows",
            "lookback_window", "train_sequences", "test_sequences",
            "feature", "scaler",
        ]
        for key in expected_keys:
            assert key in summary

    def test_train_test_add_up(self):
        df = make_sample_df(200)
        summary = get_preprocessing_summary(df, lookback=60, train_ratio=0.80)
        assert summary["train_rows"] + summary["test_rows"] == summary["total_rows"]
