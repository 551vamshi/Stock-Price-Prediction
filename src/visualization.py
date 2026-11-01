"""
visualization.py
----------------
Module 6: Visualization

All Plotly charts used across the dashboard.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# ── Common theme settings ──────────────────────────────────────────────────────
DARK_BG   = "#0d1117"
GRID_CLR  = "#1f2937"
TEXT_CLR  = "#e5e7eb"
GREEN     = "#00C853"
RED       = "#FF1744"
BLUE      = "#2196F3"
GOLD      = "#FFD700"
ORANGE    = "#FF9800"

LAYOUT_DEFAULTS = dict(
    plot_bgcolor  = DARK_BG,
    paper_bgcolor = DARK_BG,
    font          = dict(color=TEXT_CLR, family="Inter, sans-serif"),
    xaxis         = dict(gridcolor=GRID_CLR, showgrid=True, zeroline=False),
    yaxis         = dict(gridcolor=GRID_CLR, showgrid=True, zeroline=False),
    legend        = dict(bgcolor="rgba(0,0,0,0.3)", bordercolor=GRID_CLR, borderwidth=1),
    hovermode     = "x unified",
    margin        = dict(l=50, r=30, t=60, b=50),
)


def _apply_theme(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply the standard dark theme and title to any figure."""
    fig.update_layout(**LAYOUT_DEFAULTS, title=dict(text=title, font=dict(size=16)))
    return fig


def plot_closing_price(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Line chart of historical closing prices over time."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["Close"],
        mode="lines",
        name="Close Price",
        line=dict(color=BLUE, width=1.5),
        fill="tozeroy",
        fillcolor="rgba(33, 150, 243, 0.08)",
        hovertemplate="<b>Date</b>: %{x|%d %b %Y}<br><b>Price</b>: ₹%{y:,.2f}<extra></extra>",
    ))

    _apply_theme(fig, f"{ticker} — Historical Closing Price")
    fig.update_yaxes(title_text="Price (₹)")
    return fig


def plot_candlestick(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Candlestick chart showing Open, High, Low, Close for each day."""
    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(df.columns):
        return plot_closing_price(df, ticker)

    fig = go.Figure(data=[go.Candlestick(
        x=df["Date"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        increasing_line_color=GREEN,
        decreasing_line_color=RED,
        name="OHLC",
    )])

    _apply_theme(fig, f"{ticker} — Candlestick Chart")
    fig.update_yaxes(title_text="Price (₹)")
    fig.update_xaxes(rangeslider_visible=False)
    return fig


def plot_moving_averages(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Overlay MA_20 and MA_50 on the closing price chart."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Close"],
        mode="lines", name="Close",
        line=dict(color=BLUE, width=1.2),
    ))

    if "MA_20" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df["MA_20"],
            mode="lines", name="20-Day MA",
            line=dict(color=ORANGE, width=1.5, dash="dot"),
        ))

    if "MA_50" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df["MA_50"],
            mode="lines", name="50-Day MA",
            line=dict(color=GREEN, width=1.5, dash="dash"),
        ))

    _apply_theme(fig, f"{ticker} — Moving Averages")
    fig.update_yaxes(title_text="Price (₹)")
    return fig


def plot_volume(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Bar chart of daily trading volume."""
    if "Volume" not in df.columns:
        return None

    colors = []
    for i in range(len(df)):
        if i == 0:
            colors.append(BLUE)
        else:
            colors.append(GREEN if df["Close"].iloc[i] >= df["Close"].iloc[i - 1] else RED)

    fig = go.Figure(data=[go.Bar(
        x=df["Date"],
        y=df["Volume"],
        marker_color=colors,
        name="Volume",
        hovertemplate="<b>Date</b>: %{x|%d %b %Y}<br><b>Volume</b>: %{y:,.0f}<extra></extra>",
    )])

    _apply_theme(fig, f"{ticker} — Daily Trading Volume")
    fig.update_yaxes(title_text="Volume (shares)")
    return fig


def plot_daily_returns(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Histogram of daily percentage returns."""
    if "Daily_Return" not in df.columns:
        return None

    returns = df["Daily_Return"].dropna()

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=60,
        name="Daily Returns",
        marker_color=BLUE,
        opacity=0.75,
        hovertemplate="<b>Return</b>: %{x:.2f}%<br><b>Days</b>: %{y}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="white", line_dash="dash", line_width=1)

    _apply_theme(fig, f"{ticker} — Daily Returns Distribution")
    fig.update_xaxes(title_text="Daily Return (%)")
    fig.update_yaxes(title_text="Frequency")
    return fig


def plot_volatility(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Rolling 20-day volatility (standard deviation of daily returns)."""
    if "Volatility" not in df.columns:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["Volatility"],
        mode="lines",
        name="20-Day Volatility",
        line=dict(color=ORANGE, width=1.5),
        fill="tozeroy",
        fillcolor="rgba(255, 152, 0, 0.1)",
    ))

    _apply_theme(fig, f"{ticker} — Rolling Volatility (20-Day Std Dev of Returns)")
    fig.update_yaxes(title_text="Volatility (%)")
    return fig


def plot_bollinger_bands(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Bollinger Bands with price and shaded band region."""
    required = {"BB_Upper", "BB_Middle", "BB_Lower"}
    if not required.issubset(df.columns):
        return None

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=pd.concat([df["Date"], df["Date"][::-1]]),
        y=pd.concat([df["BB_Upper"], df["BB_Lower"][::-1]]),
        fill="toself",
        fillcolor="rgba(33, 150, 243, 0.1)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Band Range",
        showlegend=False,
    ))

    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_Upper"],
                             line=dict(color=BLUE, width=1, dash="dot"),
                             name="Upper Band"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_Middle"],
                             line=dict(color="white", width=1.2),
                             name="Middle (MA20)"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_Lower"],
                             line=dict(color=BLUE, width=1, dash="dot"),
                             name="Lower Band"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"],
                             line=dict(color=GREEN, width=1.5),
                             name="Close"))

    _apply_theme(fig, f"{ticker} — Bollinger Bands")
    fig.update_yaxes(title_text="Price (₹)")
    return fig


def plot_rsi(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Relative Strength Index chart with overbought/oversold lines."""
    if "RSI" not in df.columns:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["RSI"],
        mode="lines", name="RSI (14)",
        line=dict(color=GOLD, width=1.5),
    ))

    fig.add_hline(y=70, line_color=RED,   line_dash="dash", line_width=1,
                  annotation_text="Overbought (70)", annotation_position="top right")
    fig.add_hline(y=30, line_color=GREEN, line_dash="dash", line_width=1,
                  annotation_text="Oversold (30)", annotation_position="bottom right")
    fig.add_hline(y=50, line_color="grey", line_dash="dot", line_width=0.8)

    _apply_theme(fig, f"{ticker} — RSI (14-Period)")
    fig.update_yaxes(title_text="RSI", range=[0, 100])
    return fig


def plot_training_loss(history) -> go.Figure:
    """Plot training and validation loss curves from Keras History object."""
    fig = go.Figure()

    epochs = list(range(1, len(history.history["loss"]) + 1))

    fig.add_trace(go.Scatter(
        x=epochs, y=history.history["loss"],
        mode="lines+markers", name="Training Loss",
        line=dict(color=BLUE, width=2),
        marker=dict(size=4),
    ))

    if "val_loss" in history.history:
        fig.add_trace(go.Scatter(
            x=epochs, y=history.history["val_loss"],
            mode="lines+markers", name="Validation Loss",
            line=dict(color=ORANGE, width=2, dash="dash"),
            marker=dict(size=4),
        ))

    _apply_theme(fig, "LSTM Model — Training & Validation Loss")
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Loss (MSE)")
    return fig


def plot_actual_vs_predicted(
    results_df: pd.DataFrame,
    ticker: str,
    next_day_pred: float = None,
    next_day_date: str = None,
) -> go.Figure:
    """Overlay actual and predicted price lines on the test period."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=results_df["Date"], y=results_df["Actual"],
        mode="lines", name="Actual Price",
        line=dict(color=GREEN, width=2),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Actual: ₹%{y:,.2f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=results_df["Date"], y=results_df["Predicted"],
        mode="lines", name="Predicted Price",
        line=dict(color=GOLD, width=2, dash="dash"),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Predicted: ₹%{y:,.2f}<extra></extra>",
    ))

    if next_day_pred is not None and next_day_date is not None:
        fig.add_trace(go.Scatter(
            x=[next_day_date],
            y=[next_day_pred],
            mode="markers+text",
            name="Next Day Forecast",
            marker=dict(color=RED, size=12, symbol="star"),
            text=[f"₹{next_day_pred:,.2f}"],
            textposition="top center",
            textfont=dict(color=RED, size=12),
        ))

    _apply_theme(fig, f"{ticker} — Actual vs LSTM Predicted Price")
    fig.update_yaxes(title_text="Price (₹)")
    return fig


def plot_full_prediction_overlay(
    df: pd.DataFrame,
    results_df: pd.DataFrame,
    ticker: str,
) -> go.Figure:
    """Full historical price chart with the prediction overlaid on test period."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Close"],
        mode="lines", name="Historical Price",
        line=dict(color=BLUE, width=1.2),
    ))

    fig.add_trace(go.Scatter(
        x=results_df["Date"], y=results_df["Actual"],
        mode="lines", name="Actual (Test Period)",
        line=dict(color=GREEN, width=2),
    ))

    fig.add_trace(go.Scatter(
        x=results_df["Date"], y=results_df["Predicted"],
        mode="lines", name="LSTM Predicted",
        line=dict(color=GOLD, width=2, dash="dash"),
    ))

    _apply_theme(fig, f"{ticker} — Full Historical + LSTM Prediction")
    fig.update_yaxes(title_text="Price (₹)")
    return fig


def plot_metrics_comparison(lstm_metrics: dict, baseline_metrics: dict) -> go.Figure:
    """Side-by-side bar chart comparing LSTM vs Baseline on MAE, RMSE, MAPE."""
    metrics = ["MAE (₹)", "RMSE (₹)", "MAPE (%)"]
    lstm_vals = [lstm_metrics["mae"], lstm_metrics["rmse"], lstm_metrics["mape"]]
    base_vals = [baseline_metrics["mae"], baseline_metrics["rmse"], baseline_metrics["mape"]]

    fig = go.Figure(data=[
        go.Bar(name="LSTM Model",   x=metrics, y=lstm_vals, marker_color=GOLD),
        go.Bar(name="Naïve Baseline", x=metrics, y=base_vals, marker_color=BLUE),
    ])

    fig.update_layout(barmode="group")
    _apply_theme(fig, "LSTM vs Naïve Baseline — Error Metrics")
    fig.update_yaxes(title_text="Error Value")
    return fig
