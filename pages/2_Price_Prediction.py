"""
pages/2_Price_Prediction.py
---------------------------
Page 3: Deep Learning LSTM Price Prediction
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

import streamlit as st
from auth import require_auth
import numpy as np
from datetime import timedelta

st.set_page_config(
    page_title="LSTM Prediction | Stock Predictor",
    page_icon="🤖",
    layout="wide",
)

# User Authentication
current_user = require_auth()

st.markdown("""
<style>
.main { background-color: #0d1117; }
.stApp { background-color: #0d1117; }
section[data-testid="stSidebar"] { background-color: #161b22 !important; border-right: 1px solid #30363d; }
.section-header { font-size: 1.15rem; font-weight: 600; color: #e5e7eb; border-left: 3px solid #FFD700; padding-left: 10px; margin: 20px 0 10px 0; }
.pred-box { background: linear-gradient(135deg, #1a2a1a 0%, #0d1a0d 100%); border: 2px solid #00C853; border-radius: 12px; padding: 24px; text-align: center; margin: 16px 0; }
.pred-price { font-size: 3rem; font-weight: 800; color: #FFD700; }
.pred-label { font-size: 0.9rem; color: #9ca3af; margin-top: 4px; }
.warning-banner { background: rgba(234,179,8,0.1); border: 1px solid #eab308; border-radius: 8px; padding: 12px 16px; font-size: 0.85rem; color: #fde68a; margin-bottom: 16px; }
div[data-testid="stMetric"] { background: #1f2937; border-radius: 10px; padding: 12px; border: 1px solid #30363d; }
</style>
""", unsafe_allow_html=True)

if st.session_state.get("df") is None:
    st.warning("⚠️ No data loaded. Go to the **Home** page and fetch stock data first.")
    st.stop()

try:
    from src.lstm_model import check_tensorflow, TF_AVAILABLE
    from src.preprocessing import prepare_lstm_data, inverse_transform, get_preprocessing_summary
    from src.lstm_model import build_lstm_model, train_model, predict, predict_next_day, model_exists
    from src.evaluation import evaluate_model, evaluate_baseline, align_predictions_with_dates
    from src.visualization import plot_training_loss, plot_actual_vs_predicted, plot_full_prediction_overlay
except (ImportError, ModuleNotFoundError):
    from lstm_model import check_tensorflow, TF_AVAILABLE
    from preprocessing import prepare_lstm_data, inverse_transform, get_preprocessing_summary
    from lstm_model import build_lstm_model, train_model, predict, predict_next_day, model_exists
    from evaluation import evaluate_model, evaluate_baseline, align_predictions_with_dates
    from visualization import plot_training_loss, plot_actual_vs_predicted, plot_full_prediction_overlay

df     = st.session_state.df
ticker = st.session_state.ticker

st.markdown(f"## 🤖 LSTM Price Prediction — **{ticker}**")
st.markdown("""
<div class="warning-banner">
    ⚠️ <strong>Important:</strong> LSTM predictions are statistical estimates based on historical patterns.
    Stock markets are influenced by news, geopolitics, and human behaviour — none of which a model can fully capture.
    <strong>Do NOT use this for real investment decisions.</strong>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

st.markdown('<div class="section-header">⚙️ Model Configuration</div>', unsafe_allow_html=True)

cfg1, cfg2, cfg3, cfg4 = st.columns(4)

with cfg1:
    lookback = st.slider(
        "Lookback Window (days)",
        min_value=20, max_value=120, value=60, step=10,
        help="How many past trading days the LSTM uses to predict the next day.",
    )
with cfg2:
    epochs = st.slider(
        "Max Training Epochs",
        min_value=10, max_value=100, value=50, step=10,
        help="Maximum times the model trains over the full dataset.",
    )
with cfg3:
    batch_size = st.selectbox(
        "Batch Size",
        options=[16, 32, 64],
        index=1,
        help="Number of training samples processed together in one step.",
    )
with cfg4:
    train_ratio = st.slider(
        "Train/Test Split",
        min_value=0.70, max_value=0.90, value=0.80, step=0.05,
        format="%.0f%%",
        help="Fraction of data used for training. 80% = 80% train, 20% test.",
    )

with st.expander("📋 Preprocessing Summary"):
    summary = get_preprocessing_summary(df, lookback, train_ratio)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        | Parameter | Value |
        |-----------|-------|
        | Total Rows | {summary['total_rows']:,} |
        | Date Range | {summary['date_range']} |
        | Training Rows | {summary['train_rows']:,} |
        | Testing Rows  | {summary['test_rows']:,} |
        """)
    with col2:
        st.markdown(f"""
        | Parameter | Value |
        |-----------|-------|
        | Lookback Window | {summary['lookback_window']} days |
        | Training Sequences | {summary['train_sequences']:,} |
        | Test Sequences | {summary['test_sequences']:,} |
        | Scaler | {summary['scaler']} |
        """)

st.markdown("")

col_btn1, col_btn2 = st.columns([1, 3])
with col_btn1:
    train_btn = st.button("🚀 Train LSTM Model", type="primary", use_container_width=True)

with col_btn2:
    if model_exists(ticker):
        st.success(f"✅ A saved model exists for **{ticker}**. Training will overwrite it.")
    else:
        st.info("No saved model found for this ticker. Click Train to build one.")

if train_btn:
    if len(df) < lookback + 50:
        st.error(
            f"❌ Not enough data to train. Need at least {lookback + 50} rows. "
            f"Current data has {len(df)} rows. Try a longer date range or smaller lookback."
        )
        st.stop()

    progress_bar = st.progress(0, text="Step 1/4 — Preprocessing data…")
    status_text  = st.empty()

    try:
        X_train, y_train, X_test, y_test, scaler, train_size = prepare_lstm_data(
            df,
            lookback=lookback,
            train_ratio=train_ratio,
        )
        val_split = int(len(X_train) * 0.9)
        X_tr, X_val = X_train[:val_split], X_train[val_split:]
        y_tr, y_val = y_train[:val_split], y_train[val_split:]

        progress_bar.progress(25, text="Step 2/4 — Building LSTM model…")
        status_text.markdown(
            f"✅ Data preprocessed: **{len(X_tr):,}** train, **{len(X_val):,}** val, **{len(X_test):,}** test sequences."
        )
    except Exception as e:
        st.error(f"❌ Preprocessing failed: {e}")
        st.stop()

    try:
        model = build_lstm_model(lookback=lookback)
        progress_bar.progress(40, text="Step 3/4 — Training model (this may take 1–3 minutes)…")
        status_text.markdown("🔧 Model built successfully. Starting training…")
    except Exception as e:
        st.error(f"❌ Failed to build model: {e}")
        st.stop()

    try:
        with st.spinner(f"Training LSTM for up to {epochs} epochs…"):
            results = train_model(
                model, X_tr, y_tr, X_val, y_val,
                epochs=epochs,
                batch_size=batch_size,
                ticker=ticker,
            )

        epochs_ran = results["epochs_ran"]
        progress_bar.progress(75, text="Step 4/4 — Generating predictions…")
        status_text.markdown(
            f"✅ Training complete in **{epochs_ran} epochs** "
            f"(early stopping saved {epochs - epochs_ran} epochs)."
        )
    except Exception as e:
        st.error(f"❌ Training failed: {e}")
        st.stop()

    try:
        pred_scaled   = predict(model, X_test)
        actual_scaled = y_test

        pred_prices   = inverse_transform(scaler, pred_scaled)
        actual_prices = inverse_transform(scaler, actual_scaled)

        all_scaled = scaler.transform(df["Close"].values.reshape(-1, 1))
        next_price = predict_next_day(model, all_scaled, lookback, scaler)
        next_date  = df["Date"].iloc[-1] + timedelta(days=1)
        while next_date.weekday() >= 5:
            next_date += timedelta(days=1)

        results_df = align_predictions_with_dates(df, actual_prices, pred_prices, train_size, lookback)

        lstm_metrics     = evaluate_model(actual_prices, pred_prices)
        baseline_metrics = evaluate_baseline(actual_prices)

        progress_bar.progress(100, text="✅ All done!")
        status_text.markdown("✅ Predictions generated successfully.")

        st.session_state.model_results = {
            "results_df":       results_df,
            "actual_prices":    actual_prices,
            "pred_prices":      pred_prices,
            "next_price":       next_price,
            "next_date":        next_date.strftime("%Y-%m-%d"),
            "lstm_metrics":     lstm_metrics,
            "baseline_metrics": baseline_metrics,
            "history":          results["history"],
            "epochs_ran":       epochs_ran,
            "lookback":         lookback,
            "train_size":       train_size,
            "scaler":           scaler,
        }

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.stop()

if st.session_state.get("model_results") is not None:
    res = st.session_state.model_results

    st.markdown("---")
    st.markdown('<div class="section-header">🎯 Next Trading Day Forecast</div>', unsafe_allow_html=True)

    last_close  = float(df["Close"].iloc[-1])
    pred_change = res["next_price"] - last_close
    pred_pct    = (pred_change / last_close) * 100 if last_close != 0 else 0

    col_pred, col_info = st.columns([1, 2])

    with col_pred:
        direction_emoji = "📈" if pred_change >= 0 else "📉"
        st.markdown(f"""
<div class="pred-box">
    <div class="pred-label">Estimated Price for {res["next_date"]}</div>
    <div class="pred-price">₹{res["next_price"]:,.2f}</div>
    <div class="pred-label">{direction_emoji} {pred_pct:+.2f}% vs latest close (₹{last_close:,.2f})</div>
</div>
""", unsafe_allow_html=True)

    with col_info:
        st.markdown(f"""
        **How this forecast was made:**
        - The LSTM model was trained on {res['epochs_ran']} epochs of historical data.
        - It used the **last {res['lookback']} trading days** of closing prices as input.
        - Scaled prediction converted back to ₹ using MinMaxScaler.
        - This is a **mathematical estimate** — not a financial guarantee.
        """)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("MAE",  f"₹{res['lstm_metrics']['mae']:,.2f}")
    with m2:
        st.metric("RMSE", f"₹{res['lstm_metrics']['rmse']:,.2f}")
    with m3:
        st.metric("MAPE", f"{res['lstm_metrics']['mape']:.2f}%")
    with m4:
        st.metric("Epochs Trained", res["epochs_ran"])

    st.markdown('<div class="section-header">📊 Results Visualisation</div>', unsafe_allow_html=True)

    chart_tab1, chart_tab2, chart_tab3 = st.tabs([
        "Actual vs Predicted",
        "Full History + Predictions",
        "Training Loss",
    ])

    with chart_tab1:
        fig = plot_actual_vs_predicted(
            res["results_df"],
            ticker,
            next_day_pred=res["next_price"],
            next_day_date=res["next_date"],
        )
        st.plotly_chart(fig, use_container_width=True)

    with chart_tab2:
        fig2 = plot_full_prediction_overlay(df, res["results_df"], ticker)
        st.plotly_chart(fig2, use_container_width=True)

    with chart_tab3:
        fig3 = plot_training_loss(res["history"])
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    with st.expander("📋 View Prediction Data Table"):
        st.dataframe(
            res["results_df"].round(2).sort_values("Date", ascending=False),
            use_container_width=True,
            height=300,
        )
        pred_csv = res["results_df"].to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Predictions CSV",
            data=pred_csv,
            file_name=f"{ticker}_predictions.csv",
            mime="text/csv",
        )
else:
    st.markdown("---")
    st.info("👆 Configure parameters above and click **🚀 Train LSTM Model** to start.")
