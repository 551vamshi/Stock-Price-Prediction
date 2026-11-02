
"""
pages/3_Model_Performance.py
----------------------------
Page 4: Model Performance & Evaluation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

import streamlit as st
import pandas as pd
import numpy as np
from auth import require_auth

st.set_page_config(
    page_title="Model Performance | Stock Predictor",
    page_icon="📉",
    layout="wide",
)

# User Authentication
current_user = require_auth()

st.markdown("""
<style>
.main { background-color: #0d1117; }
.stApp { background-color: #0d1117; }
section[data-testid="stSidebar"] { background-color: #161b22 !important; border-right: 1px solid #30363d; }
.section-header { font-size: 1.15rem; font-weight: 600; color: #e5e7eb; border-left: 3px solid #00C853; padding-left: 10px; margin: 20px 0 10px 0; }
div[data-testid="stMetric"] { background: #1f2937; border-radius: 10px; padding: 12px; border: 1px solid #30363d; }
</style>
""", unsafe_allow_html=True)

if st.session_state.get("df") is None:
    st.warning("⚠️ Go to the Home page and fetch stock data first.")
    st.stop()

if st.session_state.get("model_results") is None:
    st.warning("⚠️ No model trained yet. Go to **🤖 Price Prediction** page and train the model first.")
    st.stop()

try:
    from src.evaluation import compare_models
    from src.visualization import plot_actual_vs_predicted, plot_metrics_comparison
except (ImportError, ModuleNotFoundError):
    from evaluation import compare_models
    from visualization import plot_actual_vs_predicted, plot_metrics_comparison

df     = st.session_state.df
ticker = st.session_state.ticker
res    = st.session_state.model_results

lstm_m = res["lstm_metrics"]
base_m = res["baseline_metrics"]
comp   = compare_models(lstm_m, base_m)

st.markdown(f"## 📉 Model Performance — **{ticker}**")
st.markdown("---")

if comp["lstm_beats_baseline"]:
    st.success(
        "✅ **The LSTM model outperforms the naïve baseline!** "
        "The deep learning approach adds predictive value beyond simply "
        "using yesterday's price as today's prediction."
    )
else:
    st.warning(
        "⚠️ The LSTM model did not clearly outperform the naïve baseline on RMSE. "
        "This is common with highly volatile stocks or insufficient training data."
    )

st.markdown("")
st.markdown('<div class="section-header">📊 LSTM vs Naïve Baseline</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

def delta_str(imp):
    if imp is None:
        return "N/A"
    return f"↓ {imp:.1f}% better" if imp > 0 else f"↑ {abs(imp):.1f}% worse"

with col1:
    st.markdown("""**Mean Absolute Error (MAE)**""")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("LSTM MAE", f"₹{lstm_m['mae']:,.2f}")
    with c2:
        st.metric("Baseline MAE", f"₹{base_m['mae']:,.2f}")
    imp = comp["mae_improvement_pct"]
    if imp is not None and imp > 0:
        st.success(delta_str(imp))
    elif imp is not None:
        st.error(delta_str(imp))

with col2:
    st.markdown("""**Root Mean Squared Error (RMSE)**""")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("LSTM RMSE", f"₹{lstm_m['rmse']:,.2f}")
    with c2:
        st.metric("Baseline RMSE", f"₹{base_m['rmse']:,.2f}")
    imp = comp["rmse_improvement_pct"]
    if imp is not None and imp > 0:
        st.success(delta_str(imp))
    elif imp is not None:
        st.error(delta_str(imp))

with col3:
    st.markdown("""**Mean Absolute Percentage Error (MAPE)**""")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("LSTM MAPE", f"{lstm_m['mape']:.2f}%")
    with c2:
        st.metric("Baseline MAPE", f"{base_m['mape']:.2f}%")
    imp = comp["mape_improvement_pct"]
    if imp is not None and imp > 0:
        st.success(delta_str(imp))
    elif imp is not None:
        st.error(delta_str(imp))

st.markdown("")
with st.expander("📚 Understanding These Metrics"):
    st.markdown("""
    | Metric | Formula | Unit | Interpretation |
    |--------|---------|------|----------------|
    | **MAE** | mean(|actual - predicted|) | ₹ | Average absolute error in rupees. |
    | **RMSE** | √mean((actual - predicted)²) | ₹ | Penalises large errors more. |
    | **MAPE** | mean(|actual - predicted| / actual) × 100 | % | Relative error percentage. |
    """)

st.markdown('<div class="section-header">📊 Visual Comparison</div>', unsafe_allow_html=True)
fig_bar = plot_metrics_comparison(lstm_m, base_m)
st.plotly_chart(fig_bar, use_container_width=True)

st.markdown('<div class="section-header">📈 Actual vs Predicted Price</div>', unsafe_allow_html=True)
fig_pred = plot_actual_vs_predicted(
    res["results_df"],
    ticker,
    next_day_pred=res["next_price"],
    next_day_date=res["next_date"],
)
st.plotly_chart(fig_pred, use_container_width=True)

st.markdown("---")
with st.expander("📋 Full Metrics Summary Table"):
    summary_data = {
        "Metric": ["MAE (₹)", "RMSE (₹)", "MAPE (%)", "Samples"],
        "LSTM Model": [
            f"₹{lstm_m['mae']:,.2f}",
            f"₹{lstm_m['rmse']:,.2f}",
            f"{lstm_m['mape']:.2f}%",
            lstm_m["num_samples"],
        ],
        "Naïve Baseline": [
            f"₹{base_m['mae']:,.2f}",
            f"₹{base_m['rmse']:,.2f}",
            f"{base_m['mape']:.2f}%",
            base_m.get("num_samples", len(res["actual_prices"])),
        ],
    }
    st.table(pd.DataFrame(summary_data))
