"""
app.py
------
Main entry point for the Stock Price Prediction Dashboard.

Run with:
    streamlit run app.py
"""

import sys
import os

# Ensure src/ is on the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

try:
    from src.data_collection import (
        fetch_stock_data,
        load_csv_data,
        get_stock_info,
        get_default_date_range,
        POPULAR_STOCKS,
    )
    from src.data_analysis import (
        compute_all_indicators,
        get_summary_statistics,
        get_price_change_today,
    )
    from src.visualization import plot_closing_price
except (ImportError, ModuleNotFoundError):
    from data_collection import (
        fetch_stock_data,
        load_csv_data,
        get_stock_info,
        get_default_date_range,
        POPULAR_STOCKS,
    )
    from data_analysis import (
        compute_all_indicators,
        get_summary_statistics,
        get_price_change_today,
    )
    from visualization import plot_closing_price

st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Stock Price Prediction using Deep Learning — B.Tech CSE Mini Project",
    },
)

st.markdown("""
<style>
:root {
    --bg-primary:   #0d1117;
    --bg-secondary: #161b22;
    --bg-card:      #1f2937;
    --accent-green: #00C853;
    --accent-red:   #FF1744;
    --accent-gold:  #FFD700;
    --accent-blue:  #2196F3;
    --text-primary: #e5e7eb;
    --text-muted:   #9ca3af;
    --border:       #30363d;
}

.main { background-color: var(--bg-primary); }
.stApp { background-color: var(--bg-primary); }

section[data-testid="stSidebar"] {
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--border);
}

.section-header {
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--text-primary);
    border-left: 3px solid var(--accent-blue);
    padding-left: 10px;
    margin: 24px 0 12px 0;
}

.info-banner {
    background: linear-gradient(135deg, #1e3a5f 0%, #0d2137 100%);
    border: 1px solid #2563eb;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 20px;
}

.warning-banner {
    background: rgba(234, 179, 8, 0.1);
    border: 1px solid #eab308;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.85rem;
    color: #fde68a;
}

div[data-testid="stMetric"] {
    background: var(--bg-card);
    border-radius: 10px;
    padding: 14px 10px;
    border: 1px solid var(--border);
}
</style>
""", unsafe_allow_html=True)

if "df" not in st.session_state:
    st.session_state.df = None
if "ticker" not in st.session_state:
    st.session_state.ticker = "RELIANCE.NS"
if "stock_info" not in st.session_state:
    st.session_state.stock_info = {}
if "model_results" not in st.session_state:
    st.session_state.model_results = None
if "df_indicators" not in st.session_state:
    st.session_state.df_indicators = None

with st.sidebar:
    st.markdown("## 📈 Stock Selector")
    st.markdown("---")

    selection_mode = st.radio(
        "Choose stock by",
        options=["Popular Indian Stocks", "Custom Ticker"],
        horizontal=True,
    )

    if selection_mode == "Popular Indian Stocks":
        display_name = st.selectbox(
            "Select Stock",
            options=list(POPULAR_STOCKS.keys()),
            index=0,
        )
        ticker_input = POPULAR_STOCKS[display_name]
    else:
        ticker_input = st.text_input(
            "Enter Ticker Symbol",
            value="RELIANCE.NS",
            help="Examples: RELIANCE.NS, TCS.NS, AAPL, TSLA",
        ).strip().upper()

    st.markdown("---")
    st.markdown("**📅 Date Range**")

    default_start, default_end = get_default_date_range()
    start_date = st.date_input("Start Date",
                               value=datetime.strptime(default_start, "%Y-%m-%d"),
                               min_value=datetime(2000, 1, 1),
                               max_value=datetime.today())
    end_date   = st.date_input("End Date",
                               value=datetime.strptime(default_end, "%Y-%m-%d"),
                               min_value=datetime(2000, 1, 1),
                               max_value=datetime.today())

    st.markdown("---")
    st.markdown("**📂 CSV Fallback**")
    uploaded_file = st.file_uploader(
        "Upload CSV (when offline)",
        type=["csv"],
        help="Must have Date and Close columns.",
    )

    fetch_btn = st.button("🔄  Fetch Data", type="primary", use_container_width=True)

    st.markdown("---")
    st.caption("**📌 Navigation**")
    st.caption("Use the pages above to navigate:")
    st.caption("• 📊 Stock Analysis")
    st.caption("• 🤖 Price Prediction")
    st.caption("• 📉 Model Performance")
    st.caption("• ℹ️ About")

if fetch_btn or st.session_state.df is None:
    if uploaded_file is not None:
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            with st.spinner("Reading CSV file…"):
                df = load_csv_data(tmp_path)

            st.session_state.df = df
            st.session_state.ticker = uploaded_file.name.replace(".csv", "").upper()
            st.session_state.df_indicators = compute_all_indicators(df)
            st.session_state.stock_info = {"name": st.session_state.ticker}
            st.success(f"✅ Loaded {len(df):,} rows from CSV.")
        except Exception as e:
            st.error(f"❌ CSV Error: {e}")
    else:
        if start_date >= end_date:
            st.error("❌ Start date must be before End date.")
        else:
            with st.spinner(f"Fetching {ticker_input} from Yahoo Finance…"):
                try:
                    df = fetch_stock_data(
                        ticker_input,
                        start_date.strftime("%Y-%m-%d"),
                        end_date.strftime("%Y-%m-%d"),
                    )
                    info = get_stock_info(ticker_input)

                    st.session_state.df = df
                    st.session_state.ticker = ticker_input
                    st.session_state.stock_info = info
                    st.session_state.df_indicators = compute_all_indicators(df)
                    st.session_state.model_results = None

                    st.success(f"✅ Fetched {len(df):,} trading days for **{ticker_input}**.")
                except Exception as e:
                    st.error(f"❌ {e}")

st.markdown("""
<div class="info-banner">
    <h1 style="margin:0; font-size:1.8rem;">📈 Stock Price Prediction</h1>
    <p style="margin:6px 0 0 0; color:#93c5fd; font-size:0.95rem;">
        Deep Learning & Data Analytics Dashboard &nbsp;|&nbsp;
        B.Tech CSE Mini Project
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="warning-banner">
    ⚠️ <strong>Disclaimer:</strong> This project is for educational purposes only.
    Predictions are model estimates and <strong>NOT</strong> financial advice.
</div>
""", unsafe_allow_html=True)

st.markdown("")

if st.session_state.df is not None:
    df   = st.session_state.df
    info = st.session_state.stock_info
    df_i = st.session_state.df_indicators
    stats = get_summary_statistics(df)
    change_amt, change_pct, direction = get_price_change_today(df)

    col_name, col_sector = st.columns([3, 1])
    with col_name:
        company_name = info.get("name", st.session_state.ticker)
        st.markdown(f"### 🏢 {company_name}")
        st.caption(f"**{st.session_state.ticker}** | {info.get('exchange', 'NSE')} | {info.get('currency', 'INR')}")
    with col_sector:
        sector = info.get("sector", "N/A")
        if sector != "N/A":
            st.info(f"**Sector:** {sector}")

    st.markdown("---")
    st.markdown('<div class="section-header">📊 Quick Statistics</div>', unsafe_allow_html=True)

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    delta_color = "normal" if direction == "up" else ("inverse" if direction == "down" else "off")

    with k1:
        st.metric(
            label="Latest Close",
            value=f"₹{stats['latest_close']:,.2f}",
            delta=f"{change_pct:+.2f}% today",
            delta_color=delta_color,
        )
    with k2:
        st.metric(label="All-Time High", value=f"₹{stats['all_time_high']:,.2f}")
    with k3:
        st.metric(label="All-Time Low",  value=f"₹{stats['all_time_low']:,.2f}")
    with k4:
        st.metric(label="Avg Close",     value=f"₹{stats['avg_close']:,.2f}")
    with k5:
        st.metric(
            label="Total Return",
            value=f"{stats['total_return_pct']:+.2f}%",
            delta_color="normal" if stats['total_return_pct'] > 0 else "inverse",
        )
    with k6:
        st.metric(label="Trading Days", value=f"{stats['num_trading_days']:,}")

    st.markdown("")
    st.markdown('<div class="section-header">📉 Price History Overview</div>', unsafe_allow_html=True)
    fig = plot_closing_price(df, st.session_state.ticker)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("🗂️ View Raw Data Table"):
        st.dataframe(
            df.sort_values("Date", ascending=False).head(30),
            use_container_width=True,
            height=300,
        )
        csv_str = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Full Data as CSV",
            data=csv_str,
            file_name=f"{st.session_state.ticker}_data.csv",
            mime="text/csv",
        )
else:
    st.markdown("---")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("### 🚀 Getting Started")
        st.markdown("""
1. **Select a stock** from the sidebar.
2. **Choose a date range** (default: last 3 years).
3. Click **🔄 Fetch Data** to download from Yahoo Finance.
4. Explore using the sidebar pages.
        """)
    with c2:
        st.markdown("### 📋 Sample CSV Format")
        sample_csv = pd.DataFrame({
            "Date":   ["2023-01-02", "2023-01-03", "2023-01-04"],
            "Open":   [2400.00, 2420.50, 2415.75],
            "High":   [2450.00, 2440.00, 2430.00],
            "Low":    [2390.00, 2410.00, 2400.00],
            "Close":  [2435.00, 2425.00, 2418.50],
            "Volume": [4500000, 3200000, 5100000],
        })
        st.dataframe(sample_csv, use_container_width=True)

st.markdown("---")
st.caption("Stock Price Prediction Using Deep Learning & Data Analytics | B.Tech CSE Mini Project")
