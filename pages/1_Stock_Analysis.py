"""
pages/1_Stock_Analysis.py
-------------------------
Page 2: Stock Market Analysis
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

import streamlit as st

try:
    from src.data_analysis import get_summary_statistics, get_price_change_today
    from src.visualization import (
        plot_candlestick,
        plot_moving_averages,
        plot_volume,
        plot_daily_returns,
        plot_volatility,
        plot_bollinger_bands,
        plot_rsi,
        plot_closing_price,
    )
except (ImportError, ModuleNotFoundError):
    from data_analysis import get_summary_statistics, get_price_change_today
    from visualization import (
        plot_candlestick,
        plot_moving_averages,
        plot_volume,
        plot_daily_returns,
        plot_volatility,
        plot_bollinger_bands,
        plot_rsi,
        plot_closing_price,
    )

st.set_page_config(
    page_title="Stock Analysis | Stock Predictor",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
.main { background-color: #0d1117; }
.stApp { background-color: #0d1117; }
section[data-testid="stSidebar"] { background-color: #161b22 !important; border-right: 1px solid #30363d; }
.section-header { font-size: 1.15rem; font-weight: 600; color: #e5e7eb; border-left: 3px solid #2196F3; padding-left: 10px; margin: 20px 0 10px 0; }
div[data-testid="stMetric"] { background: #1f2937; border-radius: 10px; padding: 12px; border: 1px solid #30363d; }
</style>
""", unsafe_allow_html=True)

if st.session_state.get("df") is None:
    st.warning("⚠️ No data loaded. Go to the **Home** page and fetch stock data first.")
    st.stop()

df   = st.session_state.df
df_i = st.session_state.df_indicators
ticker = st.session_state.ticker

st.markdown(f"## 📊 Stock Market Analysis — **{ticker}**")
st.caption(f"Showing data from {df['Date'].min().date()} to {df['Date'].max().date()}")
st.markdown("---")

stats = get_summary_statistics(df)
change_amt, change_pct, direction = get_price_change_today(df)

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Latest Close", f"₹{stats['latest_close']:,.2f}",
              delta=f"{change_pct:+.2f}%",
              delta_color="normal" if direction == "up" else "inverse")
with c2:
    st.metric("52W High", f"₹{stats['all_time_high']:,.2f}")
with c3:
    st.metric("52W Low",  f"₹{stats['all_time_low']:,.2f}")
with c4:
    st.metric("Avg Daily Return", f"{stats['avg_daily_return']:+.4f}%")
with c5:
    st.metric("Daily Volatility", f"{stats['std_daily_return']:.4f}%")

st.markdown("")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🕯️ Candlestick",
    "📈 Moving Averages",
    "📊 Volume",
    "📉 Daily Returns",
    "〰️ Volatility",
    "🎯 Bollinger Bands",
    "⚡ RSI",
])

with tab1:
    st.markdown('<div class="section-header">🕯️ Candlestick Chart</div>', unsafe_allow_html=True)
    st.markdown("""
    **How to read:**  
    - 🟢 **Green candle** → Price closed HIGHER than it opened (bullish day)  
    - 🔴 **Red candle** → Price closed LOWER than it opened (bearish day)  
    - **Top wick** = Highest price of the day | **Bottom wick** = Lowest price
    """)
    fig = plot_candlestick(df_i, ticker)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown('<div class="section-header">📈 Moving Averages</div>', unsafe_allow_html=True)
    st.markdown("""
    - **🟠 MA-20** (short-term trend): Average of the last 20 trading days  
    - **🟢 MA-50** (medium-term trend): Average of the last 50 trading days  
    - When MA-20 crosses **above** MA-50 → **Golden Cross** (bullish signal)  
    - When MA-20 crosses **below** MA-50 → **Death Cross** (bearish signal)
    """)
    fig = plot_moving_averages(df_i, ticker)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown('<div class="section-header">📊 Trading Volume</div>', unsafe_allow_html=True)
    st.markdown("""
    Volume = number of shares traded on a given day.  
    - 🟢 **Green bar** → Price went up that day  
    - 🔴 **Red bar** → Price went down that day  
    - **High volume + price rise** = Strong bullish signal  
    - **Low volume + price rise** = Weak/uncertain signal
    """)
    if "Volume" in df_i.columns:
        fig = plot_volume(df_i, ticker)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Daily Volume", f"{stats['avg_volume']:,}" if stats['avg_volume'] else "N/A")
        with col2:
            st.metric("Latest Volume", f"{stats['latest_volume']:,}" if stats['latest_volume'] else "N/A")
        with col3:
            max_vol = int(df["Volume"].max()) if "Volume" in df.columns else 0
            st.metric("Highest Volume", f"{max_vol:,}")
    else:
        st.warning("Volume data not available in the dataset.")

with tab4:
    st.markdown('<div class="section-header">📉 Daily Returns Distribution</div>', unsafe_allow_html=True)
    st.markdown("""
    **Daily Return** = (Today's Close - Yesterday's Close) / Yesterday's Close × 100  
    - Positive returns → stock went up  
    - Negative returns → stock went down  
    - The histogram shows how frequently each return magnitude occurs
    """)
    fig = plot_daily_returns(df_i, ticker)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    daily_ret = df_i["Daily_Return"].dropna()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Daily Return", f"{daily_ret.mean():.4f}%")
    with col2:
        st.metric("Std Dev (Volatility)", f"{daily_ret.std():.4f}%")
    with col3:
        st.metric("Best Day", f"{daily_ret.max():.2f}%")
    with col4:
        st.metric("Worst Day", f"{daily_ret.min():.2f}%")

with tab5:
    st.markdown('<div class="section-header">〰️ Rolling Volatility</div>', unsafe_allow_html=True)
    st.markdown("""
    **Volatility** = Standard deviation of daily returns over a 20-day rolling window.  
    - High peaks = periods of market uncertainty or major events  
    - Low periods = calm, stable markets
    """)
    fig = plot_volatility(df_i, ticker)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

with tab6:
    st.markdown('<div class="section-header">🎯 Bollinger Bands</div>', unsafe_allow_html=True)
    st.markdown("""
    **Bollinger Bands** = MA-20 ± 2 standard deviations  
    - Price near **Upper Band** → possibly overbought (may fall)  
    - Price near **Lower Band** → possibly oversold (may rise)
    """)
    fig = plot_bollinger_bands(df_i, ticker)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Bollinger Bands require more data (at least 20 trading days).")

with tab7:
    st.markdown('<div class="section-header">⚡ Relative Strength Index (RSI)</div>', unsafe_allow_html=True)
    st.markdown("""
    **RSI** measures the speed and magnitude of price changes (0–100).  
    - **RSI > 70** → Stock is *overbought* (price may correct downward)  
    - **RSI < 30** → Stock is *oversold* (price may bounce upward)  
    - Period used: **14 trading days**
    """)
    fig = plot_rsi(df_i, ticker)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
        latest_rsi = df_i["RSI"].iloc[-1]
        rsi_status = "🔴 Overbought" if latest_rsi > 70 else ("🟢 Oversold" if latest_rsi < 30 else "⚪ Neutral")
        st.metric("Current RSI", f"{latest_rsi:.2f}", delta=rsi_status, delta_color="off")

st.markdown("---")
with st.expander("📋 View Data with Indicators"):
    display_cols = [c for c in ["Date","Open","High","Low","Close","Volume",
                                "MA_20","MA_50","Daily_Return","Volatility","RSI"]
                    if c in df_i.columns]
    st.dataframe(
        df_i[display_cols].sort_values("Date", ascending=False).head(50).round(2),
        use_container_width=True, height=350,
    )
