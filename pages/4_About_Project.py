"""
pages/4_About_Project.py
------------------------
Page 5: About This Project
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

import streamlit as st

st.set_page_config(
    page_title="About | Stock Predictor",
    page_icon="ℹ️",
    layout="wide",
)

st.markdown("""
<style>
.main { background-color: #0d1117; }
.stApp { background-color: #0d1117; }
section[data-testid="stSidebar"] { background-color: #161b22 !important; border-right: 1px solid #30363d; }
h1, h2, h3, h4 { color: #e5e7eb !important; }
.tech-badge {
    display: inline-block;
    background: #1f2937;
    border: 1px solid #374151;
    border-radius: 20px;
    padding: 4px 14px;
    margin: 4px;
    font-size: 0.85rem;
    color: #93c5fd;
}
.objective-card {
    background: #1f2937;
    border-left: 4px solid #2196F3;
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 10px;
}
.limitation-card {
    background: #1a1a2e;
    border-left: 4px solid #FF1744;
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 10px;
}
.future-card {
    background: #0d2137;
    border-left: 4px solid #00C853;
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("## ℹ️ About This Project")
st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Overview",
    "🏗️ Architecture",
    "🧠 LSTM Theory",
    "📊 Metrics",
    "🔮 Limitations & Future",
])

with tab1:
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
### 📌 Project Title
**Stock Price Prediction Using Deep Learning and Data Analytics**

### 📝 Introduction
This project demonstrates the application of **Deep Learning (LSTM neural networks)**
and **Data Analytics** to predict future stock prices based on historical market data.

### ❓ Problem Statement
Stock prices exhibit complex, non-linear, and sequential relationships that traditional linear models struggle to forecast.
This project uses an LSTM recurrent neural network to capture sequential price patterns.

### 🎯 Objectives
        """)

        objectives = [
            ("1️⃣", "Collect real historical stock market data using Yahoo Finance API."),
            ("2️⃣", "Perform comprehensive exploratory data analysis with technical indicators."),
            ("3️⃣", "Build and train an LSTM neural network for time-series price prediction."),
            ("4️⃣", "Evaluate model performance using MAE, RMSE, and MAPE metrics."),
            ("5️⃣", "Compare LSTM against a naïve baseline model."),
            ("6️⃣", "Present results through a professional interactive web dashboard."),
        ]

        for icon, obj in objectives:
            st.markdown(f"""
<div class="objective-card">
    <strong>{icon}</strong> {obj}
</div>
""", unsafe_allow_html=True)

    with col2:
        st.markdown("### 🛠️ Technologies Used")
        tech_categories = {
            "Language": ["Python 3.10+"],
            "Data Collection": ["yfinance"],
            "Data Processing": ["Pandas", "NumPy"],
            "Machine Learning": ["TensorFlow", "Keras", "Scikit-learn"],
            "Visualisation": ["Plotly", "Matplotlib"],
            "Web Dashboard": ["Streamlit"],
        }
        for category, techs in tech_categories.items():
            st.markdown(f"**{category}:**")
            badges = " ".join([f'<span class="tech-badge">{t}</span>' for t in techs])
            st.markdown(badges, unsafe_allow_html=True)
            st.markdown("")

with tab2:
    st.markdown("### 🏗️ Project Architecture")
    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────┐
    │                    USER (Web Browser)                       │
    └───────────────────────────┬─────────────────────────────────┘
                                │
    ┌───────────────────────────▼─────────────────────────────────┐
    │               STREAMLIT DASHBOARD (app.py + pages/)        │
    │   Home | Stock Analysis | Prediction | Performance | About  │
    └────┬────────────────┬────────────────┬────────────┬─────────┘
         │                │                │            │
    ┌────▼──────┐  ┌──────▼──────┐  ┌─────▼─────┐  ┌──▼──────────┐
    │   Data    │  │    Data     │  │   LSTM    │  │ Evaluation  │
    │Collection │  │  Analysis   │  │   Model   │  │  Module     │
    │(yfinance) │  │(indicators) │  │(Keras/TF) │  │(MAE,RMSE)  │
    └────┬──────┘  └──────┬──────┘  └─────┬─────┘  └─────────────┘
         │                │                │
    ┌────▼──────┐  ┌──────▼──────┐  ┌─────▼─────┐
    │  Raw CSV  │  │Preprocessed │  │  Trained  │
    │   Data    │  │  DataFrame  │  │   Model   │
    └───────────┘  └─────────────┘  └───────────┘
    ```
    """)

with tab3:
    st.markdown("""
### 🧠 LSTM — Theory & Methodology
- **Memory Cell:** Retains long-term state across many timesteps.
- **Forget Gate:** Decides what past information to discard.
- **Input Gate:** Decides what new information to store.
- **Output Gate:** Controls what information to pass to the next layer/output.

**Architecture:**
- Input: `(lookback, 1)`
- LSTM Layer 1: 64 units (`return_sequences=True`)
- Dropout: 0.2
- LSTM Layer 2: 32 units (`return_sequences=False`)
- Dropout: 0.2
- Dense Layer: 16 units (ReLU)
- Output Layer: 1 unit (predicted price)
    """)

with tab4:
    st.markdown("""
### 📊 Evaluation Metrics
- **MAE:** Mean Absolute Error in ₹
- **RMSE:** Root Mean Squared Error in ₹ (penalises larger outliers)
- **MAPE:** Mean Absolute Percentage Error (%)
- **Naïve Baseline:** Tomorrow's price = today's close ($y_t = y_{t-1}$)
    """)

with tab5:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ⚠️ Limitations")
        st.markdown("""
        - Stock markets are non-stationary and influenced by unexpected news.
        - Single feature (close price only) limits macroeconomic awareness.
        - Deep learning models require larger datasets for generalized production trading.
        """)
    with col2:
        st.markdown("### 🚀 Future Enhancements")
        st.markdown("""
        - Incorporate news sentiment analysis using NLP.
        - Multi-feature input (High, Low, Open, Volume, RSI).
        - Ensemble modelling with Transformer & ARIMA.
        """)

st.markdown("---")
st.info("B.Tech Computer Science & Engineering | Academic Project")
