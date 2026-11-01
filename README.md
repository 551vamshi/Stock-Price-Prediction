# 📈 Stock Price Prediction Using Deep Learning and Data Analytics

> **B.Tech CSE Mini Project** | Python · Streamlit · TensorFlow · Plotly

---

## 📌 Introduction

This project demonstrates the application of **Deep Learning (LSTM Neural Networks)** and **Data Analytics** to predict future stock prices based on historical market data.

The web application:
- Fetches real historical stock data from **Yahoo Finance**
- Performs comprehensive **technical analysis** (Moving Averages, RSI, Bollinger Bands)
- Trains an **LSTM (Long Short-Term Memory)** deep learning model
- Predicts the next trading day's estimated price
- Compares LSTM against a **naïve baseline** model

> ⚠️ **Disclaimer:** This project is for **educational purposes only**. Predictions are statistical estimates, not financial advice. Do not make real investment decisions based on this tool.

---

## ⚙️ Installation & Running the Application

### 1. Open the project in VS Code or Terminal
```bash
cd C:\stock-price-prediction
```

### 2. Install dependencies (if not already installed)
```bash
python3 -m pip install -r requirements.txt
```

### 3. Run the Streamlit Dashboard
```bash
python3 -m streamlit run app.py
```

The dashboard opens automatically in your browser at `http://localhost:8501`.

### 4. Run automated tests
```bash
python3 -m pytest tests/test_preprocessing.py -v
```

---

## 📂 Folder Structure

```
C:\stock-price-prediction\
│
├── app.py                    ← Home dashboard (main entry point)
├── requirements.txt          ← Python dependencies
├── README.md                 ← Project documentation
├── .gitignore                ← Git exclusion rules
│
├── src/                      ← Core Python modules
│   ├── __init__.py
│   ├── data_collection.py    ← Module 1: yfinance + CSV loader
│   ├── preprocessing.py      ← Module 2: scaling, sequences, split
│   ├── data_analysis.py      ← Module 3: indicators, statistics
│   ├── lstm_model.py         ← Module 4: LSTM build, train, predict
│   ├── evaluation.py         ← Module 5: MAE, RMSE, MAPE, baseline
│   └── visualization.py      ← Module 6: all Plotly chart functions
│
├── pages/                    ← Streamlit multi-page app
│   ├── 1_Stock_Analysis.py   ← Candlestick, MA, RSI, Volume
│   ├── 2_Price_Prediction.py ← Train LSTM, next-day forecast
│   ├── 3_Model_Performance.py← Metrics and comparison
│   └── 4_About_Project.py    ← Theory, methodology, limitations
│
├── data/                     ← CSV uploads / sample data
│   ├── .gitkeep
│   └── sample_RELIANCE.csv   ← Sample data for offline testing
│
├── models/                   ← Saved trained models (.keras)
│   └── .gitkeep
│
└── tests/                    ← Unit tests
    └── test_preprocessing.py
```

---

*Built with ❤️ using Python · Streamlit · TensorFlow · Plotly*
