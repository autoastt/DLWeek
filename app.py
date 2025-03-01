import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import requests

# Set app title
st.set_page_config(page_title="ThorEMore - AI Trading Dashboard", layout="wide")

# Sidebar Header
st.sidebar.header("ThorEMore: AI-Powered Trading")
st.sidebar.markdown("This app provides real-time trading insights using AI-powered models (LSTM + Transformer + RL).")

# Predefined Nasdaq-100 tickers 
# Need to update with actual Nasdaq-100 tickers
nasdaq_100_tickers = ["AAPL", "MSFT", "META", "GOOGL", "AMZN", "TSLA", "NVDA", "NFLX", "AMD", "INTC"]

# Sidebar: Dropdown for stock selection
st.sidebar.subheader("Select Stock for Analysis")
ticker = st.sidebar.selectbox("Choose a Nasdaq-100 Ticker:", nasdaq_100_tickers, index=0)
start_date = st.sidebar.date_input("Start Date", datetime.date(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

# Function to Fetch Data
@st.cache_data
def get_nasdaq100_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    return df

# Load Data
st.subheader(f"📊 Stock Data for {ticker}")
data = get_nasdaq100_data(ticker, start_date, end_date)

# Show only the latest 5 days
st.dataframe(data)

# Plot stock closing price using Plotly
# Ensure index is in datetime format
# Flatten the MultiIndex columns
data.columns = data.columns.get_level_values(0)  # Keeps only the first level (e.g., 'Close', 'High')

# Ensure data exists before plotting
if not data.empty and 'Close' in data.columns:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data['Close'], 
        mode='lines', 
        name='Close Price', 
        line=dict(color='blue')
    ))

    fig.update_layout(
        title=f"Closing Price of {ticker}",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis=dict(
            type="date",
            showgrid=True,
            tickangle=45
        ),
        yaxis=dict(showgrid=True),
    )

    st.plotly_chart(fig)
else:
    st.warning(f"No valid closing price data found for {ticker}. Check the ticker or try a different date range.")




# Feature Engineering (Technical Indicators)
data['Returns'] = data['Close'].pct_change()
data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['SMA_200'] = data['Close'].rolling(window=200).mean()
data.dropna(inplace=True)

# Show Processed Data
st.subheader("📈 Feature Engineering & Indicators")
st.dataframe(data[['Close', 'SMA_50', 'SMA_200', 'Returns']].tail(5))

# AI Trading Model Predictions (Mocked)
st.subheader("🤖 AI Trading Model Predictions")
st.markdown("(Mocked Prediction Model - Replace with actual LSTM/Transformer/RL)")
mock_signals = np.random.choice(["BUY", "HOLD", "SELL"], size=len(data))
data['Signal'] = mock_signals

# Show Model Predictions
st.dataframe(data[['Close', 'Signal']].tail(5))

# Backtesting (Cumulative Returns)
st.subheader("📊 Backtesting Performance")
data['Strategy Returns'] = np.where(data['Signal'] == "BUY", data['Returns'], 0).cumsum()
data['Benchmark Returns'] = data['Returns'].cumsum()

# Plot Backtest Results
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Strategy Returns'], mode='lines', name='AI Strategy', line=dict(color='green')))
fig.add_trace(go.Scatter(x=data.index, y=data['Benchmark Returns'], mode='lines', name='Nasdaq-100 Benchmark', line=dict(color='red')))
fig.update_layout(title="Cumulative Returns Comparison", xaxis_title="Date", yaxis_title="Cumulative Returns")
st.plotly_chart(fig)

# API Interface (Mocked)
st.subheader("🔌 API Access for Live Trading")
st.markdown("Use the API to fetch real-time trading signals.")
st.code("""
GET /api/predict?ticker=AAPL
Response:
{
    "ticker": "AAPL",
    "signal": "BUY",
    "confidence": 0.85
}
""", language='json')

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Developed by ThorEMore Team 🚀")
