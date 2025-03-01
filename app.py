import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import datetime

# stock_df = pd.read_csv("./data/nasdaq100_tickers.csv.csv")
# Set up Streamlit page
st.set_page_config(page_title="ThorEMore - AI Trading Dashboard", layout="wide")

# Sidebar Header
st.sidebar.header("ThorEMore: AI-Powered Trading")
st.sidebar.markdown("This app provides real-time trading insights using AI-powered models (LSTM + Transformer + RL).")

# Fixed list of tickers
top_stocks = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]

# Sidebar: Stock Selection
st.sidebar.subheader("📌 Select Stock for Analysis")
ticker = st.sidebar.selectbox("Choose a Stock:", top_stocks, index=0)
start_date = st.sidebar.date_input("📅 Start Date", datetime.date(2020, 1, 1))
end_date = st.sidebar.date_input("📅 End Date", datetime.date.today())

# Function to Fetch Data
@st.cache_data
def get_stock_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    return df

# Load Stock Data
st.subheader(f"📊 Stock Data for {ticker}")
data = get_stock_data(ticker, start_date, end_date)
data.columns = data.columns.get_level_values(0)

# Ensure data is valid
if data.empty or "Close" not in data.columns:
    st.warning(f"⚠️ No valid data found for {ticker}. Try a different stock or date range.")
else:
    # Display Latest 5 Days
    st.dataframe(data.tail(5))

    # Plot Closing Price
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["Close"], mode="lines", name="Close Price", line=dict(color="blue")))
    fig.update_layout(title=f"📈 Closing Price of {ticker}", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig)

    # Feature Engineering (Technical Indicators)
    data["Returns"] = data["Close"].pct_change()
    data["SMA_50"] = data["Close"].rolling(window=50).mean()
    data["SMA_200"] = data["Close"].rolling(window=200).mean()
    data.dropna(inplace=True)

    # Show Processed Data
    st.subheader("🛠 Feature Engineering & Indicators")
    st.dataframe(data[["Close", "SMA_50", "SMA_200", "Returns"]].tail(10))

    # AI Trading Model Predictions (Mocked)
    st.subheader("🤖 AI Trading Model Predictions")
    mock_signals = np.random.choice(["BUY", "HOLD", "SELL"], size=len(data))
    data["Signal"] = mock_signals
    st.dataframe(data[["Close", "Signal"]].tail(10))

    # Backtesting (Cumulative Returns)
    st.subheader("📊 Backtesting Performance")
    data["Strategy Returns"] = np.where(data["Signal"] == "BUY", data["Returns"], 0).cumsum()
    data["Benchmark Returns"] = data["Returns"].cumsum()

    # Plot Backtest Results
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["Strategy Returns"], mode="lines", name="AI Strategy", line=dict(color="green")))
    fig.add_trace(go.Scatter(x=data.index, y=data["Benchmark Returns"], mode="lines", name="Nasdaq-100 Benchmark", line=dict(color="red")))
    fig.update_layout(title="Cumulative Returns Comparison", xaxis_title="Date", yaxis_title="Cumulative Returns")
    st.plotly_chart(fig)

# Sidebar: Market Trends
st.sidebar.subheader("🔥 Market Trends")

@st.cache_data
def fetch_stock_info(tickers):
    stock_info = {}
    for ticker in tickers:
        df = yf.download(ticker, period="7d", interval="1d")

        # Fix MultiIndex issue if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)  # Flatten MultiIndex

        if df.empty or "Close" not in df.columns:
            continue  # Skip if no valid data

        valid_data = df["Close"].dropna()
        if len(valid_data) < 2:
            continue  # Skip if not enough data

        latest_close = float(valid_data.iloc[-1])  # Ensure it's a float
        prev_close = float(valid_data.iloc[-2])  # Ensure it's a float
        price_change = latest_close - prev_close

        stock_info[ticker] = {
            "latest_price": latest_close,
            "price_change": price_change,
            "data": valid_data
        }
    return stock_info

# Fetch stock data for sidebar
stock_data = fetch_stock_info(top_stocks)

# Display Stock Info in Sidebar
for ticker, info in stock_data.items():
    if "latest_price" in info:
        latest_price = float(info["latest_price"])  # Ensure it's a float
        price_change = float(info["price_change"])  # Ensure it's a float

        # ✅ Format the price nicely (e.g., 5,954.50)
        formatted_price = f"{latest_price:,.2f}"

        # ✅ Format the profit/loss change with a green or red box
        change_color = "green" if price_change > 0 else "red"
        change_box = f"<div style='background-color:{change_color}; color:white; padding:5px; border-radius:5px; display:inline-block;'> {price_change:+.2f} </div>"

        # 🔹 Display stock symbol and latest price
        st.sidebar.markdown(f"### {ticker} - ${formatted_price}")
        st.sidebar.markdown(change_box, unsafe_allow_html=True)

        # ✅ Sidebar Mini Graph (Ensure valid data)
        if len(info["data"]) > 3:
            last_10_days = info["data"].iloc[-100:]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=last_10_days.index,
                y=last_10_days.values,
                mode="lines",
                line=dict(color=change_color, width=5)
            ))
            fig.update_layout(
                height=100,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(visible=False),
                yaxis=dict(visible=False)
            )
            st.sidebar.plotly_chart(fig, use_container_width=True)
        else:
            st.sidebar.warning(f"⚠️ Not enough data to plot {ticker}.")

# Debugging Section
st.subheader("📊 Data Preview")
if not data.empty:
    st.write(data.head())

    # Verify Close Prices
    if "Close" in data.columns:
        st.success(f"✅ Data looks good! {data['Close'].notna().sum()} valid Close prices.")
    else:
        st.error("❌ 'Close' column not found! Data might be corrupted.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Developed by ThorEMore Team 🚀")
