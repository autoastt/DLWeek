import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime

# Load tickers and stock price data
stock_df = pd.read_csv("./data/nasdaq100_tickers.csv")
stock_price = pd.read_csv("./data/nasdaq100_stock_prices.csv", parse_dates=["date"])

# Ensure the first unnamed index column is removed (if it exists)
if stock_price.columns[0] == "Unnamed: 0":
    stock_price = stock_price.drop(columns=["Unnamed: 0"])

# Extract all tickers from the CSV for dropdown
tickers_list = stock_df["Ticker"].tolist()

# Keep fixed top stocks for sidebar graphs
top_stocks = ["AAPL", "INTC", "MSFT", "GOOGL", "CSCO", "TSLA", "NVDA", "AMZN", "ON", "PYPL"]

# Mapping tickers to company names
ticker_display = {row.Ticker: f"{row.Ticker} - {row.Company}" for _, row in stock_df.iterrows()}

# Set up Streamlit page
st.set_page_config(page_title="ThorEMore - AI Trading Dashboard", layout="wide")

# Sidebar Header
st.sidebar.header("ThorEMore: AI-Powered Trading")
st.sidebar.markdown("This app provides real-time trading insights using AI-powered models (LSTM + Transformer + RL).")

# Sidebar: Stock Selection (Dropdown now includes all tickers)
st.sidebar.subheader("📌 Select Stock for Analysis")
ticker = st.sidebar.selectbox("Choose a Stock:", tickers_list, format_func=lambda x: ticker_display[x])

# Date Selection
start_date = st.sidebar.date_input("📅 Start Date", datetime.date(2020, 1, 1))
end_date = st.sidebar.date_input("📅 End Date", datetime.date.today())

# Function to Fetch Data from CSV
@st.cache_data
def get_stock_data(ticker, start, end):
    df = stock_price[(stock_price["ticker"] == ticker) & 
                     (stock_price["date"] >= pd.Timestamp(start)) & 
                     (stock_price["date"] <= pd.Timestamp(end))]
    df = df.sort_values("date").reset_index(drop=True)  # Ensure proper ordering & remove index column
    return df

# Load Stock Data
st.subheader(f"📊 Stock Data for {ticker_display[ticker]}")
data = get_stock_data(ticker, start_date, end_date)

# Ensure data is valid
if data.empty:
    st.warning(f"⚠️ No valid data found for {ticker}. Try a different stock or date range.")
else:
    # Display Latest 5 Days (without extra index column)
    st.dataframe(data)

    # Plot Closing Price
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["date"], y=data["close"], mode="lines", name="Close Price", line=dict(color="blue")))
    fig.update_layout(title=f"📈 Closing Price of {ticker_display[ticker]}", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig)

    # Feature Engineering (Technical Indicators)
    data["Returns"] = data["close"].pct_change()
    data["SMA_50"] = data["close"].rolling(window=50).mean()
    data["SMA_200"] = data["close"].rolling(window=200).mean()
    data.dropna(inplace=True)

    # Remove extra index column from processed data
    data.reset_index(drop=True, inplace=True)

    # Show Processed Data
    st.subheader("🛠 Feature Engineering & Indicators")
    st.dataframe(data[["date", "close", "SMA_50", "SMA_200", "Returns"]].tail(10))

    # AI Trading Model Predictions (Mocked)
    st.subheader("🤖 AI Trading Model Predictions")
    mock_signals = np.random.choice(["BUY", "HOLD", "SELL"], size=len(data))
    data["Signal"] = mock_signals
    st.dataframe(data[["date", "close", "Signal"]].tail(10))

    # **Backtesting (Cumulative Returns)**
    st.subheader("📊 Backtesting Performance")
    data["Strategy Returns"] = np.where(data["Signal"] == "BUY", data["Returns"], 0).cumsum()
    data["Benchmark Returns"] = data["Returns"].cumsum()

    # Plot Backtest Results
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["date"], y=data["Strategy Returns"], mode="lines", name="AI Strategy", line=dict(color="green")))
    fig.add_trace(go.Scatter(x=data["date"], y=data["Benchmark Returns"], mode="lines", name="Nasdaq-100 Benchmark", line=dict(color="red")))
    fig.update_layout(title="Cumulative Returns Comparison", xaxis_title="Date", yaxis_title="Cumulative Returns")
    st.plotly_chart(fig)

# Sidebar: Market Trends (Only for Top Stocks)
st.sidebar.subheader("🔥 Market Trends")

@st.cache_data
def fetch_stock_info(tickers):
    stock_info = {}
    for ticker in tickers:
        df = stock_price[stock_price["ticker"] == ticker].sort_values("date")

        if df.empty:
            continue  # Skip if no valid data

        valid_data = df["close"].dropna()
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

# Fetch stock data for sidebar graphs (Only for Top Stocks)
stock_data = fetch_stock_info(top_stocks)

# Display Stock Info in Sidebar (Only for Top Stocks)
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
        st.sidebar.markdown(f"### {ticker_display[ticker]} - ${formatted_price}")
        st.sidebar.markdown(change_box, unsafe_allow_html=True)

        # ✅ Sidebar Mini Graph (Ensure valid data)
        if len(info["data"]) > 3:
            last_10_days = info["data"].iloc[-20:]
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

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Developed by ThorEMore Team 🚀")
