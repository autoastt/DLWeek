import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime

# Load tickers and stock price data
stock_df = pd.read_csv("./data/nasdaq100_tickers.csv").sort_values(by="Ticker")
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

# --- Sidebar Layout ---
with st.sidebar:
    # st.markdown("<h2 style='font-size:20px;'>📊 Market Overview</h2>", unsafe_allow_html=True)
    st.markdown("<h2 style='font-size:22px;'>⚙️ Settings</h3>", unsafe_allow_html=True)
    # st.markdown("<p style='font-size:14px;'>Optimizing Adaptive Reinforcement Learning for Stock Trading: Smaller and Faster Models</p>", unsafe_allow_html=True)

    # Stock Selection Dropdown
    st.markdown("<h3 style='font-size:18px;'>📌 Select Stock for Analysis</h3>", unsafe_allow_html=True)
    selected_ticker = st.selectbox("Choose a Stock:", tickers_list, format_func=lambda x: ticker_display[x])

    # Date Selection (Moved back to Sidebar)
    st.markdown("<h3 style='font-size:18px;'>📅 Select Date Range</h3>", unsafe_allow_html=True)
    start_date = st.date_input("Start Date", datetime.date(2020, 1, 1))
    end_date = st.date_input("End Date", datetime.date.today())

    # --- Market Trends (Only for Top Stocks) ---
    st.markdown("<h3 style='font-size:18px;'>🔥 Market Trends</h3>", unsafe_allow_html=True)

    @st.cache_data
    def fetch_stock_info(tickers):
        stock_info = {}
        for ticker in tickers:
            df = stock_price[stock_price["ticker"] == ticker].sort_values("date")
            if df.empty:
                continue
            valid_data = df["close"].dropna()
            if len(valid_data) < 2:
                continue

            latest_close = float(valid_data.iloc[-1])
            prev_close = float(valid_data.iloc[-2])
            price_change = latest_close - prev_close

            stock_info[ticker] = {
                "latest_price": latest_close,
                "price_change": price_change,
                "data": valid_data
            }
        return stock_info

    stock_data = fetch_stock_info(top_stocks)

    # Display Stock Info in Sidebar (Only for Top Stocks)
    for ticker, info in stock_data.items():
        if "latest_price" in info:
            latest_price = float(info["latest_price"])
            price_change = float(info["price_change"])

            # Price Change Styling
            change_color = "green" if price_change > 0 else "red"
            change_box = f"<div style='background-color:{change_color}; color:white; padding:5px; border-radius:5px; display:inline-block;'> {price_change:+.2f} </div>"

            # Display stock symbol and latest price
            st.markdown(f"<h3 style='font-size:16px;'>{ticker_display[ticker]} - ${latest_price:,.2f}</h3>", unsafe_allow_html=True)
            st.markdown(change_box, unsafe_allow_html=True)

            # Sidebar Mini Graph (Ensure valid data)
            if len(info["data"]) > 3:
                last_10_days = info["data"].iloc[-20:]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=last_10_days.index,
                    y=last_10_days.values,
                    mode="lines",
                    line=dict(color=change_color, width=3)
                ))
                fig.update_layout(
                    height=100,
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False)
                )
                st.plotly_chart(fig, use_container_width=True)

# --- Main Layout ---
st.markdown("<h1 style='font-size:36px; padding-top:0px'>⚡️ ThorEMore</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size:16px;'>Optimizing Adaptive Reinforcement Learning for Stock Trading: Smaller and Faster Models</p>", unsafe_allow_html=True)

st.markdown(f"<h1 style='font-size:28px;'>📈 {ticker_display[selected_ticker]} Stock Overview</h1>", unsafe_allow_html=True)

# Timeframe Selection (Reversed Order)
timeframe = st.radio(
    "Select Timeframe:",
    ["ALL", "10Y", "5Y", "1Y", "6M", "3M", "1M"],  # Reordered options
    horizontal=True
)


# Fetch Stock Data
@st.cache_data
def get_stock_data(ticker):
    df = stock_price[(stock_price["ticker"] == ticker) & 
                     (stock_price["date"] >= pd.Timestamp(start_date)) & 
                     (stock_price["date"] <= pd.Timestamp(end_date))]
    df = df.sort_values("date").reset_index(drop=True)
    return df

data = get_stock_data(selected_ticker)

# Filter data based on selected timeframe
end_date = data["date"].max()
if timeframe == "1M":
    start_date = end_date - pd.DateOffset(months=1)
elif timeframe == "3M":
    start_date = end_date - pd.DateOffset(months=3)
elif timeframe == "6M":
    start_date = end_date - pd.DateOffset(months=6)
elif timeframe == "1Y":
    start_date = end_date - pd.DateOffset(years=1)
elif timeframe == "5Y":
    start_date = end_date - pd.DateOffset(years=5)
elif timeframe == "10Y":
    start_date = end_date - pd.DateOffset(years=10)
else:
    start_date = data["date"].min()

data = data[(data["date"] >= start_date) & (data["date"] <= end_date)]

# Ensure data is valid
if data.empty:
    st.warning(f"⚠️ No valid data found for {selected_ticker}. Try a different stock or date range.")
else:
    # --- Candlestick Chart ---
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=data["date"], open=data["open"], high=data["high"],
        low=data["low"], close=data["close"],
        name="OHLC", increasing_line_color="green", decreasing_line_color="red"
    ))

    # Moving Averages
    data["SMA_50"] = data["close"].rolling(window=50).mean()
    data["SMA_200"] = data["close"].rolling(window=200).mean()
    fig.add_trace(go.Scatter(x=data["date"], y=data["SMA_50"], mode="lines", name="SMA 50", line=dict(color="blue", width=1)))
    fig.add_trace(go.Scatter(x=data["date"], y=data["SMA_200"], mode="lines", name="SMA 200", line=dict(color="orange", width=1)))

    fig.update_layout(title=f"{ticker_display[selected_ticker]} - Stock Price", xaxis_title="Date", yaxis_title="Price", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- Feature Engineering Indicators ---
    st.markdown("<h2 style='font-size:24px;'>🔍 Percentage Change</h2>", unsafe_allow_html=True)
    data["Returns"] = data["close"].pct_change()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["date"], y=data["Returns"], mode="lines", name="Returns", line=dict(color="purple")))
    st.plotly_chart(fig, use_container_width=True)

    # --- Backtesting ---
    st.markdown("<h2 style='font-size:24px;'>📊 Backtesting Performance</h2>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='font-size:20px;'>Dual Moving Average Crossover Strategy</h3>", unsafe_allow_html=True)
    st.image(f'assets/dma/{selected_ticker}.png')
    
    dma = pd.read_csv("./assets/dma/results.csv")
    dma = dma.loc[dma['ticker'] == selected_ticker, ['sharpe', 'drawdown']].T.rename(index={"sharpe": "Sharpe Ratio", "drawdown": "Drawdown"})
    dma.columns = ["Value"]
    st.dataframe(dma, use_container_width=True)
    st.write("""
    **💡 Metrics Explanation**
    - **Sharpe Ratio**: Measures risk-adjusted return. A higher value indicates better risk-adjusted performance.
    - **Drawdown**: Represents the maximum drop from a peak before recovery, indicating downside risk.
    """)
    
    st.markdown("<h3 style='font-size:20px;'>Our Trading Strategy</h3>", unsafe_allow_html=True)
    st.image("results/trading_bot/performance.png")
    data["Strategy Returns"] = np.where(data["close"].pct_change() > 0, data["close"].pct_change(), 0).cumsum()
    data["Benchmark Returns"] = data["close"].pct_change().cumsum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["date"], y=data["Strategy Returns"], mode="lines", name="AI Strategy", line=dict(color="green")))
    fig.add_trace(go.Scatter(x=data["date"], y=data["Benchmark Returns"], mode="lines", name="Benchmark", line=dict(color="red")))
    # st.plotly_chart(fig, use_container_width=True)

# --- Development Team ---
st.markdown("""
    <style>
        .team-container {
            text-align: center;
            padding: 30px 0px;
        }
        .team-member {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 10px;
        }
        .team-img {
            border-radius: 50%;
            width: 100px; /* Smaller Image */
            height: 100px;
            object-fit: cover;
            display: block;
            margin: 0 auto;
        }
        .team-name {
            font-size: 16px;
            font-weight: bold;
            margin-top: 10px;
        }
        .team-role {
            font-size: 14px;
            color: #bbb;
        }
    </style>
""", unsafe_allow_html=True)

# Centered Title
st.markdown("<h2 style='font-size:24px;'>👨‍💻 Development Team</h2>", unsafe_allow_html=True)

team_data = [
    {"name": "Peeranat Kongkjipipat", "role": "Business & Computer Science", "img": "./photo/Picture1.jpg"},
    {"name": "Kulpatch Chananam", "role": "Computer Science & Economics", "img": "./photo/Picture2.jpg"},
    {"name": "Nathan Juirnarongrit", "role": "Business & Computer Engineering", "img": "./photo/Picture3.jpg"},
    {"name": "Chindanai Trakantannarong", "role": "Computer Science", "img": "./photo/Picture4.jpg"},
]

# Create 4 evenly spaced columns
columns = st.columns(len(team_data))

for col, member in zip(columns, team_data):
    with col:
        # Display Image using Streamlit method
        st.image(member["img"], width=100)
        
        # Centered Name & Role
        st.markdown(f"<p class='team-name'>{member['name']}</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='team-role'>{member['role']}</p>", unsafe_allow_html=True)

