import streamlit as st
import yfinance as yf
from neuralprophet import NeuralProphet
import pandas as pd
import plotly.graph_objects as go
from datetime import date
import logging

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="meta_prophet_forecast.log",
    filemode="a"
)

# Streamlit UI
st.title("📈 Stock Forecast App with Meta Prophet (NeuralProphet)")
st.markdown("Enter a stock ticker to forecast closing prices using a deep learning model.")

ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL, MSFT, TSLA)", value="AAPL").upper()
n_days = st.slider("Days to Forecast", 30, 365, 90)

start_date = "2015-01-01"
end_date = date.today().strftime("%Y-%m-%d")

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    data.reset_index(inplace=True)
    return data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})

if ticker:
    try:
        st.info(f"Fetching data for {ticker}...")
        df = load_data(ticker)
        st.write("✅ Raw Data", df.tail())

        # NeuralProphet model
        m = NeuralProphet(
            n_forecasts=1,
            n_lags=14,
            yearly_seasonality=True,
            weekly_seasonality=True,
            learning_rate=1.0,
        )

        metrics = m.fit(df, freq="D", epochs=100)
        future = m.make_future_dataframe(df, periods=n_days, n_historic_predictions=True)
        forecast = m.predict(future)

        # Plot
        st.subheader("📊 Forecast Plot")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat1"], name="Forecast"))
        fig.add_trace(go.Scatter(x=df["ds"], y=df["y"], name="Actual"))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📉 Forecast Table")
        st.dataframe(forecast.tail(n_days)[["ds", "yhat1"]].rename(columns={"ds": "Date", "yhat1": "Predicted Close"}))

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
        logging.error(f"App failed: {e}")
