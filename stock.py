import streamlit as st
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import pandas as pd
from datetime import date
from xgboost import XGBRegressor
import numpy as np
import plotly.graph_objects as go
import logging
from xgboost.callback import EarlyStopping
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="stock_forecast.log",
    filemode="a"
)

# Streamlit UI
st.title("📈 Stock Price Forecast App using Prophet | XGBoost")
st.markdown("Enter a stock ticker symbol to forecast its closing prices.")

# User inputs
ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL, MSFT, TSLA)", value="AAPL").upper()
default_start = date.today().replace(year=date.today().year - 10)

range_choice = st.radio(
    "Select Historical Range",
    ("1 Year", "3 Years", "5 Years", "10 Years"),
    index=2,
    horizontal=True
)

years_back = int(range_choice.split()[0])
start_date = date.today().replace(year=date.today().year - years_back)
start_date = st.date_input("Select Start Date", value=start_date, max_value=date.today())
n_days = st.slider("Days to Forecast", 30, 365, 90)
end_date = date.today().strftime("%Y-%m-%d")

# ----------------------------------------
# Event & Market Factor Toggles
# ----------------------------------------
st.subheader("📊 Event and Market Factor Toggles")

col1, col2, col3 = st.columns(3)
with col1:
    use_earnings = st.checkbox("Include Earnings Dates", value=True)
    use_dividends = st.checkbox("Include Dividends", value=True)
    use_splits = st.checkbox("Include Splits", value=True)

with col2:
    use_economic = st.checkbox("Economic Indicators (CPI/Fed Rates)", value=False)
    use_news = st.checkbox("News & Sentiment", value=False)
    use_insider = st.checkbox("Insider Activity (SEC Filings)", value=False)

with col3:
    use_sector = st.checkbox("Sector Trends (ETF Correlation)", value=False)
    use_technical = st.checkbox("Technical Indicators (RSI/MACD)", value=False)
    use_macro = st.checkbox("Macro Events (GDP/Unemployment)", value=False)

# ----------------------------------------
# Data Loading and Event Fetch
# ----------------------------------------
@st.cache_data
def load_data(ticker, start):
    try:
        data = yf.download(ticker, start=start, end=end_date, auto_adjust=True)
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        logging.error(f"Error loading data for {ticker}: {e}")
        st.error(f"Error loading data for {ticker}: {e}")
        return pd.DataFrame()

def get_event_dates(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        earnings = pd.to_datetime(
            ticker_obj.earnings_dates.index
        ) if hasattr(ticker_obj, "earnings_dates") else pd.Series(dtype="datetime64[ns]")
        dividends = pd.to_datetime(
            ticker_obj.dividends.index
        ) if not ticker_obj.dividends.empty else pd.Series(dtype="datetime64[ns]")
        splits = pd.to_datetime(
            ticker_obj.splits.index
        ) if not ticker_obj.splits.empty else pd.Series(dtype="datetime64[ns]")
        return earnings, dividends, splits
    except Exception as e:
        logging.error(f"Event fetch failed: {e}")
        return pd.Series(), pd.Series(), pd.Series()

# ----------------------------------------
# Placeholder data for additional factors
# ----------------------------------------
economic_events = pd.to_datetime([])
sentiment_data = pd.DataFrame({"date": pd.to_datetime([]), "score": []})
insider_events = pd.to_datetime([])
sector_spike_days = pd.to_datetime([])
technical_signal = pd.Series(dtype="float64")
macro_events = pd.to_datetime([])

# ----------------------------------------
# Main Processing
# ----------------------------------------
if ticker:
    data_load_state = st.text("Loading data...")
    data = load_data(ticker, start_date)
    data_load_state.text("")

    if data.empty:
        st.error("❌ No data found. Please check the ticker symbol.")
    elif "Close" not in data.columns:
        st.error("❌ 'Close' column not found in the data.")
    else:

        tab1, tab2 = st.tabs(["📈 Prophet Forecast", "🤖 XGBoost Forecast"])

        with tab1:
            st.subheader(f"Raw data for {ticker}")
            st.write(data.tail())

            # Prepare data for Prophet
            df_train = data[["Date", "Close"]].dropna()
            df_train.columns = ["ds", "y"]
            df_train["y"] = pd.to_numeric(df_train["y"], errors="coerce")

            earnings, dividends, splits = get_event_dates(ticker)

            if not any(
                [
                    use_earnings,
                    use_dividends,
                    use_splits,
                    use_economic,
                    use_news,
                    use_insider,
                    use_sector,
                    use_technical,
                    use_macro,
                ]
            ):
                st.warning("No event types selected. Model will run without external regressors.")

            # Add regressor flags
            if use_earnings:
                df_train["earnings_flag"] = df_train["ds"].isin(earnings).astype(int)
            if use_dividends:
                df_train["dividends_flag"] = df_train["ds"].isin(dividends).astype(int)
            if use_splits:
                df_train["splits_flag"] = df_train["ds"].isin(splits).astype(int)
            if use_economic:
                df_train["economic_flag"] = df_train["ds"].isin(economic_events).astype(int)
            if use_news:
                df_train["sentiment_score"] = (
                    df_train["ds"].map(sentiment_data.set_index("date")["score"]).fillna(0)
                )
            if use_insider:
                df_train["insider_flag"] = df_train["ds"].isin(insider_events).astype(int)
            if use_sector:
                df_train["sector_flag"] = df_train["ds"].isin(sector_spike_days).astype(int)
            if use_technical:
                df_train["technical_signal"] = technical_signal.reindex(df_train.index, fill_value=0)
            if use_macro:
                df_train["macro_flag"] = df_train["ds"].isin(macro_events).astype(int)

            logging.info(f"Training data prepared with {len(df_train)} rows for ticker {ticker}")

            if df_train.empty:
                st.error("❌ No valid 'Close' price data to train the model.")
            else:
                try:
                    # Prophet model setup
                    model = Prophet()

                    # Dynamically add regressors
                    if use_earnings: model.add_regressor("earnings_flag")
                    if use_dividends: model.add_regressor("dividends_flag")
                    if use_splits: model.add_regressor("splits_flag")
                    if use_economic: model.add_regressor("economic_flag")
                    if use_news: model.add_regressor("sentiment_score")
                    if use_insider: model.add_regressor("insider_flag")
                    if use_sector: model.add_regressor("sector_flag")
                    if use_technical: model.add_regressor("technical_signal")
                    if use_macro: model.add_regressor("macro_flag")

                    model.fit(df_train)

                    # Future dataframe
                    future = model.make_future_dataframe(periods=n_days)
                    if use_earnings: future["earnings_flag"] = future["ds"].isin(earnings).astype(int)
                    if use_dividends: future["dividends_flag"] = future["ds"].isin(dividends).astype(int)
                    if use_splits: future["splits_flag"] = future["ds"].isin(splits).astype(int)
                    if use_economic: future["economic_flag"] = future["ds"].isin(economic_events).astype(int)
                    if use_news: future["sentiment_score"] = (
                        future["ds"].map(sentiment_data.set_index("date")["score"]).fillna(0)
                    )
                    if use_insider: future["insider_flag"] = future["ds"].isin(insider_events).astype(int)
                    if use_sector: future["sector_flag"] = future["ds"].isin(sector_spike_days).astype(int)
                    if use_technical: future["technical_signal"] = 0  # placeholder
                    if use_macro: future["macro_flag"] = future["ds"].isin(macro_events).astype(int)

                    forecast = model.predict(future)

                    # Forecast plot
                    st.subheader("📊 Forecast Plot")
                    fig1 = plot_plotly(model, forecast)
                    st.plotly_chart(fig1)

                    # Forecast components
                    st.subheader("📉 Forecast Components")
                    fig2 = model.plot_components(forecast)
                    st.write(fig2)

                    logging.info(f"Forecast generated for {n_days} days for {ticker}")

                except Exception as e:
                    logging.error(f"Error during forecasting: {e}")
                    st.error(f"⚠️ Error during forecasting: {e}")

        with tab2:

            st.subheader("🤖 XGBoost Stock Price Prediction")

            # -------------------------------------------------------
            # TOGGLES
            # -------------------------------------------------------
            run_wfv = st.checkbox("Run Walk-Forward Validation (slow)", value=False)

            # -------------------------------------------------------
            # FIX MULTIINDEX
            # -------------------------------------------------------
            data.columns = data.columns.to_flat_index()
            data.columns = [
                str(c).replace("(", "").replace(")", "").replace(",", "").replace("'", "").strip()
                for c in data.columns
            ]

            # -------------------------------------------------------
            # DETECT COLUMNS
            # -------------------------------------------------------
            close_col = next((c for c in data.columns if "Close" in c), None)

            if "Date" not in data.columns or close_col is None:
                st.error(f"Missing Date or Close column. Found: {data.columns}")
                st.stop()

            df = data[["Date", close_col]].dropna().copy()
            df.rename(columns={close_col: "Close"}, inplace=True)
            df["Date"] = pd.to_datetime(df["Date"])

            # -------------------------------------------------------
            # PARAMETERS
            # -------------------------------------------------------
            lag_window = max(5, n_days // 2)

            # -------------------------------------------------------
            # FEATURE ENGINEERING
            # -------------------------------------------------------
            feature_dict = {}

            for lag in range(1, lag_window + 1):
                feature_dict[f"lag_{lag}"] = df["Close"].shift(lag)

            feature_dict["day_of_week"] = df["Date"].dt.weekday
            feature_dict["day_of_month"] = df["Date"].dt.day
            feature_dict["month"] = df["Date"].dt.month
            feature_dict["is_month_start"] = df["Date"].dt.is_month_start.astype(int)
            feature_dict["is_month_end"] = df["Date"].dt.is_month_end.astype(int)

            returns = np.log(df["Close"] / df["Close"].shift(1))
            feature_dict["returns_1d"] = returns
            feature_dict["returns_3d"] = returns.rolling(3).mean()
            feature_dict["returns_7d"] = returns.rolling(7).mean()
            feature_dict["volatility_5d"] = returns.rolling(5).std()
            feature_dict["volatility_10d"] = returns.rolling(10).std()

            df["target"] = np.log(df["Close"].shift(-1) / df["Close"])

            df = pd.concat([df, pd.DataFrame(feature_dict)], axis=1).dropna().copy()

            feature_cols = (
                [f"lag_{i}" for i in range(1, lag_window + 1)] +
                [
                    "day_of_week", "day_of_month", "month",
                    "is_month_start", "is_month_end",
                    "returns_1d", "returns_3d", "returns_7d",
                    "volatility_5d", "volatility_10d"
                ]
            )

            # -------------------------------------------------------
            # WALK-FORWARD VALIDATION
            # -------------------------------------------------------
            if run_wfv:
                st.subheader("📊 Walk-Forward Validation")

                start_train_size = int(len(df) * 0.6)

                wf_preds, wf_actuals, wf_dates = [], [], []

                for i in range(start_train_size, len(df)):
                    train_df = df.iloc[:i]
                    test_df = df.iloc[i:i+1]

                    model = XGBRegressor(
                        n_estimators=300,
                        learning_rate=0.03,
                        max_depth=6,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_alpha=2,
                        reg_lambda=3,
                        objective="reg:squarederror",
                        tree_method="hist",
                        random_state=42
                    )

                    model.fit(train_df[feature_cols], train_df["target"])
                    pred = model.predict(test_df[feature_cols])[0]

                    wf_preds.append(pred)
                    wf_actuals.append(test_df["target"].values[0])
                    wf_dates.append(test_df["Date"].values[0])

                wfv_df = pd.DataFrame({
                    "Date": wf_dates,
                    "Actual_Return": wf_actuals,
                    "Predicted_Return": wf_preds
                })

                st.metric(
                    "Walk-Forward MAE",
                    f"{mean_absolute_error(wfv_df['Actual_Return'], wfv_df['Predicted_Return']):.6f}"
                )
                st.metric(
                    "Walk-Forward RMSE",
                    f"{root_mean_squared_error(wfv_df['Actual_Return'], wfv_df['Predicted_Return'], squared=False):.6f}"
                )

                fig_wf = go.Figure()
                fig_wf.add_trace(go.Scatter(
                    x=wfv_df["Date"], y=wfv_df["Actual_Return"], mode="lines", name="Actual"
                ))
                fig_wf.add_trace(go.Scatter(
                    x=wfv_df["Date"], y=wfv_df["Predicted_Return"], mode="lines", name="Predicted"
                ))
                st.plotly_chart(fig_wf, use_container_width=True)

            # -------------------------------------------------------
            # FINAL TRAINING
            # -------------------------------------------------------
            model_xgb = XGBRegressor(
                n_estimators=500,
                learning_rate=0.03,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=2,
                reg_lambda=3,
                objective="reg:squarederror",
                tree_method="hist",
                random_state=42
            )

            model_xgb.fit(df[feature_cols], df["target"])

            # -------------------------------------------------------
            # MULTI-DAY FORECAST
            # -------------------------------------------------------
            last_prices = df["Close"].iloc[-lag_window:].tolist()
            last_price = last_prices[-1]
            future_preds = []

            future_dates = pd.bdate_range(
                start=df["Date"].iloc[-1] + pd.Timedelta(days=1),
                periods=n_days
            )

            for future_date in future_dates:
                returns_hist = np.diff(np.log(last_prices))

                row = {
                    **{f"lag_{i+1}": last_prices[-(i+1)] for i in range(lag_window)},
                    "day_of_week": future_date.weekday(),
                    "day_of_month": future_date.day,
                    "month": future_date.month,
                    "is_month_start": int(future_date.is_month_start),
                    "is_month_end": int(future_date.is_month_end),
                    "returns_1d": returns_hist[-1],
                    "returns_3d": returns_hist[-3:].mean(),
                    "returns_7d": returns_hist[-7:].mean() if len(returns_hist) >= 7 else returns_hist.mean(),
                    "volatility_5d": np.std(returns_hist[-5:]),
                    "volatility_10d": np.std(returns_hist[-10:])
                }

                pred_return = model_xgb.predict(pd.DataFrame([row])[feature_cols])[0]
                pred_return = np.clip(pred_return, -0.03, 0.03)

                next_price = last_price * np.exp(pred_return)

                future_preds.append(next_price)
                last_prices.append(next_price)
                last_price = next_price

            forecast_df = pd.DataFrame({
                "Date": future_dates,
                "Prediction": future_preds
            })

            # -------------------------------------------------------
            # PLOT
            # -------------------------------------------------------
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["Date"], y=df["Close"], mode="lines", name="Historical Close"
            ))
            fig.add_trace(go.Scatter(
                x=forecast_df["Date"], y=forecast_df["Prediction"],
                mode="lines+markers", name="XGBoost Forecast"
            ))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(forecast_df)




# ----------------------------------------
# News Section
# ----------------------------------------
ticker_obj = yf.Ticker(ticker)
st.markdown("---")
try:
    news_list = ticker_obj.news[:5] if hasattr(ticker_obj, "news") else []
except Exception:
    news_list = []
    st.warning("News data not available.")

for article in news_list:
    content = article.get("content", {})
    title = content.get("title", "Untitled")
    link_info = content.get("clickThroughUrl", {})
    link = link_info.get("url", "#") if isinstance(link_info, dict) else "#"
    st.markdown(f"📰 [{title}]({link})")

# ----------------------------------------
# Disclaimer
# ----------------------------------------
st.markdown("---")
st.markdown("📌 **Disclaimer:**")
st.info(
    "This stock forecast is generated using the open-source [Facebook Prophet](https://facebook.github.io/prophet/) model. "
    "It is based on historical data and does not account for real-time market factors, news, or external economic conditions. "
    "This tool is for educational and informational purposes only and should not be considered financial advice. "
    "Always do your own research or consult a financial advisor before making investment decisions."
)
