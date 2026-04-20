# 📈 Streamlit Stock Forecast App (Prophet + XGBoost)

An interactive **Stock Price Forecasting Web App** built using **Streamlit**, combining the power of:

    * 🔮 Prophet (time-series forecasting)
    * 🤖 XGBoost (machine learning regression)
    * 📊 Plotly (interactive visualizations)
    * 📡 Yahoo Finance (real-time data via yfinance)

---

## 🚀 Features

### 📊 Dual Forecasting Models

* **Prophet Model**

  * Time-series forecasting
  * Supports external regressors (events & market factors)
  * Trend & seasonality decomposition

* **XGBoost Model**

  * Machine learning-based prediction
  * Feature engineering (lags, volatility, returns)
  * Multi-day recursive forecasting
  * Optional Walk-Forward Validation

---

### ⚙️ Smart Inputs

* Select stock ticker (e.g., AAPL, MSFT, TSLA)
* Choose historical range (1, 3, 5, 10 years)
* Custom start date
* Forecast horizon (30–365 days)

---

### 📊 Event & Market Factor Toggles

Enhance forecasting using optional signals:

* 📅 Earnings Dates
* 💰 Dividends
* 🔀 Stock Splits
* 📉 Economic Indicators (placeholder)
* 📰 News & Sentiment (placeholder)
* 🧑‍💼 Insider Activity (placeholder)
* 🏭 Sector Trends (placeholder)
* 📈 Technical Indicators (placeholder)
* 🌎 Macro Events (placeholder)

---

### 📉 Visualizations

* Interactive forecast charts (Plotly)
* Prophet components (trend, seasonality)
* XGBoost prediction curves
* Walk-forward validation metrics (MAE, RMSE)

---

### 📰 Latest News Integration

* Fetches recent news articles for selected ticker

---

## 🛠️ Tech Stack

* Python
* Streamlit
* Prophet
* XGBoost
* Pandas / NumPy
* Plotly
* yFinance
* Scikit-learn

---

## 📂 Project Structure

```
.
├── app.py
├── requirements.txt
├── stock_forecast.log
└── README.md
```

---

## ⚡ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/streamlit-stock-forecast.git
cd streamlit-stock-forecast
```

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the App

```bash
streamlit run app.py
```

---

## 📦 Requirements

Example `requirements.txt`:

```
streamlit
yfinance
prophet
pandas
numpy
plotly
xgboost
scikit-learn
```

---

## 🧠 How It Works

### 🔮 Prophet Workflow

1. Load historical stock data
2. Add optional regressors (events)
3. Train Prophet model
4. Generate future predictions
5. Visualize forecast & components

---

### 🤖 XGBoost Workflow

1. Feature engineering:

   * Lag features
   * Returns
   * Volatility
   * Calendar features
2. Train regression model
3. Predict daily returns
4. Convert to price forecasts
5. Recursive multi-day prediction

---

## 📊 Evaluation Metrics

* **MAE (Mean Absolute Error)**
* **RMSE (Root Mean Squared Error)**

Used in Walk-Forward Validation mode.

---

## 📌 Logging

* Logs saved to:

```
stock_forecast.log
```

* Tracks:

  * Data loading issues
  * Model training
  * Forecast generation

---

## ⚠️ Disclaimer

This application uses statistical and machine learning models for forecasting stock prices based on historical data.

> ❗ This is **NOT financial advice**.
> Predictions do not account for real-time events, news, or market sentiment.
> Always consult a financial advisor before making investment decisions.

---

## 🌟 Future Enhancements

* Real-time sentiment analysis (news APIs)
* Technical indicators (RSI, MACD)
* Macro-economic data integration
* Model comparison dashboard
* Deployment on Streamlit Cloud / AWS

---

## 🙌 Contributing

Contributions are welcome!

1. Fork the repo
2. Create a feature branch
3. Commit changes
4. Open a Pull Request

---

## 📬 Contact

For questions or suggestions, feel free to reach out.

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!

---
