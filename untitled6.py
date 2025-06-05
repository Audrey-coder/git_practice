import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

# -------------------------
# MAPE Function
# -------------------------
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# -------------------------
# Load Data
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("dfmonthly_modeling.csv", parse_dates=['Date'])
    df = df.rename(columns={'Date': 'ds', 'exchange_rate': 'y'})
    return df

st.set_page_config(page_title="Exchange Rate Forecast", layout="centered")
st.title("📈 Prophet Forecast with Regressors")

# Load and show data
df_prophet = load_data()
st.subheader("🔍 Data Preview")
st.dataframe(df_prophet.head())

# -------------------------
# Prophet Model
# -------------------------
regressor_cols = [col for col in df_prophet.columns if col not in ['ds', 'y']]
model = Prophet()
for col in regressor_cols:
    model.add_regressor(col)

with st.spinner("Training Prophet model..."):
    model.fit(df_prophet)

# -------------------------
# Forecasting
# -------------------------
num_future_months = 12
future = model.make_future_dataframe(periods=num_future_months, freq='MS')

# Fill in future regressor values
for col in regressor_cols:
    future[col] = None
    past_mask = future['ds'].isin(df_prophet['ds'])
    future.loc[past_mask, col] = df_prophet.set_index('ds').loc[future.loc[past_mask, 'ds'], col].values
    future[col].fillna(df_prophet[col].iloc[-1], inplace=True)

# Make predictions
forecast = model.predict(future)

st.subheader("📊 Forecast Table")
st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(13))

# -------------------------
# Plot Forecast
# -------------------------
st.subheader("📉 Forecast Plot")
fig1 = model.plot(forecast)
st.pyplot(fig1)

# -------------------------
# Plot Components
# -------------------------
st.subheader("🧩 Forecast Components")
fig2 = model.plot_components(forecast)
st.pyplot(fig2)

# -------------------------
# Evaluation Metrics
# -------------------------
st.subheader("📋 Evaluation Metrics")

# Evaluate on training data
forecast_train = model.predict(df_prophet)
y_true = df_prophet['y'].values
y_pred = forecast_train['yhat'].values

rmse = sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

st.markdown(f"""
**Training Performance:**
- RMSE: `{rmse:.4f}`
- MAE: `{mae:.4f}`
- MSE: `{mse:.4f}`
- MAPE: `{mape:.2f}%`
- R² Score: `{r2:.4f}`
""")
