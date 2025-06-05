import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import os

# --- Mean Absolute Percentage Error ---
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# --- Load Dataset ---
@st.cache_data
def load_data():
    return pd.read_csv("dfmonthly_modelling.csv", parse_dates=['Date'])

st.title("📈 Prophet Forecast with Extra Regressors")

# Load data
df = load_data()

# User selects target column
target_col = st.selectbox("Select the Target Column", [col for col in df.columns if col != 'Date'])

# Prediction horizon
months = st.slider("Months to Predict", min_value=0, max_value=24, value=12)

# Rename for Prophet format
df_prophet = df.rename(columns={'Date': 'ds', target_col: 'y'})
regressor_cols = [col for col in df.columns if col not in ['Date', target_col]]

# Initialize Prophet and add regressors
model = Prophet()
for col in regressor_cols:
    model.add_regressor(col)

# Fit model
model.fit(df_prophet)

# Create future dataframe
future = model.make_future_dataframe(periods=months, freq='MS')

# Add regressor values to future
for col in regressor_cols:
    future[col] = None
    future.loc[future['ds'].isin(df_prophet['ds']), col] = df_prophet.set_index('ds').loc[
        future.loc[future['ds'].isin(df_prophet['ds']), 'ds'], col].values
    future[col].fillna(df_prophet[col].iloc[-1], inplace=True)

# Predict
forecast = model.predict(future)

# Plot Forecast
st.subheader("Forecast Plot")
fig1 = model.plot(forecast)
st.pyplot(fig1)

# Plot Components
st.subheader("Forecast Components")
fig2 = model.plot_components(forecast)
st.pyplot(fig2)

# Evaluation on training data
forecast_train = model.predict(df_prophet)
y_true = df_prophet['y'].values
y_pred = forecast_train['yhat'].values

rmse = sqrt(mean_squared_error(y_true, y_pred))
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

# Display metrics
st.subheader("Model Evaluation Metrics (on Training Data)")
st.markdown(f"- RMSE: `{rmse:.4f}`")
st.markdown(f"- MSE: `{mse:.4f}`")
st.markdown(f"- MAE: `{mae:.4f}`")
st.markdown(f"- MAPE: `{mape:.2f}%`")
st.markdown(f"- R² Score: `{r2:.4f}`")

# Show forecast table
st.subheader("Forecasted Values (last few months)")
st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(months))
