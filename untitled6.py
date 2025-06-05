import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

# -------------------------
# Load Data
# -------------------------
# Load your data (make sure path is correct)
@st.cache_data
def load_data():
    # Adjust path if needed
    return pd.read_csv("dfmonthly_modeling.csv", parse_dates=True)

df = load_data()

st.title("📈 Prophet Forecast with All Regressors")

# Select date column (usually a datetime)
date_col = st.selectbox("Select the Date column", options=df.columns)

# Select target column (exclude date_col)
possible_targets = [col for col in df.columns if col != date_col]
target_col = st.selectbox("Select the target column to forecast", options=possible_targets)

# Prepare dataframe for Prophet
df_prophet = df.rename(columns={date_col: "ds", target_col: "y"})

# Identify regressors (all except ds and y)
regressors = [col for col in df_prophet.columns if col not in ["ds", "y"]]

# Inform user of regressors used
st.write(f"Using regressors: {', '.join(regressors)}")

# Initialize Prophet model and add regressors
model = Prophet()
for reg in regressors:
    model.add_regressor(reg)

# Fit model
model.fit(df_prophet)

# Slider for forecast months (0 to 24)
num_months = st.slider(
    "Months to predict into the future",
    min_value=0,
    max_value=24,
    value=12,
    step=1,
)

# Create future dataframe
future = model.make_future_dataframe(periods=num_months, freq='MS')

# Add regressor values for the future dataframe
for reg in regressors:
    future[reg] = None
    mask = future['ds'].isin(df_prophet['ds'])
    future.loc[mask, reg] = df_prophet.set_index('ds').loc[future.loc[mask, 'ds'], reg].values
    future[reg].fillna(df_prophet[reg].iloc[-1], inplace=True)

# Predict
forecast = model.predict(future)

# Show forecast table for future months only
if num_months > 0:
    st.subheader(f"Forecast for next {num_months} months")
    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(num_months))
else:
    st.info("Select months > 0 to see forecast.")

# Plot forecast
fig1 = model.plot(forecast)
st.pyplot(fig1)

# Plot components
fig2 = model.plot_components(forecast)
st.pyplot(fig2)
# -------------------------
# MAPE Function
# -------------------------
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

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
