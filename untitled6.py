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
import streamlit as st
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

@st.cache_data
def load_data():
    return pd.read_csv("dfmonthly_modelling.csv", parse_dates=['Date'])

df = load_data()
df_prophet = df.reset_index().rename(columns={'Date': 'ds', 'exchange_rate': 'y'})

regressor_cols = [col for col in df_prophet.columns if col not in ['ds', 'y']]

model = Prophet()
for col in regressor_cols:
    model.add_regressor(col)
model.fit(df_prophet)

num_future_months = st.slider(
    "Select number of months to forecast:",
    min_value=0,
    max_value=24,
    value=12,
    step=1,
)

future = model.make_future_dataframe(periods=num_future_months, freq='MS')

for col in regressor_cols:
    future[col] = None
    mask = future['ds'].isin(df_prophet['ds'])
    future.loc[mask, col] = df_prophet.set_index('ds').loc[future.loc[mask, 'ds'], col].values
    future[col].fillna(df_prophet[col].iloc[-1], inplace=True)

forecast = model.predict(future)

st.subheader(f"Forecast for next {num_future_months} months")
st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(num_future_months))

fig1 = model.plot(forecast)
st.pyplot(fig1)

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
