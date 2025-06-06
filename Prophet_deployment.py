import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Title
st.title("📈 Prophet Forecast with Regressors (Exchange Rate)")

# Load data from repo
@st.cache_data
def load_data():
    return pd.read_csv("dfmonthly_modelling.csv")

# Load dataset
df = load_data()

st.subheader("Preview of Exchange Rate Data")
st.write(df.head())

# Select datetime and target column
date_col = st.selectbox("Select Date Column", df.columns)
y_col = st.selectbox("Select Exchange Rate Column", [col for col in df.columns if col != date_col])

# Rename for Prophet
df_prophet = df.rename(columns={date_col: 'ds', y_col: 'y'})

# Identify regressors
regressor_cols = [col for col in df_prophet.columns if col not in ['ds', 'y']]

# Initialize Prophet model and add regressors
model = Prophet()
for col in regressor_cols:
    model.add_regressor(col)

# Fit model
model.fit(df_prophet)

# Forecast period
num_months = st.slider("Months to Predict", 1, 24, 12)

# Create future dataframe
future = model.make_future_dataframe(periods=num_months, freq='MS')
df_prophet = df_prophet.sort_values('ds')
future = future.sort_values('ds')

# Copy past regressor values and fill future with last known
for col in regressor_cols:
    past_mask = future['ds'].isin(df_prophet['ds'])
    future.loc[past_mask, col] = df_prophet.set_index('ds')[col].reindex(future.loc[past_mask, 'ds']).values
    future[col].fillna(df_prophet[col].iloc[-1], inplace=True)

# Forecast
forecast = model.predict(future)

# Output table
st.subheader("📑 Forecasted Exchange Rate")
st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(num_months))

# Forecast Plot
st.subheader("📊 Forecast Plot")
fig1 = model.plot(forecast)
st.pyplot(fig1)

# Forecast Components
st.subheader("🔍 Forecast Components")
fig2 = model.plot_components(forecast)
st.pyplot(fig2)

