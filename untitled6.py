import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import numpy as np
import plotly.graph_objs as go

# --- Custom MAPE function ---
def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculates the Mean Absolute Percentage Error (MAPE).
    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.
    Returns:
        float: The MAPE value.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero for y_true values that are 0
    return np.mean(np.abs((y_true - y_pred) / y_true[y_true != 0])) * 100 if np.any(y_true != 0) else 0.0

# --- Streamlit App ---
st.set_page_config(layout="wide", page_title="Exchange Rate Prophet Forecast")

st.title("📈 Exchange Rate Prophet Forecast with Macroeconomic Indicators")

st.markdown("""
This application allows you to forecast a target macroeconomic variable (e.g., Exchange Rate)
using the Prophet model, incorporating other variables as regressors.
""")

# --- Data Loading ---
st.header("1. Load Your Data")
uploaded_file = st.file_uploader("Upload your 'dfmonthly_modelling.csv' file", type="csv")

df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("Data loaded successfully!")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error loading file: {e}. Please ensure it's a valid CSV.")
else:
    st.info("No file uploaded. Using a sample dataset for demonstration.")
    # Create a sample dataframe if no file is uploaded for demonstration
    # In a real deployment, you'd likely load from a fixed path or prompt the user.
    try:
        df = pd.read_csv('dfmonthly_modelling.csv') # Assumes this file exists in the deployment environment
    except FileNotFoundError:
        st.warning("dfmonthly_modelling.csv not found locally. Creating a dummy dataset.")
        # Create a dummy dataframe for local testing if the actual file isn't present
        dates = pd.date_range(start='2005-01-01', periods=228, freq='MS')
        data = {
            'Date': dates,
            'exchange_rate': np.linspace(70, 150, 228) + np.random.normal(0, 5, 228),
            '12-month_inflation%': np.linspace(5, 10, 228) + np.random.normal(0, 1, 228),
            'central_bank_rate': np.linspace(6, 12, 228) + np.random.normal(0, 0.5, 228),
            'total_remittances': np.linspace(50000, 300000, 228) + np.random.normal(0, 10000, 228),
            'imports': np.linspace(1e10, 2e10, 228) + np.random.normal(0, 1e9, 228),
            'exports': np.linspace(5e9, 1.2e10, 228) + np.random.normal(0, 5e8, 228),
            'total_debt': np.linspace(1e6, 8e6, 228) + np.random.normal(0, 5e5, 228),
            'deposit': np.linspace(5, 10, 228) + np.random.normal(0, 0.3, 228),
            'savings': np.linspace(3, 8, 228) + np.random.normal(0, 0.2, 228),
            'lending': np.linspace(10, 15, 228) + np.random.normal(0, 0.8, 228),
            'overdraft': np.linspace(12, 18, 228) + np.random.normal(0, 1.0, 228),
        }
        df = pd.DataFrame(data)
        st.dataframe(df.head())
        st.warning("Please upload your actual CSV for real results.")


if df is not None:
    # --- Data Preprocessing for Prophet ---
    # Convert 'Date' column to datetime and set as index
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    df = df.sort_index()

    # Drop any non-numeric or irrelevant columns for the model besides 'Date' (which is now index)
    # Ensure all remaining columns are numeric, fill NaNs with median
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in numerical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    
    st.header("2. Configure Prophet Model")

    # Sidebar for user inputs
    with st.sidebar:
        st.header("Model Configuration")
        
        # Select target column
        available_cols = df.columns.tolist()
        target_column = st.selectbox(
            "Select Target Variable (y):",
            options=available_cols,
            index=available_cols.index('exchange_rate') if 'exchange_rate' in available_cols else 0
        )

        # Slider for prediction months
        num_future_months = st.slider(
            "Number of Months to Forecast:",
            min_value=0,
            max_value=24,
            value=12,
            step=1
        )

    # Prepare DataFrame for Prophet
    df_prophet = df.reset_index().rename(columns={'Date': 'ds', target_column: 'y'})

    # Identify regressor columns (all other numerical columns)
    regressor_cols = [col for col in df_prophet.columns if col not in ['ds', 'y']]

    st.subheader(f"Target Variable: **{target_column}**")
    st.subheader(f"Regressors Used: **{', '.join(regressor_cols) if regressor_cols else 'None'}**")

    # --- Prophet Model Initialization and Fitting ---
    if st.button("Run Prophet Forecast"):
        if len(df_prophet) < 2:
            st.error("Not enough data points to train the Prophet model. Need at least 2.")
        else:
            with st.spinner("Training Prophet model and generating forecast..."):
                model = Prophet()

                # Add each regressor to the model
                for col in regressor_cols:
                    model.add_regressor(col)

                try:
                    # Fit the model
                    model.fit(df_prophet)

                    # Create future dataframe for prediction
                    future = model.make_future_dataframe(periods=num_future_months, freq='MS')

                    # We need to add the regressor values for the future dates.
                    # For demonstration, we'll use the last known value (most recent) for future regressors.
                    # In a real-world application, these future regressor values would ideally be forecasted or provided.
                    for col in regressor_cols:
                        # For existing dates, use actual values
                        future.loc[future['ds'].isin(df_prophet['ds']), col] = df_prophet.set_index('ds').loc[future.loc[future['ds'].isin(df_prophet['ds']), 'ds'], col].values
                        # For future dates, fill with the last known value
                        future[col].fillna(df_prophet[col].iloc[-1], inplace=True)

                    # Make predictions
                    forecast = model.predict(future)

                    st.success("Forecast generated!")

                    # --- Display Forecast Plot ---
                    st.header("3. Forecast Plot")
                    fig1 = model.plot(forecast)
                    # Customizing plot to make it more interactive/readable with Plotly
                    plotly_fig1 = plot_plotly(model, forecast)
                    plotly_fig1.update_layout(
                        title=f"Prophet Forecast of {target_column} with Regressors",
                        xaxis_title="Date",
                        yaxis_title=target_column,
                        hovermode="x unified"
                    )
                    st.plotly_chart(plotly_fig1, use_container_width=True)

                    # --- Display Components Plot ---
                    st.header("4. Forecast Components")
                    plotly_fig2 = model.plot_components(forecast)
                    plotly_fig2.update_layout(
                        title=f"Components of {target_column} Forecast",
                        hovermode="x unified"
                    )
                    st.plotly_chart(plotly_fig2, use_container_width=True)

                    # --- Display Evaluation Metrics ---
                    st.header("5. Model Evaluation")
                    st.subheader("Performance Metrics on Training Data")
                    
                    forecast_train = model.predict(df_prophet)
                    y_true = df_prophet['y'].values
                    y_pred = forecast_train['yhat'].values

                    rmse = sqrt(mean_squared_error(y_true, y_pred))
                    mse = mean_squared_error(y_true, y_pred)
                    mae = mean_absolute_error(y_true, y_pred)
                    mape = mean_absolute_percentage_error(y_true, y_pred)
                    r2 = r2_score(y_true, y_pred)

                    metrics_df = pd.DataFrame({
                        "Metric": ["RMSE", "MSE", "MAE", "MAPE", "R-squared"],
                        "Value": [f"{rmse:.4f}", f"{mse:.4f}", f"{mae:.4f}", f"{mape:.2f}%", f"{r2:.4f}"]
                    })
                    st.table(metrics_df)

                    st.markdown("""
                    * **RMSE (Root Mean Squared Error):** Measures the average magnitude of the errors. Lower is better.
                    * **MSE (Mean Squared Error):** Measures the average of the squares of the errors. Lower is better.
                    * **MAE (Mean Absolute Error):** Measures the average absolute difference between predictions and actual values. Lower is better.
                    * **MAPE (Mean Absolute Percentage Error):** The average percentage error. Lower is better.
                    * **R-squared (R²):** Represents the proportion of the variance in the dependent variable that is predictable from the independent variables. Closer to 1 is better.
                    """)

                    st.subheader("Future Forecast Data")
                    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(num_future_months + 1))


                except Exception as e:
                    st.error(f"An error occurred during model fitting or prediction: {e}")
                    st.info("Please check your data and ensure it's suitable for Prophet (e.g., sufficient historical data, no non-numeric values in regressor columns after preprocessing).")

else:
    st.warning("Please upload a CSV file to proceed with the forecasting.")
