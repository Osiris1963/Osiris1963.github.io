import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
from prophet import Prophet
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import numpy as np
import warnings

# Suppress serialization warnings from Prophet
warnings.filterwarnings("ignore", message="A value is trying to be set on a copy of a slice from a DataFrame")

# --- Firebase Initialization ---
# IMPORTANT: Place your downloaded service account key file in the same directory
# and name it 'serviceAccountKey.json'
try:
    if not firebase_admin._apps:
        firebase_admin.initialize_app()
    db = firestore.client()
    print("Successfully connected to Firebase within Cloud Function.")
except Exception as e:
    print(f"Error connecting to Firebase: {e}")
    raise e
    db = firestore.client()
    print("Successfully connected to Firebase.")
except Exception as e:
    print(f"Error connecting to Firebase: {e}")
    print("Please ensure 'serviceAccountKey.json' is in the correct directory and is valid.")
    exit()

# --- Data Fetching ---
def fetch_firestore_collection(collection_name):
    """Fetches all documents from a Firestore collection and returns a list of dicts."""
    docs = db.collection(collection_name).stream()
    data = [doc.to_dict() for doc in docs]
    print(f"Fetched {len(data)} documents from '{collection_name}'.")
    return data

# --- Feature Engineering ---
def create_features(df, holidays_df):
    """Create time series features from a datetime index."""
    df['dayofweek'] = df['ds'].dt.dayofweek
    df['quarter'] = df['ds'].dt.quarter
    df['month'] = df['ds'].dt.month
    df['year'] = df['ds'].dt.year
    df['dayofyear'] = df['ds'].dt.dayofyear
    df['weekofyear'] = df['ds'].dt.isocalendar().week.astype(int)
    
    # --- Advanced Feature Engineering ---
    # 1. Payday features (15th and end of month)
    df['is_payday'] = df['ds'].dt.is_month_end | (df['ds'].dt.day == 15)
    df['days_after_payday'] = df.groupby(df['ds'].dt.to_period('M')).cumcount()
    df.loc[df['is_payday'], 'days_after_payday'] = 0
    
    # 2. Holiday proximity features
    holidays_df = holidays_df.sort_values('ds')
    df = pd.merge_asof(df.sort_values('ds'), holidays_df[['ds']].rename(columns={'ds':'holiday_date'}), 
                       left_on='ds', right_on='holiday_date', direction='forward')
    df['days_until_holiday'] = (df['holiday_date'] - df['ds']).dt.days.fillna(365) # Fill NaNs for dates past last holiday
    df = df.drop(columns=['holiday_date'])

    # 3. Lag and Rolling Window Features
    df = df.sort_values('ds')
    df['customers_lag_7'] = df['y'].shift(7)
    df['rolling_mean_7'] = df['y'].shift(1).rolling(window=7).mean()
    df['rolling_std_7'] = df['y'].shift(1).rolling(window=7).std()

    # Fill NaNs created by shift/rolling operations
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)

    return df

# --- Weather API Placeholder ---
def get_weather_forecast(days=15):
    """
    Placeholder for a real weather API call.
    In a real-world scenario, you would call a service like OpenWeatherMap here.
    This function simulates a forecast, returning mostly sunny days.
    """
    print("Fetching 15-day weather forecast (using placeholder)...")
    # This is a placeholder. Replace with a real API call.
    # For now, we assume a default weather condition (e.g., Cloudy/Sunny)
    # and return 0 for 'Rainy' and 'Storm'.
    return {
        'weather_Rainy': np.zeros(days),
        'weather_Storm': np.zeros(days)
    }

# --- Main Forecasting Logic ---
def run_daily_forecast(event=None, context=None):
    """
    Main function to run the entire forecasting process:
    1. Fetches data from Firestore.
    2. Processes and engineers features.
    3. Trains a Prophet model for baseline forecast.
    4. Trains an XGBoost model on Prophet's residuals.
    5. Generates a 15-day future forecast.
    6. Saves the forecast back to Firestore.
    """
    print("\nStarting the forecasting process...")

    # 1. Fetch and Prepare Data
    historical_data = fetch_firestore_collection('historical_data')
    future_activities_data = fetch_firestore_collection('future_activities')
    
    if not historical_data:
        print("No historical data found. Cannot proceed with forecasting.")
        return

    df = pd.DataFrame(historical_data)
    df['ds'] = pd.to_datetime(df['date'])
    df = df.rename(columns={'customers': 'y'})
    df = df[['ds', 'y', 'sales', 'add_on_sales', 'weather']]
    df = df.sort_values('ds').reset_index(drop=True)
    
    # Merge future activities data
    if future_activities_data:
        activities_df = pd.DataFrame(future_activities_data)
        activities_df['ds'] = pd.to_datetime(activities_df['date'])
        activities_df = activities_df.rename(columns={'potential_sales': 'activity_sales_impact'})
        activities_df = activities_df[['ds', 'activity_sales_impact']].groupby('ds').sum().reset_index()
        df = pd.merge(df, activities_df, on='ds', how='left')
        df['activity_sales_impact'].fillna(0, inplace=True)
    else:
        df['activity_sales_impact'] = 0


    # Handle categorical weather data
    df = pd.get_dummies(df, columns=['weather'], drop_first=True, dtype=float)

    # Philippine Holidays
    holidays_list = [
        {'holiday': 'New Year\'s Day', 'ds': '2025-01-01'},
        {'holiday': 'Chinese New Year', 'ds': '2025-01-29'},
        {'holiday': 'EDSA People Power Revolution Anniversary', 'ds': '2025-02-25'},
        {'holiday': 'Day of Valor', 'ds': '2025-04-09'},
        {'holiday': 'Maundy Thursday', 'ds': '2025-04-17'},
        {'holiday': 'Good Friday', 'ds': '2025-04-18'},
        {'holiday': 'Black Saturday', 'ds': '2025-04-19'},
        {'holiday': 'Labor Day', 'ds': '2025-05-01'},
        {'holiday': 'Independence Day', 'ds': '2025-06-12'},
        {'holiday': 'Ninoy Aquino Day', 'ds': '2025-08-21'},
        {'holiday': 'National Heroes Day', 'ds': '2025-08-25'},
        {'holiday': 'All Saints\' Day', 'ds': '2025-11-01'},
        {'holiday': 'Bonifacio Day', 'ds': '2025-11-30'},
        {'holiday': 'Feast of the Immaculate Conception', 'ds': '2025-12-08'},
        {'holiday': 'Christmas Eve', 'ds': '2025-12-24'},
        {'holiday': 'Christmas Day', 'ds': '2025-12-25'},
        {'holiday': 'Rizal Day', 'ds': '2025-12-30'},
        {'holiday': 'New Year\'s Eve', 'ds': '2025-12-31'},
    ]
    holidays = pd.DataFrame(holidays_list)
    holidays['ds'] = pd.to_datetime(holidays['ds'])

    # 2. Train Prophet Model
    print("\nTraining Prophet model...")
    prophet_model = Prophet(holidays=holidays, daily_seasonality=True)
    
    # Add regressors if they exist after one-hot encoding
    # Ensure all possible weather columns are added to avoid errors if one is missing
    for col in ['weather_Rainy', 'weather_Storm', 'weather_Sunny']:
        if col in df.columns:
            prophet_model.add_regressor(col)
        else:
            df[col] = 0 # Add column with zeros if it doesn't exist in historical data

    prophet_model.fit(df)

    prophet_forecast = prophet_model.predict(df)
    df['prophet_pred'] = prophet_forecast['yhat']
    df['residuals'] = df['y'] - df['prophet_pred']
    print("Prophet model training complete.")

    # 3. Train XGBoost Model on Residuals
    print("\nTraining XGBoost model on residuals...")
    features_df = create_features(df.copy(), holidays)
    
    FEATURES = [
        'dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'weekofyear',
        'is_payday', 'days_after_payday', 'days_until_holiday',
        'customers_lag_7', 'rolling_mean_7', 'rolling_std_7',
        'activity_sales_impact'
    ]
    weather_cols = [col for col in features_df.columns if 'weather_' in col]
    FEATURES.extend(weather_cols)
    TARGET = 'residuals'

    X = features_df[FEATURES]
    y = features_df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    xgb_reg = xgb.XGBRegressor(
        n_estimators=1000,
        early_stopping_rounds=50,
        objective='reg:squarederror',
        learning_rate=0.01,
        # Hyperparameters can be tuned further with GridSearchCV
    )
    
    xgb_reg.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_test, y_test)],
              verbose=False) # Set to 100 to see training progress
    print("XGBoost model training complete.")

    # 4. Generate 15-Day Future Forecast
    print("\nGenerating 15-day future forecast...")
    future_dates = prophet_model.make_future_dataframe(periods=15, freq='D')
    
    # Add weather features from placeholder API
    weather_forecast = get_weather_forecast(15)
    future_dates_only = future_dates[future_dates['ds'] > df['ds'].max()].copy()
    future_dates_only['weather_Rainy'] = weather_forecast['weather_Rainy']
    future_dates_only['weather_Storm'] = weather_forecast['weather_Storm']
    future_dates_only['weather_Sunny'] = 1 - (future_dates_only['weather_Rainy'] + future_dates_only['weather_Storm'])
    
    # Merge weather back into the main future dataframe
    future_dates = pd.merge(future_dates, future_dates_only[['ds', 'weather_Rainy', 'weather_Storm', 'weather_Sunny']], on='ds', how='left')
    future_dates.fillna(0, inplace=True) # Fill historical part with 0s

    # Prophet prediction for the future
    prophet_future_forecast = prophet_model.predict(future_dates)

    # Prepare future features for XGBoost
    # We need to create features for the future dates, including lag/rolling features from historical data
    last_known_y = df[['ds', 'y']].copy()
    future_features_df = future_dates.copy()
    future_features_df = pd.merge(future_features_df, last_known_y, on='ds', how='left')
    future_features_df = create_features(future_features_df, holidays)
    
    # Add future activities to the future dataframe
    if future_activities_data:
        future_features_df = pd.merge(future_features_df, activities_df, on='ds', how='left')
        future_features_df['activity_sales_impact'].fillna(0, inplace=True)
    else:
        future_features_df['activity_sales_impact'] = 0

    future_X = future_features_df[FEATURES]

    # XGBoost prediction for the future
    xgb_future_residuals = xgb_reg.predict(future_X)

    # Combine forecasts
    final_forecast_df = prophet_future_forecast[['ds', 'yhat']].copy()
    final_forecast_df['xgb_residuals'] = xgb_future_residuals
    final_forecast_df['final_pred_customers'] = final_forecast_df['yhat'] + final_forecast_df['xgb_residuals']
    
    # Calculate dynamic ATV for sales forecast
    recent_df = df.tail(90)
    if not recent_df.empty and recent_df['y'].sum() > 0:
        dynamic_atv = (recent_df['sales'] + recent_df['add_on_sales']).sum() / recent_df['y'].sum()
    else:
        dynamic_atv = 150 # Fallback ATV

    final_forecast_df['final_pred_sales'] = final_forecast_df['final_pred_customers'] * dynamic_atv

    # Filter for only the next 15 days from today
    today = pd.Timestamp.now().normalize()
    future_15_days = final_forecast_df[final_forecast_df['ds'] >= today].head(15)
    
    print("\n--- 15-Day Sophisticated Forecast ---")
    print(future_15_days[['ds', 'final_pred_customers', 'final_pred_sales']])

    # 5. Save Forecast to Firestore
    print("\nSaving forecast to Firestore collection 'forecast_log'...")
    batch = db.batch()
    for index, row in future_15_days.iterrows():
        forecast_data = {
            'forecast_for_date': row['ds'],
            'predicted_customers': round(row['final_pred_customers']),
            'predicted_sales': round(row['final_pred_sales'], 2),
            'generated_on': datetime.now(),
            'model_version': 'prophet_xgb_v2_advanced'
        }
        doc_id = row['ds'].strftime('%Y-%m-%d')
        doc_ref = db.collection('forecast_log').document(doc_id)
        batch.set(doc_ref, forecast_data)
    
    batch.commit()
    print("Successfully saved 15-day forecast to Firestore.")


if __name__ == '__main__':
    run_forecast()
