import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, timedelta
import warnings

# --- Global Initialization (Keep this light!) ---
# Only initialize Firebase here. All heavy imports will be moved inside the function.
warnings.filterwarnings("ignore", message="A value is trying to be set on a copy of a slice from a DataFrame")

try:
    if not firebase_admin._apps:
        firebase_admin.initialize_app()
    db = firestore.client()
    print("Firebase Admin SDK initialized globally.")
except Exception as e:
    print(f"Global Firebase initialization failed: {e}")

# --- Main Function ---
def run_daily_forecast(event=None, context=None):
    """
    This is our Google Cloud Function.
    Heavy libraries are imported inside this function to ensure fast startup.
    """
    # --- Step 1: Import Heavy Libraries ---
    # This is the key change: we import these only when the function runs.
    import pandas as pd
    from prophet import Prophet
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    import numpy as np
    print("Heavy libraries imported successfully inside the function.")

    # --- Step 2: Helper Functions (defined inside the main function) ---
    def fetch_firestore_collection(collection_name):
        """Fetches all documents from a Firestore collection."""
        docs = db.collection(collection_name).stream()
        data = [doc.to_dict() for doc in docs]
        print(f"Fetched {len(data)} documents from '{collection_name}'.")
        return data

    def create_features(df, holidays_df):
        """Create time series features from a datetime index."""
        df['dayofweek'] = df['ds'].dt.dayofweek
        df['quarter'] = df['ds'].dt.quarter
        df['month'] = df['ds'].dt.month
        df['year'] = df['ds'].dt.year
        df['dayofyear'] = df['ds'].dt.dayofyear
        df['weekofyear'] = df['ds'].dt.isocalendar().week.astype(int)
        df['is_payday'] = df['ds'].dt.is_month_end | (df['ds'].dt.day == 15)
        df['days_after_payday'] = df.groupby(df['ds'].dt.to_period('M')).cumcount()
        df.loc[df['is_payday'], 'days_after_payday'] = 0
        holidays_df = holidays_df.sort_values('ds')
        df = pd.merge_asof(df.sort_values('ds'), holidays_df[['ds']].rename(columns={'ds':'holiday_date'}), 
                           left_on='ds', right_on='holiday_date', direction='forward')
        df['days_until_holiday'] = (df['holiday_date'] - df['ds']).dt.days.fillna(365)
        df = df.drop(columns=['holiday_date'])
        df = df.sort_values('ds')
        df['customers_lag_7'] = df['y'].shift(7)
        df['rolling_mean_7'] = df['y'].shift(1).rolling(window=7).mean()
        df['rolling_std_7'] = df['y'].shift(1).rolling(window=7).std()
        df.fillna(method='bfill', inplace=True)
        df.fillna(method='ffill', inplace=True)
        return df

    def get_weather_forecast(days=15):
        """Placeholder for a real weather API call."""
        return {'weather_Rainy': np.zeros(days), 'weather_Storm': np.zeros(days)}

    # --- Step 3: Forecasting Logic (The rest of the original script) ---
    print("\nStarting the serverless forecasting process...")
    
    historical_data = fetch_firestore_collection('historical_data')
    future_activities_data = fetch_firestore_collection('future_activities')
    
    if not historical_data:
        print("No historical data found. Exiting.")
        return

    df = pd.DataFrame(historical_data)
    df['ds'] = pd.to_datetime(df['date'])
    df = df.rename(columns={'customers': 'y'})
    df = df[['ds', 'y', 'sales', 'add_on_sales', 'weather']]
    df = df.sort_values('ds').reset_index(drop=True)
    
    if future_activities_data:
        activities_df = pd.DataFrame(future_activities_data)
        activities_df['ds'] = pd.to_datetime(activities_df['date'])
        activities_df = activities_df.rename(columns={'potential_sales': 'activity_sales_impact'})
        activities_df = activities_df[['ds', 'activity_sales_impact']].groupby('ds').sum().reset_index()
        df = pd.merge(df, activities_df, on='ds', how='left')
        df['activity_sales_impact'].fillna(0, inplace=True)
    else:
        df['activity_sales_impact'] = 0

    df = pd.get_dummies(df, columns=['weather'], drop_first=True, dtype=float)

    holidays_list = [
        {'holiday': 'New Year\'s Day', 'ds': '2025-01-01'}, {'holiday': 'Chinese New Year', 'ds': '2025-01-29'},
        {'holiday': 'EDSA People Power Revolution Anniversary', 'ds': '2025-02-25'}, {'holiday': 'Day of Valor', 'ds': '2025-04-09'},
        {'holiday': 'Maundy Thursday', 'ds': '2025-04-17'}, {'holiday': 'Good Friday', 'ds': '2025-04-18'},
        {'holiday': 'Black Saturday', 'ds': '2025-04-19'}, {'holiday': 'Labor Day', 'ds': '2025-05-01'},
        {'holiday': 'Independence Day', 'ds': '2025-06-12'}, {'holiday': 'Ninoy Aquino Day', 'ds': '2025-08-21'},
        {'holiday': 'National Heroes Day', 'ds': '2025-08-25'}, {'holiday': 'All Saints\' Day', 'ds': '2025-11-01'},
        {'holiday': 'Bonifacio Day', 'ds': '2025-11-30'}, {'holiday': 'Feast of the Immaculate Conception', 'ds': '2025-12-08'},
        {'holiday': 'Christmas Eve', 'ds': '2025-12-24'}, {'holiday': 'Christmas Day', 'ds': '2025-12-25'},
        {'holiday': 'Rizal Day', 'ds': '2025-12-30'}, {'holiday': 'New Year\'s Eve', 'ds': '2025-12-31'},
    ]
    holidays = pd.DataFrame(holidays_list)
    holidays['ds'] = pd.to_datetime(holidays['ds'])

    prophet_model = Prophet(holidays=holidays, daily_seasonality=True)
    
    for col in ['weather_Rainy', 'weather_Storm', 'weather_Sunny']:
        if col in df.columns:
            prophet_model.add_regressor(col)
        else:
            df[col] = 0

    prophet_model.fit(df)
    prophet_forecast = prophet_model.predict(df)
    df['prophet_pred'] = prophet_forecast['yhat']
    df['residuals'] = df['y'] - df['prophet_pred']

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
    xgb_reg = xgb.XGBRegressor(n_estimators=1000, early_stopping_rounds=50, objective='reg:squarederror', learning_rate=0.01)
    xgb_reg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)

    future_dates = prophet_model.make_future_dataframe(periods=15, freq='D')
    weather_forecast = get_weather_forecast(15)
    future_dates_only = future_dates[future_dates['ds'] > df['ds'].max()].copy()
    future_dates_only['weather_Rainy'] = weather_forecast['weather_Rainy']
    future_dates_only['weather_Storm'] = weather_forecast['weather_Storm']
    future_dates_only['weather_Sunny'] = 1 - (future_dates_only['weather_Rainy'] + future_dates_only['weather_Storm'])
    future_dates = pd.merge(future_dates, future_dates_only[['ds', 'weather_Rainy', 'weather_Storm', 'weather_Sunny']], on='ds', how='left')
    future_dates.fillna(0, inplace=True)

    prophet_future_forecast = prophet_model.predict(future_dates)
    last_known_y = df[['ds', 'y']].copy()
    future_features_df = future_dates.copy()
    future_features_df = pd.merge(future_features_df, last_known_y, on='ds', how='left')
    future_features_df = create_features(future_features_df, holidays)
    
    if future_activities_data:
        future_features_df = pd.merge(future_features_df, activities_df, on='ds', how='left')
        future_features_df['activity_sales_impact'].fillna(0, inplace=True)
    else:
        future_features_df['activity_sales_impact'] = 0

    future_X = future_features_df[FEATURES]
    xgb_future_residuals = xgb_reg.predict(future_X)

    final_forecast_df = prophet_future_forecast[['ds', 'yhat']].copy()
    final_forecast_df['xgb_residuals'] = xgb_future_residuals
    final_forecast_df['final_pred_customers'] = final_forecast_df['yhat'] + final_forecast_df['xgb_residuals']
    
    recent_df = df.tail(90)
    dynamic_atv = (recent_df['sales'].sum() + recent_df['add_on_sales'].sum()) / recent_df['y'].sum() if not recent_df.empty and recent_df['y'].sum() > 0 else 150
    final_forecast_df['final_pred_sales'] = final_forecast_df['final_pred_customers'] * dynamic_atv

    today = pd.Timestamp.now().normalize()
    future_15_days = final_forecast_df[final_forecast_df['ds'] >= today].head(15)
    
    print("\n--- Saving 15-Day Sophisticated Forecast to Firestore ---")
    batch = db.batch()
    for index, row in future_15_days.iterrows():
        forecast_data = {
            'forecast_for_date': row['ds'],
            'predicted_customers': round(row['final_pred_customers']),
            'predicted_sales': round(row['final_pred_sales'], 2),
            'generated_on': datetime.now(),
            'model_version': 'prophet_xgb_v2_advanced_serverless'
        }
        doc_id = row['ds'].strftime('%Y-%m-%d')
        doc_ref = db.collection('forecast_log').document(doc_id)
        batch.set(doc_ref, forecast_data)
    
    batch.commit()
    print("Successfully saved 15-day forecast to Firestore.")

# This part is for local testing if you want to run `python main.py`
if __name__ == '__main__':
    run_daily_forecast()
