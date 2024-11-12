import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, timedelta
import logging
from typing import Dict, Optional, Tuple
from dotenv import load_dotenv
import os
import requests
import time
import json
import pathlib


#logger conf
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('weather_sales.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WeatherSalesPredictor:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('OWM_API_KEY')
        if not self.api_key:
            raise ValueError("OWM_API_KEY not found in environment variables")
        
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.is_trained = False
        
        #coordinates for London
        self.default_lat = 51.5074
        self.default_lon = -0.1278
        
        #directories for output files
        self.data_dir = pathlib.Path('data')
        self.weather_data_dir = self.data_dir / 'weather'
        self.weather_data_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_dir = self.data_dir / 'models'
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def get_weather_cache_path(self, date: datetime, lat: float, lon: float) -> pathlib.Path:
        date_str = date.strftime('%Y-%m-%d')
        return self.weather_data_dir / f"weather_{date_str}_{lat}_{lon}.json"

    def load_cached_weather(self, cache_path: pathlib.Path) -> Optional[Dict]:
        try:
            if cache_path.exists():
                logger.info(f"Loading cached weather data from {cache_path}")
                with open(cache_path, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            logger.error(f"Error loading cached weather data: {str(e)}")
            return None

    def save_weather_cache(self, cache_path: pathlib.Path, weather_data: Dict):
        try:
            logger.info(f"Saving weather data to cache: {cache_path}")
            with open(cache_path, 'w') as f:
                json.dump(weather_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving weather data to cache: {str(e)}")

    def get_weather_data(self, date: datetime, lat: float = None, lon: float = None) -> Optional[Dict]:
        try:
            lat = lat or self.default_lat
            lon = lon or self.default_lon
            
            #checkif there is a cache first
            cache_path = self.get_weather_cache_path(date, lat, lon)
            cached_data = self.load_cached_weather(cache_path)
            if cached_data:
                logger.info(f"Using cached weather data for {date.strftime('%Y-%m-%d')}")
                return cached_data
            
            logger.info(f"Fetching weather data for {date.strftime('%Y-%m-%d')} at coordinates ({lat}, {lon})")
            
            #datetime -> Unix 
            timestamp = int(date.timestamp())
            
            # openweather API URL
            url = f"https://api.openweathermap.org/data/3.0/onecall/timemachine?lat={lat}&lon={lon}&dt={timestamp}&appid={self.api_key}"
            
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            if 'data' in data and len(data['data']) > 0:
                weather = data['data'][0]
                weather_data = {
                    'temperature': weather['temp'],
                    'temp_max': weather['temp'],
                    'temp_min': weather['temp'],
                    'humidity': weather['humidity'],
                    'wind_speed': weather['wind_speed'],
                    'wind_gust': weather.get('wind_gust', 0),
                    'clouds': weather.get('clouds', 0),
                    'pressure': weather['pressure'],
                    'feels_like': weather['feels_like'],
                    'visibility': weather.get('visibility', 0),
                    'status': weather['weather'][0]['description'],
                    'weather_main': weather['weather'][0]['main'],
                    'weather_id': weather['weather'][0]['id'],
                    'timezone': data['timezone'],
                    'timezone_offset': data['timezone_offset']
                }
                
                # save cache
                self.save_weather_cache(cache_path, weather_data)
                
                logger.info(f"Successfully retrieved weather data for {date.strftime('%Y-%m-%d')}")
                return weather_data
                
            logger.warning(f"No weather data available for {date.strftime('%Y-%m-%d')}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting historical weather: {str(e)}")
            return None

    def prepare_training_data(self, sales_df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Starting training data preparation")
            df = sales_df.copy()
            df['Date'] = pd.to_datetime(df['Date'])
            
            #some simple features creation..
            df['day_of_week'] = df['Date'].dt.dayofweek
            df['month'] = df['Date'].dt.month
            df['is_weekend'] = df['Date'].dt.dayofweek.isin([5, 6]).astype(int)
            
            #historical weather data collection
            logger.info(f"Collecting weather data for {len(df)} dates")
            weather_features = []
            total_dates = len(df)
            
            for idx, date in enumerate(df['Date'], 1):
                logger.info(f"Processing weather data {idx}/{total_dates} for {date.strftime('%Y-%m-%d')}")
                weather = self.get_weather_data(date)
                if weather:
                    weather_features.append(weather)
                else:
                    weather_features.append({
                        'temperature': np.nan,
                        'temp_max': np.nan,
                        'temp_min': np.nan,
                        'humidity': np.nan,
                        'wind_speed': np.nan,
                        'clouds': np.nan
                    })
                #delay for api
                time.sleep(0.3)
            
            logger.info("Converting weather features to DataFrame")
            weather_df = pd.DataFrame(weather_features)
            
            #categorical f to numeric conversion
            weather_df['timezone'] = pd.Categorical(weather_df['timezone']).codes
            weather_df['weather_main'] = pd.Categorical(weather_df['weather_main']).codes

            
            final_df = pd.concat([df, weather_df], axis=1)
            
            #missing values handling.
            numeric_columns = final_df.select_dtypes(include=[np.number]).columns
            final_df[numeric_columns] = final_df[numeric_columns].fillna(
                final_df[numeric_columns].mean()
            )
            columns_to_drop = ['timezone_offset', 'weather_id', 'Date']
            final_df.drop(columns=columns_to_drop, inplace=True)

            #saving prepared data..
            prepared_data_path = self.data_dir / 'prepared_data.csv'
            final_df.to_csv(prepared_data_path, index=False)
            logger.info(f"Saved prepared training data to {prepared_data_path}")
            
            logger.info("Training data preparation completed successfully")
            return final_df
            
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            raise

    def train_model(self, sales_df: pd.DataFrame) -> Tuple[float, float]:
        try:
            logger.info("Starting model training process")
            
            # Prepare data
            df = self.prepare_training_data(sales_df)
            
            feature_columns = ['day_of_week', 'month', 'is_weekend', 
                             'temperature', 'temp_max', 'temp_min',
                             'humidity', 'wind_speed', 'clouds']
            
            X = df[feature_columns]
            y = df['Sales']
            

            logger.info("Splitting data into train and test sets")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            logger.info("Scaling features")
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            logger.info("Training linear regression model")
            self.model.fit(X_train_scaled, y_train)
            self.is_trained = True
            
            y_pred = self.model.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            metrics = {
                'r2_score': r2,
                'rmse': rmse,
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            metrics_path = self.model_dir / 'model_metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Saved model metrics to {metrics_path}")
            
            logger.info(f"Model trained successfully. R² score: {r2:.4f}, RMSE: {rmse:.2f}")
            
            return r2, rmse
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

    def predict_sales(self, date_str: str) -> Dict:
        if not self.is_trained:
            logger.error("Attempting to predict with untrained model")
            return {'error': 'Model not trained yet'}
            
        try:
            logger.info(f"Making sales prediction for date: {date_str}")
            date = datetime.strptime(date_str, '%Y-%m-%d')
            
            # Get weather data
            weather = self.get_weather_data(date)
            
            if not weather:
                logger.error(f"Could not get weather data for date: {date_str}")
                return {'error': 'Could not get weather data'}
                
            # features preparation
            features = {
                'day_of_week': date.weekday(),
                'month': date.month,
                'is_weekend': 1 if date.weekday() >= 5 else 0,
                'temperature': weather['temperature'],
                'temp_max': weather['temp_max'],
                'temp_min': weather['temp_min'],
                'humidity': weather['humidity'],
                'wind_speed': weather['wind_speed'],
                'clouds': weather['clouds'],
            }
            
            #scaling the features
            features_df = pd.DataFrame([features])
            features_scaled = self.scaler.transform(features_df)
            
            #prediction
            prediction = self.model.predict(features_scaled)[0]
            
            result = {
                'date': date_str,
                'predicted_sales': round(prediction, 2),
                'weather_conditions': weather
            }
            
            predictions_path = self.data_dir / 'predictions.csv'
            prediction_df = pd.DataFrame([{
                'date': date_str,
                'predicted_sales': round(prediction, 2),
                **weather
            }])
            
            if predictions_path.exists():
                prediction_df.to_csv(predictions_path, mode='a', header=False, index=False)
            else:
                prediction_df.to_csv(predictions_path, index=False)
            
            logger.info(f"Saved prediction for {date_str} to {predictions_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return {'error': str(e)}
        
def main():
    try:
        #loading the data
        try:
            sales_data = pd.read_excel('data/sales_data.xlsx')
            logger.info(f"Successfully loaded data from sales_data.xlsx with {len(sales_data)} records")         
            sales_data['Date'] = pd.to_datetime(sales_data['Date'])
            sales_data = sales_data.sort_values('Date').reset_index(drop=True)
            
        except FileNotFoundError:
            logger.error("sales_data.xlsx file not found in the current directory")
            raise
        except Exception as e:
            logger.error(f"Error reading Excel file: {str(e)}")
            raise
        
        #predictor init
        predictor = WeatherSalesPredictor()
        logger.info("WeatherSalesPredictor initialized successfully")
        
        #model training...
        logger.info("Starting model training...")
        r2, rmse = predictor.train_model(sales_data)
        logger.info(f"Model training completed with R² Score: {r2:.4f}, RMSE: {rmse:.2f}")
        
        print(f"\nModel Performance:")
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: {rmse:.2f}")
        
        #predict 5 next days.
        logger.info("Making predictions for next 5 days")
        print("\nPredictions for next 5 days:")
        
        prediction_results = []
        for i in range(1, 6):
            future_date = (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d')
            logger.info(f"Predicting sales for {future_date}")
            
            prediction = predictor.predict_sales(future_date)
            prediction_results.append(prediction)
            
            if 'error' not in prediction:
                print(f"\nDate: {prediction['date']}")
                print(f"Predicted Sales: ${prediction['predicted_sales']:,.2f}")
                print("Weather Conditions:")
                for key, value in prediction['weather_conditions'].items():
                    if key not in ['status', 'weather_id', 'timezone', 'timezone_offset']:
                        print(f"  - {key}: {value}")
                logger.info(f"Successfully predicted sales for {future_date}: ${prediction['predicted_sales']:,.2f}")
            else:
                print(f"\nError for {future_date}: {prediction['error']}")
                logger.error(f"Error predicting sales for {future_date}: {prediction['error']}")
        
        #saving the predictions...
        prediction_summary = pd.DataFrame([
            {
                'date': pred['date'],
                'predicted_sales': pred['predicted_sales'] if 'predicted_sales' in pred else None,
                'prediction_error': pred.get('error', None)
            }
            for pred in prediction_results
        ])
        
        summary_path = pathlib.Path('data') / 'prediction_summary.csv'
        prediction_summary.to_csv(summary_path, index=False)
        logger.info(f"Saved prediction summary to {summary_path}")
        
        print(f"\nPrediction summary saved to: {summary_path}")
                
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise

if __name__ == "__main__":
    logger.info("Starting sales predictor...")
    try:
        main()
        logger.info("Sales predictor completed successfully!")
    except Exception as e:
        logger.error(f"Sales predictor failed: {str(e)}")
        raise
