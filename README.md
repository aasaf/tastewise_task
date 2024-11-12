# TasteW Task

## Setup Instructions

1. Create and activate virtual environment:
```bash
python3 -m venv venv
source ./venv/bin/activate
```
2. locate your api key under .env file
```bash
pip install -r requirements.txt
```

3. Configure API Key:
Create a .env file in the root directory and paste your api key to the file
```bash
OWM_API_KEY = 'your_owm_api_key'
```


4. Run the command: 
```bash
python ./main.py
```

## Task requirements:
1. Accessing weather api- implemented in `get_weather_data()` function
2. Integrate Weather Data- implemented in `prepare_training_data()` function 
3. Train a Simple Model- implemented in `train_model()` function
4. Implement a Prediction Function- implemented in `predictor.predict_sales()`
