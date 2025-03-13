# تطبيق التنبؤ بالمبيعات | Sales Forecasting Application

## الوصف | Description
تطبيق ويب للتنبؤ بالمبيعات المستقبلية باستخدام نموذج ARIMA. يقوم التطبيق بتحليل البيانات التاريخية وتقديم توقعات دقيقة للمبيعات المستقبلية.

A web application for forecasting future sales using the ARIMA model. The application analyzes historical data and provides accurate predictions for future sales.

## المتطلبات | Requirements
- Python 3.8 أو أحدث | Python 3.8 or newer
- pip (مدير حزم Python) | pip (Python package manager)

## التثبيت | Installation
1. قم بتنزيل أو استنساخ المستودع | Download or clone the repository:
```bash
git clone https://github.com/yourusername/sales-forecasting.git
cd sales-forecasting
```

2. قم بتثبيت التبعيات | Install dependencies:
```bash
pip install -e .
```

## التشغيل | Running the Application
1. قم بتشغيل التطبيق | Start the application:
```bash
python -m sales-forecast
```

2. افتح المتصفح على | Open your browser at:
```
http://localhost:8000
```

## الميزات | Features
- تحليل البيانات التاريخية | Historical data analysis
- التنبؤ بالمبيعات المستقبلية | Future sales forecasting
- عرض النتائج بيانياً | Graphical results display
- تدريب النموذج وتحديثه | Model training and updating

## الترخيص | License
MIT License

## Features

- Sales forecasting using ARIMA and Prophet models
- Model performance evaluation
- Interactive visualization of forecasts
- Configurable forecast horizon
- Real-time model training and prediction

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset from Kaggle:
[E-commerce Sales Data](https://www.kaggle.com/datasets/thedevastator/unlock-profits-with-e-commerce-sales-data)

4. Place the downloaded CSV file in the `data` directory as:
```
data/International sale Report.csv
```

## Running the Application

1. Start the Flask server:
```bash
python app/app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

## Using the Application

1. Select the desired forecasting model (ARIMA or Prophet)
2. Set the forecast horizon (number of days to forecast)
3. Click "Train Model" to train the selected model
4. Click "Generate Forecast" to see the predictions
5. View performance metrics and forecast visualization

## Model Details

### ARIMA (Autoregressive Integrated Moving Average)
- Default order: (1,1,1)
- Handles temporal dependencies
- Good for stationary time series

### Prophet
- Handles yearly, weekly, and daily seasonality
- Robust to missing data
- Automatically detects changepoints

## Performance Metrics

- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Square Error)

## Project Structure

```
├── app/
│   ├── app.py              # Main Flask application
│   ├── models/
│   │   └── forecasting.py  # Model implementations
│   ├── static/             # Static files
│   └── templates/
│       └── index.html      # Web interface
├── data/                   # Dataset directory
├── requirements.txt        # Python dependencies
└── README.md              # This file
``` 