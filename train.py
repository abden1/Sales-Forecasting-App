import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

def train_model(n_estimators, random_state):

    df = pd.read_csv("Amazon Sale Time Series.csv")
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)
    df = df.set_index('Date')

    y = df['Amount']
    x = df.drop(['Amount'], axis=1)

    # Time-Based Split (last 20% as test)
    test_size = int(len(x) * 0.2)
    x_train, x_test = x.iloc[:-test_size], x.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    mae = mean_absolute_error(y_test,y_pred)
    mse = mean_squared_error(y_test,y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test,y_pred)

    with mlflow.start_run():
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", random_state)
        mlflow.log_metric("Mean Absolute Error", mae)
        mlflow.log_metric("Mean Squared Error", mse)
        mlflow.log_metric("Root Mean Squared Error", rmse)
        mlflow.log_metric("R2 Score", r2)

        mlflow.sklearn.log_model(model, "Random Forest Regressor")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', type=int, default=10)
    parser.add_argument('--random_state', type=int, default=42)
    args = parser.parse_args()
    train_model(args.n_estimators, args.random_state)
