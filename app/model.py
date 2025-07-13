from sklearn.linear_model import LinearRegression
import pandas as pd

def train_model(df: pd.DataFrame):
    X = df[['days']]
    y = df['modal_price']
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_price(model, df, future_date):
    future_date = pd.to_datetime(future_date)
    days = (future_date - df['arrival_date'].min()).days
    return model.predict([[days]])[0]