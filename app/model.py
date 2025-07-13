from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

def train_model(df):
    df_model = df.copy()
    df_model = df_model.sort_values('arrival_date')
    df_model = df_model.dropna(subset=['modal_price'])

    if len(df_model) < 2:
        return None, None  # Not enough data

    # Convert dates to ordinal
    df_model['days'] = df_model['arrival_date'].map(pd.Timestamp.toordinal)

    X = df_model[['days']]
    y = df_model['modal_price']

    model = LinearRegression()
    model.fit(X, y)

    return model, df_model


def predict_price(model, df_model, future_date):
    if model is None or df_model is None:
        return np.nan  # Invalid model/data

    days = pd.Timestamp(future_date).toordinal()
    prediction = model.predict([[days]])[0]

    # Optional: Clamp to reasonable bounds
    if prediction < 0:
        prediction = 0
    return prediction
