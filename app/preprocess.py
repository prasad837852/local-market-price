import pandas as pd

def clean_data(filepath: str):
    df = pd.read_csv(filepath)
    df['arrival_date'] = pd.to_datetime(df['arrival_date'])
    df['days'] = (df['arrival_date'] - df['arrival_date'].min()).dt.days
    return df