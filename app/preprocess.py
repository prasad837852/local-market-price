# preprocess.py
import pandas as pd

def clean_data(df):
    # Drop rows with missing key fields
    df = df.dropna(subset=['arrival_date', 'market', 'commodity', 'modal_price'])

    # Convert price columns to numeric safely
    for col in ['min_price', 'max_price', 'modal_price']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows where modal price is invalid
    df = df.dropna(subset=['modal_price'])

    return df
