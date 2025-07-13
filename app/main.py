import streamlit as st
from preprocess import clean_data
from model import train_model, predict_price

# Load and preprocess data
df = clean_data("dataset/tomato_prices_hyderabad.csv")
model = train_model(df)

# Streamlit UI
st.set_page_config(page_title="Tomato Price Tracker", layout="centered")
st.title("🌾 Local Market Price Tracker for Farmers")
st.subheader("Market: Bowenpally, Hyderabad")

# Show trend
st.line_chart(df.set_index("arrival_date")["modal_price"])

# Prediction
future_date = st.date_input("📅 Select future date to predict price:")
if future_date:
    predicted_price = predict_price(model, df, future_date)
    st.success(f"🔮 Modal Price on {future_date}: ₹{round(predicted_price)} /kg")

# Show dataset (optional)
if st.checkbox("Show raw dataset"):
    st.dataframe(df[['arrival_date', 'modal_price', 'min_price', 'max_price']])