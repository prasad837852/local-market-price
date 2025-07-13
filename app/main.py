import streamlit as st
import pandas as pd
from datetime import date
from preprocess import clean_data
from model import train_model, predict_price

# Update to point to the new dataset
DATA_PATH = "dataset/local_prices.csv"  # Replace with the latest if needed

# Streamlit page settings
st.set_page_config(page_title="📈 Local Price Tracker", layout="wide")
st.title("🌾 Local Market Price Tracker")
st.markdown("Analyze and forecast produce prices across local markets using min, max, and modal prices.")

try:
    # ✅ Read CSV safely with correct encoding
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig', delimiter=",", on_bad_lines="skip")
    df['arrival_date'] = pd.to_datetime(df['arrival_date'], errors='coerce')
    df = clean_data(df)

    st.success(f"✅ Loaded {len(df)} records from {DATA_PATH}")

    # Sidebar filters
    st.sidebar.header("🔍 Filter Options")
    years = sorted(df['arrival_date'].dt.year.dropna().unique())
    months = sorted(df['arrival_date'].dt.month.dropna().unique())
    markets = sorted(df['market'].dropna().unique())
    commodities = sorted(df['commodity'].dropna().unique())

    year = st.sidebar.selectbox("Year", ["All"] + list(map(str, years)))
    month_names = ["All"] + [date(1900, m, 1).strftime('%B') for m in months]
    month = st.sidebar.selectbox("Month", month_names)
    market = st.sidebar.selectbox("Market", ["All"] + markets)
    commodity = st.sidebar.selectbox("Commodity", ["All"] + commodities)

    # Apply filters
    filtered = df.copy()
    if year != "All":
        filtered = filtered[filtered['arrival_date'].dt.year == int(year)]
    if month != "All":
        month_index = month_names.index(month)
        filtered = filtered[filtered['arrival_date'].dt.month == month_index]
    if market != "All":
        filtered = filtered[filtered['market'] == market]
    if commodity != "All":
        filtered = filtered[filtered['commodity'] == commodity]

    # Display filtered results
    st.subheader("📊 Filtered Market Data")
    st.write(f"Showing {len(filtered)} records")
    st.dataframe(filtered.sort_values(by="arrival_date"), use_container_width=True)

    if not filtered.empty:
        st.subheader("📈 Modal Price Trend")

        # Plot modal price trend
        trend = filtered.groupby("arrival_date")["modal_price"].mean().reset_index()
        st.line_chart(trend.set_index("arrival_date"))

        # Optional: min/max range info
        with st.expander("📌 View Monthly Min/Max Summary"):
            price_summary = filtered.groupby("arrival_date")[["min_price", "modal_price", "max_price"]].mean().reset_index()
            st.dataframe(price_summary.set_index("arrival_date"), use_container_width=True)

        st.subheader("🔮 Predict Future Modal Price")
        model, df_model = train_model(filtered)
        future_date = st.date_input("Select a future date", value=date.today())

        if st.button("Predict Modal Price"):
            pred_price = predict_price(model, df_model, pd.to_datetime(future_date))
            st.success(f"📅 Predicted modal price on {future_date.strftime('%d-%b-%Y')}: ₹{pred_price:.2f}")

    else:
        st.warning("⚠️ No data matches your filters.")

except Exception as e:
    st.error(f"❌ Error: {e}")
