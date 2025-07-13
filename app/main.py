import streamlit as st
import pandas as pd
from datetime import date
from preprocess import clean_data
from model import train_model, predict_price

# Path to the dataset
DATA_PATH = "dataset/local_prices.csv"

# Streamlit page setup
st.set_page_config(page_title="📈 Local Market Price Tracker", layout="wide")
st.title("🌾 Local Market Price Tracker")
st.markdown("Analyze and forecast produce prices across local markets using min, max, and modal prices.")

try:
    # Load dataset
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig', on_bad_lines='skip')
    df['arrival_date'] = pd.to_datetime(df['arrival_date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['arrival_date', 'min_price', 'modal_price', 'max_price', 'market', 'commodity'])

    # Optional cleaning
    df = clean_data(df)

    # Extract time-based fields
    df['year'] = df['arrival_date'].dt.year
    df['month'] = df['arrival_date'].dt.month
    df['month_name'] = df['arrival_date'].dt.strftime('%B')

    st.success(f"✅ Loaded {len(df)} records from `{DATA_PATH}`")

    # Sidebar filter options
    st.sidebar.header("🔍 Filter Options")
    selected_year = st.sidebar.selectbox("Year", ["All"] + sorted(df['year'].astype(str).unique().tolist()))
    selected_month = st.sidebar.selectbox("Month", ["All"] + df['month_name'].unique().tolist())
    selected_market = st.sidebar.selectbox("Market", ["All"] + sorted(df['market'].unique()))
    selected_commodity = st.sidebar.selectbox("Commodity", ["All"] + sorted(df['commodity'].unique()))

    # Filter data
    filtered_df = df.copy()
    if selected_year != "All":
        filtered_df = filtered_df[filtered_df['year'] == int(selected_year)]
    if selected_month != "All":
        filtered_df = filtered_df[filtered_df['month_name'] == selected_month]
    if selected_market != "All":
        filtered_df = filtered_df[filtered_df['market'] == selected_market]
    if selected_commodity != "All":
        filtered_df = filtered_df[filtered_df['commodity'] == selected_commodity]

    # Show filtered data
    st.subheader("📊 Filtered Market Data")
    st.write(f"Showing {len(filtered_df)} records")
    st.dataframe(filtered_df.sort_values(by='arrival_date'), use_container_width=True)

    if not filtered_df.empty:
        # Modal price trend
        st.subheader("📈 Modal Price Trend")
        trend = filtered_df.groupby("arrival_date")["modal_price"].mean().reset_index()
        st.line_chart(trend.set_index("arrival_date"))

        # Summary of min, max, modal
        with st.expander("📌 Daily Min/Max/Modal Summary"):
            summary = filtered_df.groupby("arrival_date")[["min_price", "modal_price", "max_price"]].mean().reset_index()
            st.dataframe(summary.set_index("arrival_date"), use_container_width=True)

        # Future price prediction
        st.subheader("🔮 Predict Future Modal Price")
        model, df_model = train_model(filtered_df)
        future_date = st.date_input("Select a future date", value=date.today())

        if st.button("Predict Modal Price"):
            pred = predict_price(model, df_model, pd.to_datetime(future_date))
            st.success(f"📅 Predicted modal price on {future_date.strftime('%d-%b-%Y')}: ₹{pred:.2f}")
    else:
        st.warning("⚠️ No data available for the selected filters.")

except Exception as e:
    st.error(f"❌ Error: {e}")
