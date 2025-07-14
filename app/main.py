import streamlit as st
import pandas as pd
from datetime import date, datetime
from preprocess import clean_data
from model import train_model, predict_price

DATA_PATH = "dataset/local_prices.csv"

# -------------------- LOGIN SYSTEMM --------------------

def login():
    st.set_page_config(page_title="📈 Local Market Price Tracker", layout="centered")
    st.title("🔐 Login to Market Price Tracker")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "Prabha" and password == "1234":
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("✅ Login successful!")
            st.rerun()
        else:
            st.error("❌ Invalid username or password.")

# -------------------- MAIN APP --------------------

def run_app():
    st.set_page_config(page_title="📈 Local Market Price Tracker", layout="wide")
    st.title("🌾 Local Market Price Tracker")
    st.markdown("Analyze and forecast produce prices across local markets using min, max, and modal prices.")

    # Logout
    st.sidebar.success(f"Welcome, {st.session_state.get('username', 'User')}!")
    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

    try:
        df = pd.read_csv(DATA_PATH, encoding='utf-8-sig', on_bad_lines='skip')
        df['arrival_date'] = pd.to_datetime(df['arrival_date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['arrival_date'])

        df = clean_data(df)

        df['year'] = df['arrival_date'].dt.year
        df['month'] = df['arrival_date'].dt.month
        df['month_name'] = df['arrival_date'].dt.strftime('%B')

        st.success(f"✅ Loaded {len(df)} records from `{DATA_PATH}`")

        st.sidebar.header("🔍 Filter Options")
        year_list = sorted(df['year'].unique())
        market_list = sorted(df['market'].dropna().unique())
        commodity_list = sorted(df['commodity'].dropna().unique())
        month_list = sorted(df['month'].unique())

        selected_year = st.sidebar.selectbox("Year", ["All"] + list(map(str, year_list)))
        selected_month = st.sidebar.selectbox("Month", ["All"] + [date(1900, m, 1).strftime('%B') for m in month_list])
        selected_market = st.sidebar.selectbox("Market", ["All"] + list(market_list))
        selected_commodity = st.sidebar.selectbox("Commodity", ["All"] + list(commodity_list))

        filtered_df = df.copy()
        if selected_year != "All":
            filtered_df = filtered_df[filtered_df['year'] == int(selected_year)]
        if selected_month != "All":
            month_index = datetime.strptime(selected_month, '%B').month
            filtered_df = filtered_df[filtered_df['month'] == month_index]
        if selected_market != "All":
            filtered_df = filtered_df[filtered_df['market'] == selected_market]
        if selected_commodity != "All":
            filtered_df = filtered_df[filtered_df['commodity'] == selected_commodity]

        st.subheader("📊 Filtered Market Data")
        st.write(f"Showing {len(filtered_df)} records")
        st.dataframe(filtered_df.sort_values(by="arrival_date"), use_container_width=True)

        if not filtered_df.empty:
            st.subheader("📈 Modal Price Trend")
            modal_trend = filtered_df.groupby("arrival_date")["modal_price"].mean().reset_index()
            st.line_chart(modal_trend.set_index("arrival_date"))

            with st.expander("📌 View Daily Min/Max/Modal Summary"):
                summary = filtered_df.groupby("arrival_date")[["min_price", "modal_price", "max_price"]].mean().reset_index()
                st.dataframe(summary.set_index("arrival_date"), use_container_width=True)

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

# -------------------- APP ENTRY --------------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
else:
    run_app()
