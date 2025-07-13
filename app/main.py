import streamlit as st
import pandas as pd
from datetime import date, datetime
from preprocess import clean_data
from model import train_model, predict_price
from deep_translator import GoogleTranslator

DATA_PATH = "dataset/local_prices.csv"

st.set_page_config(page_title="📈 Local Market Price Tracker", layout="wide")
st.title("🌾 Local Market Price Tracker")
st.markdown("Analyze and forecast produce prices across local markets using min, max, and modal prices.")

def translate(text, target_lang='te'):
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except:
        return text

try:
    # Load and parse date
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig', on_bad_lines='skip')
    df['arrival_date'] = pd.to_datetime(df['arrival_date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['arrival_date'])

    # Clean data
    df = clean_data(df)

    # Extract year/month for filters
    df['year'] = df['arrival_date'].dt.year
    df['month'] = df['arrival_date'].dt.month
    df['month_name'] = df['arrival_date'].dt.strftime('%B')

    st.success(f"✅ Loaded {len(df)} records from `{DATA_PATH}`")

    # Sidebar filters
    st.sidebar.header("🔍 Filter Options")
    selected_lang = st.sidebar.selectbox("🌐 Language", ["English", "Telugu", "Hindi"])

    def t(text):
        if selected_lang == "English":
            return text
        lang_map = {"Telugu": "te", "Hindi": "hi"}
        return translate(text, target_lang=lang_map[selected_lang])

    year_list = sorted(df['year'].unique())
    market_list = sorted(df['market'].dropna().unique())
    commodity_list = sorted(df['commodity'].dropna().unique())
    month_list = sorted(df['month'].unique())

    selected_year = st.sidebar.selectbox(t("Year"), ["All"] + list(map(str, year_list)))
    selected_month = st.sidebar.selectbox(t("Month"), ["All"] + [date(1900, m, 1).strftime('%B') for m in month_list])
    selected_market = st.sidebar.selectbox(t("Market"), ["All"] + list(market_list))
    selected_commodity = st.sidebar.selectbox(t("Commodity"), ["All"] + list(commodity_list))

    # Apply filters
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

    st.subheader(t("📊 Filtered Market Data"))
    st.write(f"{t('Showing')} {len(filtered_df)} {t('records')}")
    st.dataframe(filtered_df.sort_values(by="arrival_date"), use_container_width=True)

    if not filtered_df.empty:
        st.subheader(t("📈 Modal Price Trend"))
        modal_trend = filtered_df.groupby("arrival_date")["modal_price"].mean().reset_index()
        st.line_chart(modal_trend.set_index("arrival_date"))

        with st.expander(t("📌 View Daily Min/Max/Modal Summary")):
            summary = filtered_df.groupby("arrival_date")[["min_price", "modal_price", "max_price"]].mean().reset_index()
            st.dataframe(summary.set_index("arrival_date"), use_container_width=True)

        st.subheader(t("🔮 Predict Future Modal Price"))
        model, df_model = train_model(filtered_df)
        future_date = st.date_input(t("Select a future date"), value=date.today())

        if st.button(t("Predict Modal Price")):
            pred = predict_price(model, df_model, pd.to_datetime(future_date))
            st.success(f"📅 {t('Predicted modal price on')} {future_date.strftime('%d-%b-%Y')}: ₹{pred:.2f}")
    else:
        st.warning(t("⚠️ No data available for the selected filters."))

except Exception as e:
    st.error(f"❌ Error: {e}")
