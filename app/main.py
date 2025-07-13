import streamlit as st
import pandas as pd
from datetime import date, datetime
from preprocess import clean_data
from model import train_model, predict_price
from deep_translator import GoogleTranslator

DATA_PATH = "dataset/local_prices.csv"

# Translation helper
def translate(text, lang):
    try:
        return GoogleTranslator(source='en', target=lang).translate(text)
    except Exception:
        return text  # fallback

# Language selector
lang = st.sidebar.selectbox("🌐 Select Language", ["en", "te", "hi", "ta", "ml", "kn"])
_ = lambda text: translate(text, lang) if lang != "en" else text

st.set_page_config(page_title="📈 Local Market Price Tracker", layout="wide")
st.title(_("🌾 Local Market Price Tracker"))
st.markdown(_("Analyze and forecast produce prices across local markets using min, max, and modal prices."))

try:
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig', on_bad_lines='skip')
    df['arrival_date'] = pd.to_datetime(df['arrival_date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['arrival_date'])
    df = clean_data(df)

    df['year'] = df['arrival_date'].dt.year
    df['month'] = df['arrival_date'].dt.month
    df['month_name'] = df['arrival_date'].dt.strftime('%B')

    st.success(_("✅ Loaded {0} records from `{1}`").format(len(df), DATA_PATH))

    st.sidebar.header(_("🔍 Filter Options"))
    year_list = sorted(df['year'].unique())
    market_list = sorted(df['market'].dropna().unique())
    commodity_list = sorted(df['commodity'].dropna().unique())
    month_list = sorted(df['month'].unique())

    selected_year = st.sidebar.selectbox(_("Year"), ["All"] + list(map(str, year_list)))
    selected_month = st.sidebar.selectbox(_("Month"), ["All"] + [date(1900, m, 1).strftime('%B') for m in month_list])
    selected_market = st.sidebar.selectbox(_("Market"), ["All"] + list(market_list))
    selected_commodity = st.sidebar.selectbox(_("Commodity"), ["All"] + list(commodity_list))

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

    st.subheader(_("📊 Filtered Market Data"))
    st.write(_("Showing {0} records").format(len(filtered_df)))
    st.dataframe(filtered_df.sort_values(by="arrival_date"), use_container_width=True)

    if not filtered_df.empty:
        st.subheader(_("📈 Modal Price Trend"))
        modal_trend = filtered_df.groupby("arrival_date")["modal_price"].mean().reset_index()
        st.line_chart(modal_trend.set_index("arrival_date"))

        with st.expander(_("📌 View Daily Min/Max/Modal Summary")):
            summary = filtered_df.groupby("arrival_date")[["min_price", "modal_price", "max_price"]].mean().reset_index()
            st.dataframe(summary.set_index("arrival_date"), use_container_width=True)

        st.subheader(_("🔮 Predict Future Modal Price"))
        model, df_model = train_model(filtered_df)
        future_date = st.date_input(_("Select a future date"), value=date.today())

        if st.button(_("Predict Modal Price")):
            pred = predict_price(model, df_model, pd.to_datetime(future_date))
            st.success(_("📅 Predicted modal price on {0}: ₹{1:.2f}").format(
                future_date.strftime('%d-%b-%Y'), pred))
    else:
        st.warning(_("⚠️ No data available for the selected filters."))

except Exception as e:
    st.error(_("❌ Error: ") + str(e))
