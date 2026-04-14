# ================================
# 1. IMPORT LIBRARIES
# ================================
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ================================
# 2. PAGE CONFIG
# ================================
st.set_page_config(page_title="UAC Dashboard", layout="wide")

# ================================
# 3. LOAD DATA
# ================================
@st.cache_data
def load_data():
    df = pd.read_csv("HHS.csv")

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')
    df.set_index('Date', inplace=True)

    df.rename(columns={
        'Children transferred out of CBP custody': 'Transfers',
        'Children discharged from HHS Care': 'Discharges',
        'Children in CBP custody': 'CBP Custody',
        'Children in HHS Care': 'HHS Care'
    }, inplace=True)

    cols = ['Transfers', 'Discharges', 'CBP Custody', 'HHS Care']
    for col in cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(',', '')
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.groupby(df.index).sum()
    df = df.asfreq('D')
    df = df.fillna(method='ffill')

    return df

df = load_data()

# ================================
# 4. FEATURE ENGINEERING
# ================================
df['Total_Load'] = df['CBP Custody'] + df['HHS Care']
df['Net_Intake'] = df['Transfers'] - df['Discharges']
df['7_day_avg'] = df['Total_Load'].rolling(7).mean()

# ================================
# 5. SIDEBAR FILTERS
# ================================
st.sidebar.header("🔍 Filters")

start_date = st.sidebar.date_input("Start Date", df.index.min())
end_date = st.sidebar.date_input("End Date", df.index.max())

filtered_df = df.loc[start_date:end_date]

show_forecast = st.sidebar.checkbox("Show Forecast")

# ================================
# 6. KPI CALCULATIONS
# ================================
kpis = {
    "Total Children": int(filtered_df['Total_Load'].iloc[-1]),
    "Avg Net Intake": round(filtered_df['Net_Intake'].mean(), 2),
    "Max Load": int(filtered_df['Total_Load'].max()),
    "Min Load": int(filtered_df['Total_Load'].min())
}

# ================================
# 7. TITLE
# ================================
st.title("📊 UAC System Capacity Dashboard")

# ================================
# 8. KPI CARDS
# ================================
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Children", kpis["Total Children"])
col2.metric("Avg Net Intake", kpis["Avg Net Intake"])
col3.metric("Max Load", kpis["Max Load"])
col4.metric("Min Load", kpis["Min Load"])

# ================================
# 9. CHARTS
# ================================
st.subheader("📈 Total Load Over Time")
st.line_chart(filtered_df[['Total_Load', '7_day_avg']])

st.subheader("🏥 CBP vs HHS Load")
st.area_chart(filtered_df[['CBP Custody', 'HHS Care']])

st.subheader("📊 Net Intake Trend")
st.bar_chart(filtered_df['Net_Intake'])

# ================================
# 10. FORECAST
# ================================
if show_forecast:
    try:
        from statsmodels.tsa.arima.model import ARIMA

        model = ARIMA(filtered_df['Total_Load'], order=(5,1,0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=30)

        st.subheader("🔮 30-Day Forecast")

        fig, ax = plt.subplots()
        ax.plot(filtered_df.index, filtered_df['Total_Load'], label="Actual")

        future_dates = pd.date_range(filtered_df.index[-1], periods=30, freq='D')
        ax.plot(future_dates, forecast, label="Forecast")

        ax.legend()
        st.pyplot(fig)

    except:
        st.warning("Forecasting not available")