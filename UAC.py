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

    # Handle duplicate dates
    df = df.groupby(df.index).sum()

    # Fill missing dates
    df = df.asfreq('D')

    # Fill missing values
    df = df.ffill()

    return df

df = load_data()

# ================================
# 4. FEATURE ENGINEERING
# ================================
df['Total_Load'] = df['CBP Custody'] + df['HHS Care']
df['Net_Intake'] = df['Transfers'] - df['Discharges']
df['Growth_Rate'] = df['Total_Load'].pct_change() * 100
df['Backlog'] = df['Net_Intake'].rolling(7).sum()
df['7_day_avg'] = df['Total_Load'].rolling(7).mean()
df['14_day_avg'] = df['Total_Load'].rolling(14).mean()

# ================================
# 5. DATA VALIDATION
# ================================
df['Invalid_Transfer'] = df['Transfers'] > df['CBP Custody']
df['Invalid_Discharge'] = df['Discharges'] > df['HHS Care']

# ================================
# 6. SIDEBAR FILTERS
# ================================
st.sidebar.header("🔍 Filters")

start_date = st.sidebar.date_input("Start Date", df.index.min())
end_date = st.sidebar.date_input("End Date", df.index.max())

filtered_df = df.loc[start_date:end_date]

# Time Granularity
st.sidebar.subheader("📅 Time Granularity")
granularity = st.sidebar.selectbox("Select", ["Daily", "Weekly", "Monthly"])

if granularity == "Weekly":
    filtered_df = filtered_df.resample('W').mean()
elif granularity == "Monthly":
    filtered_df = filtered_df.resample('M').mean()

show_forecast = st.sidebar.checkbox("Show Forecast")

# ================================
# 7. KPI CALCULATIONS
# ================================
kpis = {
    "Total Load": int(filtered_df['Total_Load'].iloc[-1]),
    "Net Intake Pressure": round(filtered_df['Net_Intake'].mean(), 2),
    "Volatility Index": round(filtered_df['Total_Load'].std(), 2),
    "Backlog Rate": int(filtered_df['Net_Intake'].sum()),
    "Discharge Ratio": round(
        filtered_df['Discharges'].sum() / filtered_df['Transfers'].sum(), 2
    ) if filtered_df['Transfers'].sum() != 0 else 0
}

# ================================
# 8. TITLE
# ================================
st.title("📊 UAC System Capacity Dashboard")

# ================================
# 9. KPI DISPLAY
# ================================
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Load", kpis["Total Load"])
col2.metric("Net Intake", kpis["Net Intake Pressure"])
col3.metric("Volatility", kpis["Volatility Index"])
col4.metric("Backlog", kpis["Backlog Rate"])
col5.metric("Discharge Ratio", kpis["Discharge Ratio"])

st.markdown("---")

# ================================
# 10. CHARTS
# ================================
st.subheader("📈 Total Load Over Time")
st.line_chart(filtered_df[['Total_Load', '7_day_avg', '14_day_avg']])

st.markdown("---")

st.subheader("🏥 CBP vs HHS Load")
st.area_chart(filtered_df[['CBP Custody', 'HHS Care']])

st.markdown("---")

st.subheader("📊 Net Intake Trend")
st.bar_chart(filtered_df['Net_Intake'])

st.markdown("---")

st.subheader("📊 Backlog Trend")
st.line_chart(filtered_df['Backlog'])

st.markdown("---")

st.subheader("📈 Growth Rate (%)")
st.line_chart(filtered_df['Growth_Rate'])

st.markdown("---")

# ================================
# 11. DATA VALIDATION DISPLAY
# ================================
st.subheader("⚠️ Data Validation Issues")

st.write("Invalid Transfers:", int(filtered_df['Invalid_Transfer'].sum()))
st.write("Invalid Discharges:", int(filtered_df['Invalid_Discharge'].sum()))

st.markdown("---")

# ================================
# 12. HIGH LOAD ANALYSIS
# ================================
st.subheader("🔥 High Load Periods")

threshold = filtered_df['Total_Load'].mean() + filtered_df['Total_Load'].std()
high_load = filtered_df[filtered_df['Total_Load'] > threshold]

st.write(high_load[['Total_Load']].tail())

st.markdown("---")

# ================================
# 13. INSIGHTS
# ================================
st.subheader("💡 Key Insights")

st.info("📌 Sustained positive net intake indicates increasing system pressure.")
st.info("📌 High volatility suggests unstable system behavior.")
st.info("📌 Discharge ratio reflects efficiency of system.")

# ================================
# 14. FORECAST
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
