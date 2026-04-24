 
# 1. IMPORT LIBRARIES
 

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA


 
# 2. PAGE CONFIG
 

st.set_page_config(page_title="UAC Dashboard", layout="wide")


 
# 3. LOAD DATA (ROBUST DATE HANDLING)
 

@st.cache_data
def load_data():
    df = pd.read_csv("HHS.csv")

    # 🔥 FIX FOR EXCEL SERIAL DATE ISSUE
    if df['Date'].dtype != 'datetime64[ns]':
        try:
            df['Date'] = pd.to_datetime(df['Date'])
        except:
            df['Date'] = pd.to_datetime('1899-12-30') + pd.to_timedelta(df['Date'], unit='D')

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
        df[col] = df[col].astype(str).str.replace(',', '').str.strip()
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.groupby(df.index).sum()
    df = df.asfreq('D')
    df = df.ffill()

    return df


df = load_data()


 
# 4. FEATURE ENGINEERING
 

df['Total_Load'] = df['CBP Custody'] + df['HHS Care']
df['Net_Intake'] = df['Transfers'] - df['Discharges']
df['7_day_avg'] = df['Total_Load'].rolling(7).mean()
df['14_day_avg'] = df['Total_Load'].rolling(14).mean()
df['Growth_Rate'] = df['Total_Load'].pct_change() * 100
df['Backlog'] = df['Net_Intake'].cumsum()


 
# 5. SIDEBAR FILTERS
 

st.sidebar.header("🔍 Filters")

start_date = st.sidebar.date_input("Start Date", df.index.min())
end_date = st.sidebar.date_input("End Date", df.index.max())

freq = st.sidebar.selectbox("Time Granularity", ["Daily", "Weekly", "Monthly"])
show_forecast = st.sidebar.checkbox("Show Forecast")

filtered_df = df.loc[start_date:end_date]

if freq == "Weekly":
    filtered_df = filtered_df.resample('W').mean()
elif freq == "Monthly":
    filtered_df = filtered_df.resample('M').mean()


 
# 6. KPI CALCULATIONS
 

kpis = {
    "Total Children": int(filtered_df['Total_Load'].iloc[-1]),
    "Avg Net Intake": round(filtered_df['Net_Intake'].mean(), 2),
    "Volatility": round(filtered_df['Total_Load'].std(), 2),
    "Backlog": int(filtered_df['Backlog'].iloc[-1]),
    "Discharge Ratio": round(
        filtered_df['Discharges'].sum() / filtered_df['Transfers'].sum(), 2
    )
}


 
# 7. TITLE
 

st.title("📊 UAC System Capacity Analytics Dashboard")


 
# 8. KPI CARDS
 

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("👶 Total Children", kpis["Total Children"],
            delta=kpis["Total Children"] - filtered_df['Total_Load'].iloc[-2])

col2.metric("📥 Net Intake", kpis["Avg Net Intake"])
col3.metric("📊 Volatility", kpis["Volatility"])
col4.metric("📦 Backlog", kpis["Backlog"])
col5.metric("🏁 Discharge Ratio", kpis["Discharge Ratio"])

st.markdown("---")


 
# 9. SYSTEM STATUS
 

if kpis["Avg Net Intake"] > 0:
    st.error("⚠️ System Under Pressure (More intake than discharge)")
else:
    st.success("✅ System Stable (Discharge ≥ Intake)")


 
# 10. CHARTS
 

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

st.subheader("📉 Growth Rate (%)")
st.line_chart(filtered_df['Growth_Rate'])
st.markdown("---")


 
# 11. DATA VALIDATION
 

st.subheader("⚠️ Data Validation Issues")

invalid_transfer = (filtered_df['Transfers'] > filtered_df['CBP Custody']).sum()
invalid_discharge = (filtered_df['Discharges'] > filtered_df['HHS Care']).sum()

st.write("Invalid Transfers:", int(invalid_transfer))
st.write("Invalid Discharges:", int(invalid_discharge))


 
# 12. HIGH LOAD PERIODS
 

st.subheader("🔥 High Load Periods")

threshold = filtered_df['Total_Load'].mean() + filtered_df['Total_Load'].std()
high_load = filtered_df[filtered_df['Total_Load'] > threshold]

st.dataframe(high_load[['Total_Load']].head(10))


 
# 13. INSIGHTS
 

st.subheader("💡 Key Insights")

if kpis["Avg Net Intake"] < 0:
    st.success("System is efficiently handling discharges (Discharge > Intake).")

if kpis["Backlog"] > 0:
    st.warning("Backlog accumulation indicates potential capacity stress.")

if kpis["Volatility"] > filtered_df['Total_Load'].mean() * 0.3:
    st.warning("High volatility detected — system instability risk.")

if len(high_load) > 0:
    st.error("Multiple high-load periods detected — monitor capacity closely.")


 
# 14. CORRELATION
 

st.subheader("🔗 Correlation Insight")

corr = filtered_df['Net_Intake'].corr(filtered_df['Total_Load'])
st.write("Correlation between Load & Intake:", round(corr, 2))


 
# 15. FORECAST (WITH CONFIDENCE INTERVAL)
 

if show_forecast:
    try:
        model = ARIMA(filtered_df['Total_Load'], order=(5,1,0))
        model_fit = model.fit()

        forecast_obj = model_fit.get_forecast(steps=30)
        forecast = forecast_obj.predicted_mean
        conf_int = forecast_obj.conf_int()

        future_dates = pd.date_range(
            filtered_df.index[-1] + pd.Timedelta(days=1),
            periods=30
        )

        st.subheader("🔮 30-Day Forecast")

        fig, ax = plt.subplots()

        ax.plot(filtered_df.index, filtered_df['Total_Load'], label="Actual")
        ax.plot(future_dates, forecast, label="Forecast")

        ax.fill_between(
            future_dates,
            conf_int.iloc[:, 0],
            conf_int.iloc[:, 1],
            alpha=0.2
        )

        ax.legend()

        st.pyplot(fig)

    except:
        st.warning("Forecasting failed due to data limitations.")


 
# 16. DOWNLOAD DATA
 

st.subheader("📥 Download Data Report")

csv = filtered_df.to_csv().encode('utf-8')

st.download_button(
    label="Download CSV",
    data=csv,
    file_name="uac_dashboard_data.csv",
    mime="text/csv",
)
