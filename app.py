import pandas as pd
import plotly.express as px
import streamlit as st

# Load data
df = pd.read_csv('train.csv', encoding='latin1')
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
df['Month'] = df['Order Date'].dt.to_period('M')
df['Year'] = df['Order Date'].dt.year

# Dashboard title
st.title("Sales Dashboard")
st.markdown("Superstore Sales Analysis 2015-2018")

# Sidebar filters
st.sidebar.header("Filters")
year = st.sidebar.multiselect("Select Year", sorted(df['Year'].unique()), default=sorted(df['Year'].unique()))
category = st.sidebar.multiselect("Select Category", df['Category'].unique(), default=df['Category'].unique())
region = st.sidebar.multiselect("Select Region", df['Region'].unique(), default=df['Region'].unique())

# Filter data
filtered = df[(df['Year'].isin(year)) & (df['Category'].isin(category)) & (df['Region'].isin(region))]

# KPI metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Sales", f"${filtered['Sales'].sum():,.0f}")
col2.metric("Total Orders", f"{filtered['Order ID'].nunique():,}")
col3.metric("Avg Order Value", f"${filtered['Sales'].mean():,.0f}")

# Charts
fig1 = px.bar(filtered.groupby('Category')['Sales'].sum().reset_index(), x='Category', y='Sales', color='Category', title='Sales by Category')
st.plotly_chart(fig1)

fig2 = px.pie(filtered.groupby('Region')['Sales'].sum().reset_index(), values='Sales', names='Region', title='Sales by Region')
st.plotly_chart(fig2)

fig3 = px.bar(filtered.groupby('Sub-Category')['Sales'].sum().reset_index().sort_values('Sales', ascending=True).tail(10), x='Sales', y='Sub-Category', orientation='h', title='Top 10 Sub-Categories')
st.plotly_chart(fig3)

monthly = filtered.groupby('Month')['Sales'].sum().reset_index()
monthly['Month'] = monthly['Month'].astype(str)
fig4 = px.line(monthly, x='Month', y='Sales', title='Monthly Sales Trend')
st.plotly_chart(fig4)
# Forecasting section
st.markdown("---")
st.header("Sales Forecast")

from prophet import Prophet

# Prepare data for Prophet
monthly_forecast = df.groupby('Month')['Sales'].sum().reset_index()
monthly_forecast['Month'] = monthly_forecast['Month'].astype(str)
monthly_forecast.columns = ['ds', 'y']
monthly_forecast['ds'] = pd.to_datetime(monthly_forecast['ds'])

# Train model
model = Prophet()
model.fit(monthly_forecast)

# Predict next 6 months
future = model.make_future_dataframe(periods=6, freq='ME')
forecast = model.predict(future)

# Plot
fig5 = px.line(forecast, x='ds', y='yhat', title='Sales Forecast - Next 6 Months')
fig5.add_scatter(x=monthly_forecast['ds'], y=monthly_forecast['y'], mode='lines', name='Actual Sales')
st.plotly_chart(fig5)

st.markdown("**Forecast insight:** Based on historical trends, sales are projected to continue growing into 2019.")

