import pandas as pd
import streamlit as st
import joblib
import numpy as np

st.title('Sales Forecasting')

st.write('We forecast sales')

data = pd.read_csv('/Users/alyssa/Desktop/ftw-webapp-deployment/data/advertising_regression.csv')

data

st.sidebar.subheader('Advertising Costs')
TV = st.sidebar.slider('TV Advertising Cost', 0, 300, 150)

radio = st.sidebar.slider('Radio Advertising Cost', 0, 300, 150)

newspaper = st.sidebar.slider('Newspaper Advertising Cost', 0, 300, 150)

hist_values = np.histogram(data.radio, bins=300, range=(0,300))[0]
st.bar_chart(hist_values)

hist_values = np.histogram(data.newspaper, bins=300, range=(0,300))[0]
st.bar_chart(hist_values)

hist_values = np.histogram(data.TV, bins=300, range=(0,300))[0]
st.bar_chart(hist_values)

saved_model = joblib.load('advertising_model.sav')

predicted_sales = saved_model.predict([[TV, radio, newspaper]])

st.write(f'Predcited sales is {predicted_sales} dollars')