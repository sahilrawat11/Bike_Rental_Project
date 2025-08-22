#!/usr/bin/env python
# coding: utf-8

# In[9]:


import streamlit as st
import pandas as pd
import joblib  

# ----------------------------
# Load saved objects
# ----------------------------
model = joblib.load("trained_model.pkl")
scaler = joblib.load("scaler.pkl")

# ----------------------------
# Mapping dictionaries
# ----------------------------
season_map = {"Spring": 1, "Summer": 2, "Fall": 3, "Winter": 4}
month_map = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}
weekday_map = {
    "Sunday": 0, "Monday": 1, "Tuesday": 2,
    "Wednesday": 3, "Thursday": 4, "Friday": 5, "Saturday": 6
}
weather_map = {
    "Clear": 1,
    "Mist + Cloudy": 2,
    "Light Snow/Rain": 3,
    "Heavy Rain/Snow": 4
}

# ----------------------------
# Title
# ----------------------------
st.set_page_config(page_title="Bike Rental Prediction", page_icon="🚲")
st.title("🚲 Bike Rental Demand Prediction App")
st.markdown("Choose inputs below and the model will predict expected **bike demand**.")

# ----------------------------
# User Inputs
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    season = st.selectbox("Season", list(season_map.keys()))
    month = st.selectbox("Month", list(month_map.keys()))
    hour = st.slider("Hour of the Day", 0, 23, 12)

with col2:
    weekday = st.selectbox("Weekday", list(weekday_map.keys()))
    weather = st.selectbox("Weather", list(weather_map.keys()))
    temp = st.slider("Temperature (normalized 0–1)", 0.0, 1.0, 0.5)
    hum = st.slider("Humidity (normalized 0–1)", 0.0, 1.0, 0.5)
    windspeed = st.slider("Windspeed (normalized 0–1)", 0.0, 1.0, 0.2)

# ----------------------------
# Convert categorical to numeric (Directly as training)
# ----------------------------
input_data = pd.DataFrame([{
    "season": season_map[season],
    "mnth": month_map[month],
    "hr": hour,
    "weekday": weekday_map[weekday],
    "weathersit": weather_map[weather],
    "temp": temp,
    "hum": hum,
    "windspeed": windspeed
}])

# ----------------------------
# Scale input
# ----------------------------
input_scaled = scaler.transform(input_data)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict 🚲"):
    prediction = model.predict(input_scaled)[0]
    st.success(f"🔮 Predicted Bike Demand: **{int(prediction)} bikes**")


# In[ ]:




