#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd

# ---------- Load Saved Model & Scaler ----------
@st.cache_resource
def load_model():
    model = joblib.load("trained_model.pkl")
    scaler = joblib.load("scaler.pkl")
    with open("feature_names.json", "r") as f:
        feature_names = json.load(f)
    return model, scaler, feature_names

model, scaler, feature_names = load_model()

st.title("🚲 Bike Rental Prediction App")
st.write("Enter the details below to predict bike rental count.")

# ---------- User Inputs ----------
# ---------- User Inputs ----------
season = st.selectbox("Season", [1, 2, 3, 4])
yr = st.selectbox("Year", [2023 , 2024])  
mnth = st.selectbox("Month", list(range(1, 13)))
hr = st.slider("Hour of Day", 0, 23, 12)
holiday = st.selectbox("Holiday (1 = Yes, 0 = No)", [0, 1])
weekday = st.selectbox("Weekday (0=Sunday ... 6=Saturday)", list(range(0, 7)))
workingday = st.selectbox("Working Day (1=Yes, 0=No)", [0, 1])
weathersit = st.selectbox("Weather Situation", [1, 2, 3, 4])
temp = st.number_input("Temperature (Normalized: 0 to 1)", min_value=0.0, max_value=1.0, value=0.5)
atemp = st.number_input("Feels Like Temperature (Normalized: 0 to 1)", min_value=0.0, max_value=1.0, value=0.5)
hum = st.number_input("Humidity (Normalized: 0 to 1)", min_value=0.0, max_value=1.0, value=0.5)
windspeed = st.number_input("Windspeed (Normalized: 0 to 1)", min_value=0.0, max_value=1.0, value=0.2)
registered = st.number_input("Registered Users", min_value=0, max_value=1000, value=50)


# ---------- Preprocess Inputs ----------
if st.button("Predict"):
    # Convert to dataframe
    input_data = pd.DataFrame([[season ,yr, mnth, hr, holiday, weekday, workingday, weathersit, temp, atemp, hum, windspeed , registered]],
                              columns=['season','yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 
                                       'temp', 'atemp', 'hum', 'windspeed' , 'registered'])
    
    # Scale required columns
    cols_to_scale = ['temp', 'atemp', 'hum', 'windspeed']
    input_data[cols_to_scale] = scaler.transform(input_data[cols_to_scale])

    # Ensure column order matches training
    input_data = input_data.reindex(columns=feature_names, fill_value=0)

    # Predict
    prediction = model.predict(input_data)[0]
    st.success(f"🚲 Predicted Bike Count: **{int(prediction)}**")


# In[ ]:




