import pandas as pd
import numpy as np
import pickle
import streamlit as st
import joblib

# Load model and columns
model = joblib.load("pollution_model.pkl")
model_cols = joblib.load("model_columns.pkl")

# Page configuration
st.set_page_config(page_title="Water Pollution Predictor", layout="centered")

# Custom CSS styling
st.markdown("""
    <style>
        .main {
            background-color: #f4f8fb;
        }
        h1, h3 {
            color: #003366;
        }
        .stButton button {
            background-color: #0066cc;
            color: white;
            border-radius: 10px;
            padding: 0.5em 2em;
        }
        .stButton button:hover {
            background-color: #004999;
        }
        .prediction-box {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0px 0px 10px #c6d4e1;
        }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("💧 Water Pollutants Prediction App")
st.markdown("### 🌍 Predict pollution levels based on **Year** and **Station ID**")
st.write("Use this tool to estimate levels of major water pollutants across different monitoring stations.")

# Inputs
col1, col2 = st.columns(2)
with col1:
    year_input = st.number_input("📅 Enter Year", min_value=2000, max_value=2100, value=2022)

with col2:
    station_id = st.text_input("🏷️ Enter Station ID (1-22)", value='1')

# Prediction
if st.button("🔍 Predict Pollutants"):
    if not station_id.strip():
        st.warning('⚠️ Please enter a valid Station ID.')
    else:
        # Prepare and encode input
        input_df = pd.DataFrame({'year': [year_input], 'id':[station_id]})
        input_encoded = pd.get_dummies(input_df, columns=['id'])

        # Align with model columns
        for col in model_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[model_cols]

        # Make prediction
        predicted_pollutants = model.predict(input_encoded)[0]
        pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']

        # Display results
        st.markdown("### 🔬 Predicted Pollutant Levels")
        with st.container():
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            for p, val in zip(pollutants, predicted_pollutants):
                st.markdown(f"<b>{p}:</b> {val:.2f}", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
