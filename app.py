import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler

# ---- Load Model & Scaler ----
MODEL_PATH = "trained_model.sav"
SCALER_PATH = "scaler.sav"

# Load Trained Model
try:
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("⚠️ Model file 'trained_model.sav' not found. Please check the file path.")
    st.stop()

# Load Scaler
try:
    with open(SCALER_PATH, 'rb') as file:
        scaler = pickle.load(file)
except FileNotFoundError:
    st.warning("⚠️ Scaler file 'scaler.sav' not found. Predictions might be inaccurate.")
    scaler = None

# ---- Streamlit UI ----
st.set_page_config(page_title="🏡 NestFinder - Bangalore House Price Prediction", page_icon="🏡")
st.title('🏡 NestFinder - Bangalore House Price Prediction 🏠')

st.markdown("""
    <style>
    .title { font-size: 36px; color: #2e8b57; font-weight: bold; }
    .subheader { font-size: 24px; color: #ff6347; font-weight: bold; }
    .description { font-size: 18px; color: #556b2f; }
    .stButton>button { background-color: #4CAF50; color: white; border-radius: 8px; padding: 12px 28px; font-size: 18px; }
    .stTextInput input { padding: 10px; font-size: 16px; border-radius: 8px; border: 2px solid #4CAF50; }
    .stTextInput input:focus { border-color: #ff6347; }
    </style>
    """, unsafe_allow_html=True)

# ---- Prediction Section ----
st.subheader("💰 House Price Prediction")

# Input fields
total_sqft = st.number_input('Total Sqft', min_value=0.0, value=1000.0, step=1.0)
bhk = st.number_input('BHK', min_value=1, value=2, step=1)
bath = st.number_input('Bathrooms', min_value=1, value=1, step=1)

# Location options
location_options = ['Whitefield', 'Yelahanka', 'Uttarahalli', 'Raja Rajeshwari Nagar', 'Electronic City', 'Kengeri',
                    'Devarachikkanahalli', 'other']  # Update with real dataset locations
location = st.selectbox('Location', location_options)

if st.button('🔮 Predict Price'):
    input_data = pd.DataFrame({
        'total_sqft': [total_sqft],
        'bhk': [bhk],
        'bath': [bath],
        'location': [location]
    })

    # One-hot encode location
    dummies = pd.get_dummies(input_data['location'])
    input_data = pd.concat([input_data, dummies.drop(columns=['other'], errors='ignore')], axis=1)
    input_data.drop(columns=['location'], inplace=True)

    # Ensure input data has the same features as the trained model
    for col in model.feature_names_in_:
        if col not in input_data.columns:
            input_data[col] = 0  # Add missing columns

    input_data = input_data[model.feature_names_in_]  # Ensure correct column order

    # Scale input if scaler exists
    if scaler:
        input_data = scaler.transform(input_data)

    try:
        prediction = model.predict(input_data)[0]

        try:
            # Make prediction
            prediction = model.predict(input_data)[0]  # Extract single value from array

            # Display price in Crores or Lakhs
            if prediction >= 100:  # If price is above or equal to 1 Crore
                price_in_crores = prediction / 100
                st.success(f'💵 Predicted Price: ₹{price_in_crores:,.2f} Crores')
            else:  # If price is below 1 Crore
                price_in_lakhs = prediction
                st.success(f'💵 Predicted Price: ₹{price_in_lakhs:,.2f} Lakhs')

        except Exception as e:
            st.error(f"Prediction error: {e}")

    except Exception as e:
        st.error(f"Prediction error: {e}")

# ---- Sidebar & About ----
with st.sidebar:
    st.title("📜 About NestFinder")
    st.write(
        "Welcome to NestFinder! 🏡 Your go-to platform for predicting house prices in Bangalore. We use AI to help you make the best decisions for buying or selling a home.")
    st.markdown("Made with ❤️ for Humanity")

# ---- Expander for Model Details ----
with st.expander("🔍 Model Details"):
    st.write(
        "The model is trained on house prices in Bangalore, considering factors such as total square footage, number of rooms, and more. It provides price predictions based on input features.")

# ---- Custom CSS Enhancements ----
st.markdown("""
    <style>
    .main .block-container { max-width: 900px; }
    .stButton>button { background-color: #4CAF50; color: white; border-radius: 10px; font-size: 18px; }
    .stTextInput>div>div>input { font-size: 18px; }
    </style>
    """, unsafe_allow_html=True)