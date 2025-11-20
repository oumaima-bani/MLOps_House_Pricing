import streamlit as st
import requests
import os

# Get API URL from environment variable (default to localhost if not set)
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000") + "/predict"

st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="centered")
st.title("🏠 House Price Prediction App")
st.write("Enter the housing features to predict the median house price (USD).")

# Sidebar inputs
MedInc = st.sidebar.number_input("Median Income (MedInc)", min_value=0.0, max_value=20.0, value=8.3)
HouseAge = st.sidebar.number_input("House Age (HouseAge)", min_value=0.0, max_value=100.0, value=41.0)
AveRooms = st.sidebar.number_input("Average Rooms (AveRooms)", min_value=0.0, max_value=20.0, value=6.98)
AveBedrms = st.sidebar.number_input("Average Bedrooms (AveBedrms)", min_value=0.0, max_value=10.0, value=1.02)
Population = st.sidebar.number_input("Population", min_value=0.0, max_value=5000.0, value=322.0)
AveOccup = st.sidebar.number_input("Average Occupancy (AveOccup)", min_value=0.0, max_value=10.0, value=2.55)
Latitude = st.sidebar.number_input("Latitude", min_value=0.0, max_value=90.0, value=37.88)

if st.button("Predict Price"):
    payload = {
        "MedInc": MedInc,
        "HouseAge": HouseAge,
        "AveRooms": AveRooms,
        "AveBedrms": AveBedrms,
        "Population": Population,
        "AveOccup": AveOccup,
        "Latitude": Latitude
    }
    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            price = response.json()["prediction_usd"]
            st.success(f"Predicted Median House Price: ${price:,.2f}")
        else:
            st.error(f"API Error: {response.json()}")
    except Exception as e:
        st.error(f"Could not connect to API: {e}")
