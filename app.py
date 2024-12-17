import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import sklearn
import warnings
warnings.filterwarnings('ignore')

# Load the trained model and scaler
try:
    with open("Demand_Forecast.pkl", "rb") as f:
        xgb_hyp = pickle.load(f)
    with open("scale.pkl", "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("Error: 'DF.pkl' or 'scaler.pkl' not found.")
    st.stop()

# Custom CSS for styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f4f7;
    }
    .title h1 {
        font-size: 48px;
        font-weight: bold;
        color: #009688;
        text-align: center;
        margin-bottom: 40px;
    }
    .stButton button {
        background-color: #009688;
        color: white;
        font-size: 18px;
        padding: 10px 20px;
        border-radius: 8px;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.2);
        transition: background-color 0.3s ease;
    }
    .stButton button:hover {
        background-color: #00796b;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title
st.markdown("<div class='title'><h1>Demand Forecasting</h1></div>", unsafe_allow_html=True)

# Load input data (if needed for display or analysis)
try:
    df = pd.read_csv("retail_store_inventory_with_nulls.csv")
    st.dataframe(df.head())
except FileNotFoundError:
    st.warning("retail_store_inventory_with_nulls.csv not found.")

# Load input data
st.header("Enter the input details to predict demand:")

Inventory_Level = st.number_input("Inventory Level:", min_value=0.0, max_value=10000.0, step=0.1)
Units_Sold = st.number_input("Units Sold:", min_value=0.0, max_value=1000.0, step=0.1)
Units_Ordered = st.number_input("Units Ordered:", min_value=0.0, max_value=1000.0, step=0.1)
Price = st.number_input("Price:", min_value=0.0, max_value=1000.0, step=0.1)
Discount = st.number_input("Discount:", min_value=0.0, max_value=100.0, step=0.1)
Holiday_Promotion = st.selectbox("Holiday Promotion:", options=["No", "Yes"], index=0)
Competitor_Pricing = st.number_input("Competitor Pricing:", min_value=0.0, max_value=1000.0, step=0.1)
Seasonality = st.selectbox("Seasonality:", options=["Spring", "Summer", "Winter", "Autumn"])
Category = st.selectbox("Category:", options=["Electronics", "Furniture", "Groceries", "Toys"])
Region = st.selectbox("Region:", options=["North", "South", "West", "East"])
Weather_Condition = st.selectbox("Weather Condition:", options=["Rainy", "Snowy", "Sunny", "Cloudy"])

# Map inputs to encoded features
holiday_promotion = 1 if Holiday_Promotion == "Yes" else 0
seasonality_mapping = {"Spring": 0, "Summer": 1, "Winter": 2, "Autumn": 3}
category_mapping = {"Electronics": [1, 0, 0, 0], "Furniture": [0, 1, 0, 0],
                     "Groceries": [0, 0, 1, 0], "Toys": [0, 0, 0, 1]}
region_mapping = {"North": [1, 0, 0, 0], "South": [0, 1, 0, 0],
                  "West": [0, 0, 1, 0], "East": [0, 0, 0, 1]}
weather_mapping = {"Rainy": [1, 0, 0, 0], "Snowy": [0, 1, 0, 0],
                   "Sunny": [0, 0, 1, 0], "Cloudy": [0, 0, 0, 1]}

# Prepare input features
input_features = pd.DataFrame([[
    Inventory_Level, Units_Sold, Units_Ordered, Price, Discount, holiday_promotion,
    Competitor_Pricing, seasonality_mapping[Seasonality],
    *category_mapping[Category], *region_mapping[Region], *weather_mapping[Weather_Condition]
]], columns=[
    "Inventory Level", "Units Sold", "Units Ordered", "Price", "Discount", "Holiday/Promotion",
    "Competitor Pricing", "Seasonality", "Category_electronics", "Category_furniture",
    "Category_groceries", "Category_toys", "Region_north", "Region_south", "Region_west",
    "Region_east", "Weather Condition_rainy", "Weather Condition_snowy",
    "Weather Condition_sunny", "Weather Condition_cloudy"
])

# Debugging: Print input features and shape
st.write("Given Input Data:")
st.dataframe(input_features)

# Ensure input data matches model requirements
try:
    if hasattr(xgb_hyp, 'feature_names_in_'):
        model_features = xgb_hyp.feature_names_in_  # Expected feature names
    else:
        # Handle cases where 'feature_names_in_' is not available (e.g., older XGBoost versions)
        model_features = input_features.columns
except AttributeError:
    st.error("Error: 'xgb_hyp' does not have the attribute 'feature_names_in_'. "
             "Please ensure 'DF.pkl' contains a valid trained XGBoost model object.")
    st.stop()

input_features = input_features.reindex(columns=model_features, fill_value=0)

# Apply scaling to numerical features
numerical_columns = ["Inventory Level", "Units Sold", "Units Ordered", "Price", "Discount",
                     "Holiday/Promotion", "Competitor Pricing", "Seasonality"]
try:
    input_features[numerical_columns] = scaler.transform(input_features[numerical_columns])
except Exception as e:
    st.error(f"Error during scaling: {e}")

# Make prediction
if st.button("Predict Demand"):
    try:
        prediction = round(xgb_hyp.predict(input_features)[0], 2)
        st.success(f"Estimated Demand Value: {prediction} units")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")