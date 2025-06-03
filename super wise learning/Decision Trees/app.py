# Import necessary libraries
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load the trained model
model = joblib.load('gdp_decision_tree_model.pkl')

# Function to predict GDP category
def predict_gdp_category(annual_gdp_growth, last_inflation):
    features = np.array([[annual_gdp_growth, last_inflation]])
    prediction = model.predict(features)
    return prediction[0]

# Streamlit application
st.title("GDP Category Prediction")

# User input
annual_gdp_growth = st.number_input("Enter Annual GDP Growth (%)", min_value=-10.0, max_value=20.0, value=0.0)
last_inflation = st.number_input("Enter Last Inflation Rate (%)", min_value=0.0, max_value=100.0, value=0.0)

# Button to make prediction
if st.button("Predict"):
    gdp_category = predict_gdp_category(annual_gdp_growth, last_inflation)
    categories = {0: "Low", 1: "Medium", 2: "High"}
    st.success(f"The predicted GDP category is: {categories[gdp_category]}")
