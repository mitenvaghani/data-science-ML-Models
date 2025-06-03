import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(page_title="📈 Enrollment Predictor", layout="centered")

# Custom CSS styling
st.markdown("""
    <style>
        .main {
            background-color: #f0f2f6;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stTextInput>div>div>input {
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# Load dataset
df = pd.read_csv("caschool.csv")

# Features and target
features = ['teachers', 'calw_pct', 'meal_pct', 'computer', 'comp_stu',
            'expn_stu', 'str', 'avginc', 'el_pct', 'testscr']
X = df[features]
y = df['enrl_tot']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LinearRegression()
model.fit(X_scaled, y)

# Title
st.title("🎓 California School Enrollment Predictor")
st.markdown("Use the model below to predict total **student enrollment** based on school characteristics.")

# Input section
st.subheader("📋 Enter School Features")

cols = st.columns(2)
input_data = {}
for i, feature in enumerate(features):
    default_val = float(df[feature].mean())
    with cols[i % 2]:
        input_data[feature] = st.number_input(
            f"{feature.replace('_', ' ').capitalize()}",
            value=round(default_val, 2)
        )

# Predict
if st.button("🚀 Predict Enrollment"):
    input_array = np.array([list(input_data.values())])
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)
    st.success(f"📊 Predicted Enrollment: **{prediction[0]:,.0f} students**")
