# app.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import streamlit as st

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv("iris_train.csv")
    X = df.drop(columns=["Species"])
    y = LabelEncoder().fit_transform(df["Species"])
    model = make_pipeline(PolynomialFeatures(degree=2), StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X, y)
    return model

model = load_data()

# UI
st.title("🌸 Iris Species Prediction with Polynomial Regression")
st.markdown("Enter the measurements of the flower to predict the species:")

# Inputs
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.5)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.0)

# Predict
if st.button("Predict Species"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]
    species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
    st.success(f"🔍 Predicted Species: **{species_map[prediction]}**")
