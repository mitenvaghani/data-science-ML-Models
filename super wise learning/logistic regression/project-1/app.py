import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv("Vehicle.csv")

# Prepare features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
model.fit(X_train, y_train)

st.title("Vehicle Class Prediction App")

st.markdown("Enter the following 18 features to predict the vehicle class:")

input_data = []
for col in X.columns:
    val = st.number_input(f"{col}", value=0)
    input_data.append(val)

if st.button("Predict Class"):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    predicted_label = label_encoder.inverse_transform(prediction)
    st.success(f"Predicted Class: {predicted_label[0]}")
