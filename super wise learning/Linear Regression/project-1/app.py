import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.title("Linear Regression Predictor")

st.write("""
This app allows you to upload a CSV dataset, select independent feature(s) and a target variable, 
train a linear regression model, and see prediction results and metrics.
""")

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Dataset")
    st.write(data.head())

    # Show columns options
    all_columns = list(data.columns)
    target_column = st.selectbox("Select the target (dependent) variable", options=all_columns)
    feature_columns = st.multiselect("Select the feature (independent) variable(s)", options=[col for col in all_columns if col != target_column])

    if target_column and feature_columns:
        # Handle missing values by dropping for simplicity
        df = data[[target_column] + feature_columns].dropna()

        X = df[feature_columns]
        y = df[target_column]

        test_size = st.slider("Test set size (percentage)", min_value=10, max_value=50, value=20, step=5)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.subheader("Model Performance")
        st.write(f"Mean Squared Error (MSE): {mse:.4f}")
        st.write(f"R-squared (R2 ): {r2:.4f}")

        st.subheader("Predictions vs Actual")
        results_df = pd.DataFrame({
            "Actual": y_test,
            "Predicted": y_pred
        }).reset_index(drop=True)
        st.write(results_df)

        st.subheader("Make Your Own Prediction")
        user_input = {}
        for feature in feature_columns:
            val = st.number_input(f"Input value for {feature}", value=float(df[feature].mean()))
            user_input[feature] = val
        
        if st.button("Predict"):
            input_df = pd.DataFrame([user_input])
            prediction = model.predict(input_df)[0]
            st.write(f"Predicted {target_column}: {prediction:.4f}")

else:
    st.info("Awaiting CSV file to be uploaded.")

