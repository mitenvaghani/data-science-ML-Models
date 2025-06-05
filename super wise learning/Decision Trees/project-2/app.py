import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np

st.set_page_config(page_title="Debit Card Value Predictor", layout="centered")
st.title("🔍 Debit Card Usage Predictor using Decision Tree")

# Load dataset
df = pd.read_csv("debitcards.csv")
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['lag_1'] = df['value'].shift(1)
df = df.dropna()

# Feature matrix and target
X = df[['year', 'month', 'lag_1']]
y = df['value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train model
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Display data
with st.expander("📊 View Dataset"):
    st.dataframe(df[['date', 'value']].tail(10))


# Prediction input
st.markdown("### 🔮 Predict Future Value")
year = st.number_input("Year", min_value=2000, max_value=2050, value=2025)
month = st.selectbox("Month", list(range(1, 13)))
lag_value = st.number_input("Previous Month's Value", value=20000)

if st.button("Predict"):
    input_data = pd.DataFrame([[year, month, lag_value]], columns=['year', 'month', 'lag_1'])
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Value: {int(prediction)}")
