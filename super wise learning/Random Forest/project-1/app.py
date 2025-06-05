import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Page config
st.set_page_config(
    page_title="Car Price Predictor",
    layout="centered",
    page_icon="🚗"
)

# Custom CSS
st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
        }
        .main {
            background-color: #f0f2f6;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0px 0px 15px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333366;
        }
        .stButton>button {
            background-color: #3366cc;
            color: white;
            border-radius: 5px;
            padding: 0.5em 1em;
            border: none;
        }
        .stButton>button:hover {
            background-color: #254b9c;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main">', unsafe_allow_html=True)
st.title("🚗 Car Price Prediction App")
st.write("Use the sliders in the sidebar to input car features and get a price prediction using a Random Forest model.")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Cars93.csv")
    return df

df = load_data()

# Show dataset
with st.expander("📊 View Dataset"):
    st.dataframe(df)

# Preprocessing
df = df.dropna()
X = df[['MPG.city', 'MPG.highway', 'EngineSize', 'Horsepower', 'RPM', 'Length', 'Width', 'Weight']]
y = df['Price']

# Sidebar inputs
st.sidebar.title("Customize Car Features")
def user_input_features():
    mpg_city = st.sidebar.slider('MPG (City)', int(X['MPG.city'].min()), int(X['MPG.city'].max()), int(X['MPG.city'].mean()))
    mpg_highway = st.sidebar.slider('MPG (Highway)', int(X['MPG.highway'].min()), int(X['MPG.highway'].max()), int(X['MPG.highway'].mean()))
    engine_size = st.sidebar.slider('Engine Size (L)', float(X['EngineSize'].min()), float(X['EngineSize'].max()), float(X['EngineSize'].mean()))
    horsepower = st.sidebar.slider('Horsepower', int(X['Horsepower'].min()), int(X['Horsepower'].max()), int(X['Horsepower'].mean()))
    rpm = st.sidebar.slider('RPM', int(X['RPM'].min()), int(X['RPM'].max()), int(X['RPM'].mean()))
    length = st.sidebar.slider('Length (inches)', int(X['Length'].min()), int(X['Length'].max()), int(X['Length'].mean()))
    width = st.sidebar.slider('Width (inches)', int(X['Width'].min()), int(X['Width'].max()), int(X['Width'].mean()))
    weight = st.sidebar.slider('Weight (lbs)', int(X['Weight'].min()), int(X['Weight'].max()), int(X['Weight'].mean()))

    data = {
        'MPG.city': mpg_city,
        'MPG.highway': mpg_highway,
        'EngineSize': engine_size,
        'Horsepower': horsepower,
        'RPM': rpm,
        'Length': length,
        'Width': width,
        'Weight': weight
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predictions
prediction = model.predict(input_df)
y_pred = model.predict(X_test)

# Display prediction
st.subheader("💰 Predicted Price")
st.success(f"Estimated Price: **${prediction[0]:,.2f}**")

#  Model performance

# st.subheader("📈 Model Evaluation on Test Data")
# col1, col2 = st.columns(2)
# col1.metric("R² Score", f"{r2_score(y_test, y_pred):.2f}")
# col2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

# st.markdown('</div>', unsafe_allow_html=True)
