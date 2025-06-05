import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="DBSCAN Clustering", layout="wide")

st.title("📊 DBSCAN Clustering on Daily Min Temperatures")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)

    # Preprocessing
    df['Date'] = pd.to_datetime(df['Date'])
    df['DayOfYear'] = df['Date'].dt.dayofyear
    X = df[['Temp']].values

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # User input for DBSCAN parameters
    eps = st.slider("Epsilon (eps)", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
    min_samples = st.slider("Min Samples", min_value=1, max_value=20, value=5, step=1)

    # Apply DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = db.fit_predict(X_scaled)
    df['Cluster'] = clusters

    # Show dataframe
    st.subheader("📄 Sample Data with Cluster Labels")
    st.write(df.head())

    # Plotting
    st.subheader("📈 Temperature Clustering Plot")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.scatterplot(data=df, x='Date', y='Temp', hue='Cluster', palette='tab10', s=20, ax=ax)
    plt.title("DBSCAN Clustering on Daily Minimum Temperatures")
    plt.xlabel("Date")
    plt.ylabel("Temperature")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Show cluster counts
    st.subheader("📊 Cluster Counts")
    st.write(df['Cluster'].value_counts())

else:
    st.info("Please upload a CSV file to begin.")
