import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# App title
st.title("K-Means Clustering on Weekly Dataset")

# File uploader
uploaded_file = st.file_uploader("Upload Weekly.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data")
    st.write(df.head())

    # Clean and scale data
    df_clean = df.drop(columns=["Direction", "Year"], errors='ignore')

    if df_clean.isnull().sum().sum() > 0:
        df_clean.fillna(df_clean.mean(), inplace=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)

    # K-Means clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    df["Cluster"] = clusters

    # Show clustering results
    st.subheader("Clustered Data")
    st.write(df[["Lag1", "Lag2", "Lag3", "Lag4", "Lag5", "Volume", "Today", "Cluster"]].head())

    # Optional: Compare with actual Direction
    if "Direction" in df.columns:
        df["Direction_Code"] = df["Direction"].map({"Up": 1, "Down": 0})
        st.subheader("Cluster vs Actual Direction")
        st.write(pd.crosstab(df["Cluster"], df["Direction_Code"]))

    # Plotting
    st.subheader("K-Means Clustering (Lag1 vs Lag2)")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="Lag1", y="Lag2", hue="Cluster", palette="Set2", ax=ax)
    st.pyplot(fig)
