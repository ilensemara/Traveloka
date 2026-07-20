
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Analisis Sentimen Traveloka")

# Load the correct dataset
df = pd.read_csv("Traveloka-id application rating and review dataset.csv", sep=";", encoding="latin-1")

# Clean and preprocess the data as done in the notebook
df_clean = df[['Bintang', 'Ulasan']].dropna()
df_clean['Bintang'] = pd.to_numeric(df_clean['Bintang'], errors='coerce')
df_clean = df_clean.dropna(subset=['Bintang'])
df_clean['Bintang'] = df_clean['Bintang'].astype(int)
df_clean = df_clean[df_clean['Bintang'].between(1, 5)]

# Define the sentiment labeling function
def label_sentimen(rating):
    if rating in [1, 2]: return 'Negatif'
    elif rating == 3: return 'Netral'
    else: return 'Positif'

# Apply sentiment labeling
df_clean['Sentimen'] = df_clean['Bintang'].apply(label_sentimen)

st.subheader("Distribusi Sentimen")
fig, ax = plt.subplots()
df_clean["Sentimen"].value_counts().plot(kind="bar", ax=ax, color=['#4CAF50', '#FFC107', '#F44336'])
ax.set_title('Distribusi Sentimen Ulasan Traveloka')
ax.set_xlabel('Sentimen')
ax.set_ylabel('Jumlah Ulasan')
plt.xticks(rotation=45)
st.pyplot(fig)
