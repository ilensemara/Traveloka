import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

st.title("Analisis Sentimen Traveloka")

# Load dataset
df = pd.read_csv(
    "Traveloka-id application rating and review dataset.csv",
    sep=";",
    encoding="latin-1"
)

# Label sentimen
def label_sentimen(rating):
    if rating in [1, 2]:
        return "Negatif"
    elif rating == 3:
        return "Netral"
    else:
        return "Positif"

df["Sentimen"] = df["Bintang"].apply(label_sentimen)

# Preprocessing
stopwords_indonesia = {
    "yang","dan","di","dari","ke","ini","itu","dengan",
    "untuk","ada","adalah","bisa","aja","sudah","saya"
}

def preprocess(teks):
    teks = str(teks).lower()
    teks = re.sub(r'[^a-z\s]', ' ', teks)
    tokens = [
        kata for kata in teks.split()
        if kata not in stopwords_indonesia
    ]
    return " ".join(tokens)

df["Ulasan_Clean"] = df["Ulasan"].apply(preprocess)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["Ulasan_Clean"])
y = df["Sentimen"]

# Training model
model = MultinomialNB()
model.fit(X, y)

# Input user
st.subheader("Cek Sentimen Ulasan")

review = st.text_area(
    "Masukkan ulasan:"
)

if st.button("Prediksi"):
    clean_review = preprocess(review)
    review_vec = vectorizer.transform([clean_review])

    prediksi = model.predict(review_vec)[0]

    st.success(f"Sentimen: {prediksi}")
