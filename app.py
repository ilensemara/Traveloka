
import streamlit as st
import pandas as pd
import joblib
import re # Untuk fungsi preprocess_teks

# --- 1. Load Model dan TF-IDF Vectorizer yang Sudah Disimpan ---
model = joblib.load('model.pkl') # Ganti 'model.pkl' dengan nama file model Anda
vectorizer = joblib.load('tfidf.pkl') # Ganti 'tfidf.pkl' dengan nama file vectorizer Anda

# --- 2. Fungsi Preprocessing (sesuai dengan yang digunakan saat pelatihan model) ---
stopwords_indonesia = set([
    "yang", "dan", "di", "dari", "ke", "ini", "itu", "dengan", "untuk",
    "ada", "adalah", "bisa", "aja", "sudah", "saya", "aplikasi", "app",
    "kalo", "kalau", "tapi", "akan", "telah", "oleh", "atau", "pada"
])

def preprocess_teks(ulasan):
    ulasan_bersih = ulasan.lower()
    ulasan_bersih = re.sub(r'[^a-z\s]', ' ', ulasan_bersih)
    tokens = ulasan_bersih.split()
    tokens_filtered = [kata for kata in tokens if kata not in stopwords_indonesia]
    return " ".join(tokens_filtered)

# --- 3. Streamlit App Interface ---
st.title("Analisis Sentimen Ulasan Traveloka")
st.write("Masukkan ulasan untuk memprediksi sentimennya.")

user_input = st.text_area("Ulasan:", "")

if st.button("Prediksi Sentimen"):
    if user_input:
        # Preprocessing input pengguna
        clean_text = preprocess_teks(user_input)
        
        # Vectorisasi
        vec_text = vectorizer.transform([clean_text])
        
        # Prediksi
        prediction = model.predict(vec_text)
        prediction_proba = model.predict_proba(vec_text)
        
        st.subheader("Hasil Prediksi:")
        st.write(f"Sentimen: **{prediction[0]}**")
        
        # Menampilkan probabilitas
        proba_df = pd.DataFrame(prediction_proba, columns=model.classes_)
        st.write("Probabilitas Sentimen:")
        st.dataframe(proba_df)
    else:
        st.warning("Mohon masukkan ulasan untuk diproses.")
