app_py_content = """import streamlit as st
import joblib
import re
import pandas as pd

# Load models and vectorizer
@st.cache_resource
def load_resources():
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    nb_model = joblib.load('models/multinomial_nb_model.pkl')
    svm_model = joblib.load('models/svm_model.pkl')
    return vectorizer, nb_model, svm_model

vectorizer, nb_model, svm_model = load_resources()

# Preprocessing function
stopwords_indonesia = set([
    "yang", "dan", "di", "dari", "ke", "ini", "itu", "dengan", "untuk",
    "ada", "adalah", "bisa", "aja", "sudah", "saya", "aplikasi", "app",
    "kalo", "kalau", "tapi", "akan", "telah", "oleh", "atau", "pada"
])

def preprocess_teks(ulasan):
    ulasan_bersih = ulasan.lower()
    ulasan_bersih = re.sub(r'[^a-z\\s]', ' ', ulasan_bersih)
    tokens = ulasan_bersih.split()
    tokens_filtered = [kata for kata in tokens if kata not in stopwords_indonesia]
    return " ".join(tokens_filtered)

# Streamlit app layout
st.title('Aplikasi Analisis Sentimen Ulasan Traveloka')
st.write('Masukkan ulasan Anda untuk mengetahui sentimennya (Negatif, Netral, Positif).')

user_input = st.text_area('Masukkan ulasan di sini:', '')

if st.button('Analisis Sentimen'):
    if user_input:
        # Preprocess input
        cleaned_input = preprocess_teks(user_input)

        # Transform input using TF-IDF Vectorizer
        input_tfidf = vectorizer.transform([cleaned_input])

        # Predict sentiment
        hasil_sentimen = svm_model.predict(input_tfidf)[0]

        st.write('---')
        
        # Output Tunggal Langsung
        st.subheader(f'Hasil Analisis Sentimen: {hasil_sentimen}')

        # 1 Keterangan Tunggal
        if str(hasil_sentimen).lower() == 'negatif':
            st.error('📌 **Keterangan:** Ulasan ini berisi keluhan, kekecewaan, atau kendala layanan yang dialami oleh pengguna.')
        elif str(hasil_sentimen).lower() == 'positif':
            st.success('📌 **Keterangan:** Ulasan ini berisi pujian, kepuasan, atau apresiasi terhadap layanan yang diberikan.')
        else:
            st.info('📌 **Keterangan:** Ulasan ini bersifat netral, berupa pertanyaan, saran, atau pernyataan umum tanpa emosi spesifik.')

    else:
        st.warning('Mohon masukkan ulasan untuk dianalisis.')
"""

with open('app.py', 'w') as f:
    f.write(app_py_content)

print("File 'app.py' berhasil diperbarui!")
