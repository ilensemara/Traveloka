
import streamlit as st
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

# Preprocessing function (copy from your notebook)
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

        # Predict sentiment with Naïve Bayes
        nb_prediction = nb_model.predict(input_tfidf)[0]

        # Predict sentiment with SVM
        svm_prediction = svm_model.predict(input_tfidf)[0]

        st.subheader('Hasil Analisis Sentimen:')
        st.write(f'**Naïve Bayes:** {nb_prediction}')
        st.write(f'**SVM:** {svm_prediction}')

        # Optional: Show prediction probabilities
        nb_proba = nb_model.predict_proba(input_tfidf)[0]
        svm_proba = svm_model.predict_proba(input_tfidf)[0]

        sentiment_labels = nb_model.classes_ # Assuming both models have the same class order

        st.write('---')
        st.write('**Probabilitas Prediksi (Naïve Bayes):**')
        for label, prob in zip(sentiment_labels, nb_proba):
            st.write(f'- {label}: {prob:.2f}')
        
        st.write('---')
        st.write('**Probabilitas Prediksi (SVM):**')
        for label, prob in zip(sentiment_labels, svm_proba):
            st.write(f'- {label}: {prob:.2f}')

    else:
        st.warning('Mohon masukkan ulasan untuk dianalisis.')
