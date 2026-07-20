import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# 1. Judul Aplikasi
st.title("Hasil Analisis Perhitungan")

# 2. Logika Perhitungan Anda dari Colab
# Contoh: Membaca dataset / menghitung rumus
data = {'Kategori': ['A', 'B', 'C'], 'Nilai': [10, 25, 15]}
df = pd.DataFrame(data)

total_nilai = df['Nilai'].sum()

# 3. Menampilkan Hasil Perhitungan ke Streamlit
st.metric(label="Total Nilai Perhitungan", value=total_nilai)

# Menampilkan Tabel Data
st.subheader("Tabel Data")
st.dataframe(df)

# Menampilkan Grafik (jika ada)
fig, ax = plt.subplots()
ax.bar(df['Kategori'], df['Nilai'])
st.pyplot(fig)
