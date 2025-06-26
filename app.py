# app.py

import streamlit as st
import pandas as pd
import pickle

# Fungsi untuk memuat artifak (model dan encoder)
# @st.cache_resource digunakan agar Streamlit tidak memuat ulang file setiap kali ada interaksi
@st.cache_resource
def load_artifacts():
    with open('encoder.pkl', 'rb') as f_encoder:
        encoder = pickle.load(f_encoder)
    with open('model.pkl', 'rb') as f_model:
        model = pickle.load(f_model)
    with open('label_encoder.pkl', 'rb') as f_label_encoder:
        label_encoder = pickle.load(f_label_encoder)
    return encoder, model, label_encoder

# Muat file-file yang diperlukan
encoder, model, label_encoder = load_artifacts()

# Judul dan Deskripsi Aplikasi
st.title("🎲 Prediktor Kemenangan Tic-Tac-Toe")
st.write(
    "Aplikasi ini menggunakan model K-Nearest Neighbors (KNN) untuk memprediksi "
    "apakah pemain 'x' akan menang (`positive`) berdasarkan kondisi papan permainan akhir."
)
st.markdown("---")


# Membuat Tampilan Input Papan Permainan
st.sidebar.header("Masukkan Kondisi Papan")
options = ['x', 'o', 'b']  # 'b' untuk blank (kosong)
col_names = [
    'top-left-square', 'top-middle-square', 'top-right-square',
    'middle-left-square', 'middle-middle-square', 'middle-right-square',
    'bottom-left-square', 'bottom-middle-square', 'bottom-right-square'
]

input_data = {}
for i, col_name in enumerate(col_names):
    # Membuat label yang lebih mudah dibaca, misal: "Top Left Square"
    label = col_name.replace('-', ' ').title()
    input_data[col_name] = st.sidebar.selectbox(label, options, key=f"sb_{i}")


# Tombol untuk melakukan prediksi
if st.button("🔮 Prediksi Hasil Permainan"):
    # 1. Kumpulkan input menjadi DataFrame
    # Input harus dalam bentuk list of a dictionary agar menjadi 1 baris DataFrame
    input_df = pd.DataFrame([input_data])

    # Tampilkan input pengguna untuk verifikasi
    st.subheader("Kondisi Papan yang Dimasukkan:")
    st.table(input_df)

    # 2. Lakukan encoding pada input DataFrame menggunakan encoder yang sudah diload
    encoded_input = encoder.transform(input_df)

    # 3. Lakukan prediksi dengan model yang sudah diload
    prediction_encoded = model.predict(encoded_input)

    # 4. Decode hasil prediksi dari angka (0/1) kembali ke teks ('negative'/'positive')
    prediction_text = label_encoder.inverse_transform(prediction_encoded)

    # 5. Tampilkan hasil prediksi dengan gaya
    st.markdown("---")
    st.subheader("Hasil Prediksi:")
    result = prediction_text[0]
    
    if result == 'positive':
        st.success(f"**{result.upper()}** - Pemain 'x' kemungkinan besar **MENANG**.")
    else:
        st.error(f"**{result.upper()}** - Pemain 'x' kemungkinan besar **TIDAK MENANG** (Kalah atau Seri).")