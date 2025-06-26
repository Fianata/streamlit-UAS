# save_model.py

import pandas as pd
import pickle
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

print("Memulai proses training dan penyimpanan model...")

# 1. Muat data langsung dari UCI Repository (lebih robust daripada CSV)
tic_tac_toe_endgame = fetch_ucirepo(id=101)
X = tic_tac_toe_endgame.data.features
y = tic_tac_toe_endgame.data.targets['class'] # Langsung ambil series 'class'

# 2. Preprocessing (Encoding) - Sesuai dengan notebook Anda
print("Melakukan encoding pada data...")

# Encoder untuk fitur (X). sparse_output=False agar mudah dilihat/di-debug
encoder_X = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_encoded = encoder_X.fit_transform(X)

# Encoder untuk target (y)
encoder_y = LabelEncoder()
y_encoded = encoder_y.fit_transform(y)

# 3. Training Model Terbaik (KNN) pada Seluruh Data
print("Melatih model KNN pada seluruh dataset...")
# Menggunakan parameter yang sama seperti di notebook (n_neighbors=5)
final_knn_model = KNeighborsClassifier(n_neighbors=5)
final_knn_model.fit(X_encoded, y_encoded)

# 4. Simpan objek-objek yang diperlukan ke file menggunakan pickle
# Kita hanya perlu menyimpan encoder fitur, model, dan encoder label.
with open('encoder.pkl', 'wb') as f:
    pickle.dump(encoder_X, f)

with open('model.pkl', 'wb') as f:
    pickle.dump(final_knn_model, f)
    
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(encoder_y, f)

print("\nProses selesai!")
print("Tiga file telah berhasil disimpan:")
print("1. encoder.pkl (untuk memproses input baru)")
print("2. model.pkl (model KNN yang sudah dilatih)")
print("3. label_encoder.pkl (untuk mengubah hasil prediksi kembali ke 'positive'/'negative')")