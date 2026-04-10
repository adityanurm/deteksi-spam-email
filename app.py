import streamlit as st
import pandas as pd
import mysql.connector
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Deteksi Spam Email", page_icon="📧")

# --- 1. FUNGSI DATABASE (RAILWAY) ---
def simpan_ke_db(pesan, status):
    try:
        # KONEKSI KE RAILWAY
        mydb = mysql.connector.connect(
            host="maglev.proxy.rlwy.net",
            user="root",
            password="lJkHFNFqWKyHARuimAWMfByCIeKoIrKH", # <-- GANTI DENGAN PASSWORD RAILWAY KAMU
            database="railway",
            port=38136
        )
        cursor = mydb.cursor()
        
        # Buat tabel otomatis jika belum ada
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS riwayat (
                id INT AUTO_INCREMENT PRIMARY KEY,
                teks TEXT,
                hasil VARCHAR(50),
                waktu TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Simpan data
        sql = "INSERT INTO riwayat (teks, hasil) VALUES (%s, %s)"
        val = (pesan, status)
        cursor.execute(sql, val)
        
        mydb.commit()
        mydb.close()
        st.sidebar.success("✅ Tersimpan di Cloud (Railway)!")
    except Exception as e:
        # Jika database gagal, aplikasi tetap jalan tapi muncul peringatan
        st.sidebar.warning(f"⚠️ Database belum terkoneksi: {e}")

# --- 2. PERSIAPAN MODEL AI (NAIVE BAYES) ---
@st.cache_resource
def load_model():
    try:
        # Membaca data training (pastikan file spam_data.csv ada di GitHub)
        # Menggunakan sep='|' karena data kamu pakai garis tegak
        df = pd.read_csv('spam_data.csv', sep='|', names=['label', 'text'], skiprows=1)
        
        x = df['text']
        y = df['label']
        
        cv = CountVectorizer()
        x = cv.fit_transform(x)
        
        model = MultinomialNB()
        model.fit(x, y)
        return cv, model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None

cv, model = load_model()

# --- 3. TAMPILAN APLIKASI ---
st.title("📧 Deteksi Spam Email")
st.write("Aplikasi ini menggunakan AI untuk mendeteksi apakah pesan termasuk **Spam** atau **Bukan**.")

input_text = st.text_area("Masukkan pesan email di sini:", placeholder="Contoh: Selamat Anda menang undian 1 Miliar!")

if st.button("Periksa Sekarang"):
    if input_text:
        if model:
            # Prediksi
            data = cv.transform([input_text]).toarray()
            prediksi = model.predict(data)
            
            hasil = "SPAM" if prediksi[0] == 1 else "BUKAN SPAM (HAM)"
            
            # Tampilkan Hasil
            if prediksi[0] == 1:
                st.error(f"Hasil Analisis: **{hasil}**")
            else:
                st.success(f"Hasil Analisis: **{hasil}**")
            
            # Simpan ke Database Railway
            simpan_ke_db(input_text, hasil)
        else:
            st.error("Model AI tidak tersedia.")
    else:
        st.warning("Silakan masukkan teks terlebih dahulu!")

st.divider()
st.caption("Dibuat untuk Project Deteksi Spam - Terhubung ke Railway Cloud Database")
