import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import mysql.connector
import matplotlib.pyplot as plt
import os

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Deteksi Spam Email", page_icon="🛡️")

# --- 2. FUNGSI DATABASE XAMPP (ANTI-ERROR) ---
def simpan_ke_db(pesan, status):
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="db_spam"
        )
        cursor = conn.cursor()
        query = "INSERT INTO riwayat (teks, hasil) VALUES (%s, %s)"
        cursor.execute(query, (pesan, status))
        conn.commit()
        conn.close()
    except:
        # Jika XAMPP mati, aplikasi tidak akan error, hanya muncul info di sidebar
        st.sidebar.warning("⚠️ Gagal menyimpan ke Database (XAMPP Mati?)")

# --- 3. FUNGSI AI (STABIL) ---
@st.cache_resource
def build_ai_model():
    nama_file = 'spam_data.csv'
    if not os.path.exists(nama_file): return None, None, None
    
    try:
        # Menggunakan pemisah '|' agar aman dari koma/titik koma di teks
        df = pd.read_csv(nama_file, encoding='latin-1', sep='|', on_bad_lines='skip')
        df = df.dropna()
        
        vectorizer = CountVectorizer(stop_words='english')
        X = vectorizer.fit_transform(df[df.columns[1]].values.astype('U'))
        model = MultinomialNB()
        model.fit(X, df[df.columns[0]])
        return vectorizer, model, df
    except:
        return None, None, None

vectorizer, model, df_asli = build_ai_model()

# --- 4. TAMPILAN UTAMA ---
st.title("🛡️ Deteksi Spam Email")
st.caption("Aplikasi cerdas berbasis Machine Learning untuk menyaring pesan berbahaya")

if df_asli is not None:
    # VISUALISASI DI SIDEBAR
    st.sidebar.header("📊 Statistik Data")
    counts = df_asli[df_asli.columns[0]].value_counts()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(counts, labels=['Aman (0)', 'Spam (1)'], autopct='%1.1f%%', colors=['#4CAF50', '#FF5252'])
    st.sidebar.pyplot(fig)

    # AREA INPUT
    user_input = st.text_area("Masukkan pesan email/SMS yang ingin diperiksa:", height=150)
    
    if st.button("Periksa Sekarang"):
        if user_input.strip() != "":
            # Prediksi AI
            input_vec = vectorizer.transform([user_input.lower()])
            hasil = model.predict(input_vec)
            prob = model.predict_proba(input_vec)
            status = "SPAM" if str(hasil[0]) == '1' else "AMAN"
            
            st.divider()
            if status == "SPAM":
                st.error(f"### 🚨 HASIL: INI ADALAH {status}!")
                st.write(f"Keyakinan AI: {prob[0][1]*100:.1f}%")
            else:
                st.success(f"### ✅ HASIL: PESAN INI {status}")
                st.write(f"Keyakinan AI: {prob[0][0]*100:.1f}%")
            
            # Simpan ke Database
            simpan_ke_db(user_input, status)
        else:
            st.warning("Mohon ketikkan pesan terlebih dahulu.")

    # TABEL RIWAYAT
    st.divider()
    with st.expander("🕒 Lihat Riwayat Pengecekan dari Database"):
        try:
            conn = mysql.connector.connect(host="localhost", user="root", password="", database="db_spam")
            df_db = pd.read_sql("SELECT teks, hasil, waktu FROM riwayat ORDER BY waktu DESC LIMIT 10", conn)
            st.table(df_db)
            conn.close()
        except:
            st.info("Hubungkan ke XAMPP MySQL untuk mengaktifkan fitur riwayat.")
else:
    st.error("File 'spam_data.csv' tidak ditemukan atau rusak!")