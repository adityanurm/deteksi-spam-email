import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import mysql.connector  # --- AKTIF ---
import matplotlib.pyplot as plt
import os

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Deteksi Spam Email", page_icon="🛡️")

# --- 2. FUNGSI DATABASE (SEKARANG AKTIF) ---
def simpan_ke_db(pesan, status):
    try:
        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="db_spam"  # Sesuaikan nama DB kamu
        )
        cursor = mydb.cursor()
        # Sesuaikan 'tabel_riwayat' dan kolomnya dengan phpMyAdmin kamu
        sql = "INSERT INTO tabel_riwayat (isi_pesan, hasil_status) VALUES (%s, %s)"
        val = (pesan, status)
        cursor.execute(sql, val)
        mydb.commit()
        mydb.close()
    except Exception as e:
        st.sidebar.error(f"DB Error: {e}")

# --- 3. FUNGSI AI (STABIL) ---
@st.cache_resource
def build_ai_model():
    nama_file = 'spam_data.csv'
    if not os.path.exists(nama_file): return None, None, None
    
    try:
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
    # VISUALISASI DI SIDEBAR (DIPERBAIKI AGAR TIDAK MIRING)
    st.sidebar.header("📊 Statistik Data")
    counts = df_asli[df_asli.columns[0]].value_counts()
    
    # Supaya label sesuai dengan jumlah data yang ada
    labels = ['Aman' if x == 0 else 'Spam' for x in counts.index]
    
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(counts, labels=labels, autopct='%1.1f%%', colors=['#4CAF50', '#FF5252'], startangle=90)
    ax.axis('equal')  # Memastikan lingkaran sempurna, tidak lonjong/miring
    st.sidebar.pyplot(fig)

    # AREA INPUT
    user_input = st.text_area("Masukkan pesan email/SMS yang ingin diperiksa:", height=150)
    
    if st.button("Periksa Sekarang"):
        if user_input.strip() != "":
            input_vec = vectorizer.transform([user_input.lower()])
            hasil = model.predict(input_vec)
            prob = model.predict_proba(input_vec)
            
            # Logika penentuan status (Sesuaikan dengan label di CSV kamu, misal '1' untuk spam)
            status = "SPAM" if str(hasil[0]) == '1' else "AMAN"
            
            st.divider()
            if status == "SPAM":
                st.error(f"### 🚨 HASIL: INI ADALAH {status}!")
                st.write(f"Keyakinan AI: {prob[0][1]*100:.1f}%")
            else:
                st.success(f"### ✅ HASIL: PESAN INI {status}")
                st.write(f"Keyakinan AI: {prob[0][0]*100:.1f}%")
            
            # --- SEKARANG DATA MASUK KE DB ---
            simpan_ke_db(user_input, status)
        else:
            st.warning("Mohon ketikkan pesan terlebih dahulu.")

    st.divider()
    with st.expander("🕒 Info Riwayat"):
        st.info("Setiap pengecekan akan tersimpan secara otomatis ke database XAMPP Anda.")

else:
    st.error("File 'spam_data.csv' tidak ditemukan!")
