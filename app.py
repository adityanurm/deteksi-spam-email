import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import os

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Deteksi Spam Email", page_icon="🛡️")

# --- 2. FUNGSI DATABASE (VERSI ONLINE: DINONAKTIFKAN) ---
def simpan_ke_db(pesan, status):
    # Dikosongkan agar link internet tidak error
    pass

# --- 3. FUNGSI AI ---
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
st.caption("Aplikasi online berbasis Machine Learning")

if df_asli is not None:
    st.sidebar.header("📊 Statistik Data")
    counts = df_asli[df_asli.columns[0]].value_counts()
    labels = ['Spam' if str(x) == '1' else 'Aman' for x in counts.index]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(counts, labels=labels, autopct='%1.1f%%', colors=['#FF5252', '#4CAF50'], startangle=90)
    ax.axis('equal') 
    st.sidebar.pyplot(fig)

    user_input = st.text_area("Masukkan pesan untuk diperiksa:", height=150)
    
    if st.button("Periksa Sekarang"):
        if user_input.strip() != "":
            input_vec = vectorizer.transform([user_input.lower()])
            hasil = model.predict(input_vec)
            prob = model.predict_proba(input_vec)
            status = "SPAM" if str(hasil[0]) == '1' else "AMAN"
            
            st.divider()
            if status == "SPAM":
                st.error(f"### 🚨 HASIL: {status}")
                st.write(f"Keyakinan: {prob[0][1]*100:.1f}%")
            else:
                st.success(f"### ✅ HASIL: {status}")
                st.write(f"Keyakinan: {prob[0][0]*100:.1f}%")
            simpan_ke_db(user_input, status)
else:
    st.error("File 'spam_data.csv' tidak ditemukan!")
