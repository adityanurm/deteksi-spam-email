# --- 2. FUNGSI DATABASE (ONLINE MODE - RAILWAY) ---
def simpan_ke_db(pesan, status):
    try:
        # Koneksi ke Database Cloud Railway menggunakan data Public Network
        mydb = mysql.connector.connect(
            host="maglev.proxy.rlwy.net",
            user="root",
            password="GANTI_DENGAN_PASSWORD_DARI_RAILWAY", # Klik 'show' di Railway
            database="railway",
            port=38136
        )
        cursor = mydb.cursor()
        
        # PERINTAH SAKTI: Membuat tabel otomatis jika belum ada di Railway
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS riwayat (
                id INT AUTO_INCREMENT PRIMARY KEY,
                teks TEXT,
                hasil VARCHAR(50),
                waktu TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Simpan data ke tabel
        sql = "INSERT INTO riwayat (teks, hasil) VALUES (%s, %s)"
        val = (pesan, status)
        cursor.execute(sql, val)
        
        mydb.commit()
        mydb.close()
        st.sidebar.success("✅ Tersimpan di Cloud (Railway)!")
    except Exception as e:
        # Jika gagal, akan muncul pesan error di sidebar untuk pengecekan
        st.sidebar.error(f"⚠️ Gagal simpan ke Cloud: {e}")