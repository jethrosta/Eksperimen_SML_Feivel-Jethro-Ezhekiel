# 1. Memproses data kosong
# 2. Deteksi Outlier
# 3. Feature Scaling
# 4. Lakukan Encoder - OHE
# 5. Binning

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data_jethro_advanced(path_prefix="preprocessing/Dataset/"):
    """
    Memuat dan melakukan preprocessing lengkap pada dataset terkait kemiskinan dan upah.

    Langkah-langkah yang dilakukan:
    1.  Memuat 4 dataset dan menggabungkannya.
    2.  Membersihkan dan mengubah tipe data.
    3.  Menangani nilai kosong (meskipun pada kasus ini tidak ada).
    4.  Melakukan binning pada kolom 'tahun'.
    5.  Mendeteksi dan menampilkan outlier pada data numerik.
    6.  Melakukan One-Hot Encoding pada data kategorikal.
    7.  Melakukan Feature Scaling (StandardScaler) pada data numerik.
    8.  Menggabungkan hasil scaling dan encoding menjadi DataFrame akhir.

    Args:
        path_prefix (str): Path ke direktori tempat file-file CSV disimpan.

    Returns:
        pandas.DataFrame: Sebuah DataFrame yang siap digunakan untuk pemodelan.
                          Mengembalikan None jika file tidak ditemukan.
    """
    try:
        # **1. Memuat dan Menggabungkan Dataset**
        garis_kemiskinan_df = pd.read_csv(f"{path_prefix}garisKemiskinan.csv")
        min_upah_df = pd.read_csv(f"{path_prefix}minUpah.csv")
        pengeluaran_df = pd.read_csv(f"{path_prefix}pengeluaran.csv")
        avg_upah_df = pd.read_csv(f"{path_prefix}rataRataUpah.csv")
        print("✅ Dataset berhasil dimuat.")

        # Menggabungkan semua dataframe menjadi satu
        df_merged = pd.merge(garis_kemiskinan_df, min_upah_df, on=['provinsi', 'tahun', 'daerah'])
        df_merged = pd.merge(df_merged, pengeluaran_df, on=['provinsi', 'tahun', 'daerah', 'jenis'])
        df_merged = pd.merge(df_merged, avg_upah_df, on=['provinsi', 'tahun'])

        # Mengganti nama kolom agar lebih konsisten
        df_merged.rename(columns={
            'garis_kemiskinan_x': 'garis_kemiskinan',
            'jenis': 'Jenis_Pengeluaran',
            'upah': 'Upah_Rata_rata'
        }, inplace=True)
        df_merged.drop(columns=['periode', 'garis_kemiskinan_y'], inplace=True)
        print("🔄 Data berhasil digabungkan dan dibersihkan.")

    except FileNotFoundError as e:
        print(f"❌ Error: File tidak ditemukan. {e}")
        return None

    # **2. Menangani Data Kosong**
    if df_merged.isnull().sum().sum() > 0:
        print("Warning: Ditemukan nilai kosong. Anda mungkin perlu menambah strategi penanganan (imputasi).")
        # Contoh: df_merged.fillna(df_merged.median(), inplace=True)
    else:
        print("✅ Tidak ada nilai kosong pada data.")

    # **3. Binning untuk Kolom 'tahun'**
    bins = [2014, 2017, 2020, 2023]
    labels = ['2015-2017', '2018-2020', '2021-2023']
    df_merged['periode_tahun'] = pd.cut(df_merged['tahun'], bins=bins, labels=labels, right=True)
    print("🔄 Binning pada kolom 'tahun' selesai.")

    # **4. Deteksi Outlier**
    numerical_cols = ['garis_kemiskinan', 'ump', 'pengeluaran', 'Upah_Rata_rata']
    print("\n📊 Menganalisis Outlier (menggunakan metode IQR)...")
    for col in numerical_cols:
        Q1 = df_merged[col].quantile(0.25)
        Q3 = df_merged[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df_merged[(df_merged[col] < lower_bound) | (df_merged[col] > upper_bound)]
        print(f"  - Kolom '{col}': Ditemukan {len(outliers)} outlier.")

    # Visualisasi boxplot untuk outlier
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=df_merged[numerical_cols])
    plt.title('Boxplot untuk Deteksi Outlier pada Fitur Numerik')
    plt.ylabel('Nilai')
    plt.xticks(rotation=45)
    plt.show()

    # **5 & 6. Feature Scaling dan One-Hot Encoding**
    categorical_cols = ['provinsi', 'daerah', 'Jenis_Pengeluaran', 'periode_tahun']

    # Membuat pipeline untuk transformasi
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough' # Membiarkan kolom lain (seperti 'tahun') tidak diubah
    )

    # Melakukan transformasi
    processed_data = preprocessor.fit_transform(df_merged)
    print("\n🔄 Feature Scaling (StandardScaler) dan One-Hot Encoding (OHE) selesai.")

    # Mengambil nama kolom setelah OHE untuk membuat DataFrame baru
    ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    final_cols = numerical_cols + list(ohe_feature_names) + ['tahun']

    # Membuat DataFrame hasil proses
    # Perlu diperhatikan bahwa `processed_data` adalah numpy array (atau sparse matrix)
    # Untuk kemudahan inspeksi, kita ubah kembali ke DataFrame
    df_processed = pd.DataFrame(processed_data.toarray() if hasattr(processed_data, "toarray") else processed_data, columns=final_cols)
    print("✅ DataFrame akhir berhasil dibuat.")

    return df_processed


if __name__ == "__main__":
    # Panggil fungsi untuk mendapatkan data yang sudah diproses secara lengkap
    df_final = preprocess_data_jethro_advanced()

    if df_final is not None:
        print("\n### Informasi DataFrame Hasil Preprocessing ###")
        df_final.info()

        print("\n### 5 Baris Pertama DataFrame Hasil Preprocessing ###")
        print(df_final.head())

        print(f"\nDimensi DataFrame akhir: {df_final.shape}")