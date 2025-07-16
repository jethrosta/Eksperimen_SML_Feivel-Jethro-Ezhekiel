# 1. Memproses data kosong
# 2. Deteksi Outlier
# 3. Feature Scaling
# 4. Lakukan Encoder - OHE
# 5. Binning

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Mengunakan Metode Agregasi
def replace_outliers_with_median(df, column_name):
    """
    Mengganti outliers pada kolom tertentu dengan nilai median menggunakan metode IQR.
    
    Parameters:
    df (pd.DataFrame): DataFrame yang berisi data.
    column_name (str): Nama kolom yang ingin diproses.
    
    Returns:
    pd.DataFrame: DataFrame dengan outliers yang telah diganti median.
    """
    Q1 = df[column_name].quantile(0.25)  # Kuartil pertama (Q1)
    Q3 = df[column_name].quantile(0.75)  # Kuartil ketiga (Q3)
    IQR = Q3 - Q1  # Interquartile Range (IQR)

    lower_bound = Q1 - 1.5 * IQR  # Batas bawah
    upper_bound = Q3 + 1.5 * IQR  # Batas atas

    median_value = df[column_name].median()  # Nilai median
    
    # Ganti outlier dengan median menggunakan apply()
    df[column_name] = df[column_name].apply(lambda x: median_value if x < lower_bound or x > upper_bound else x)

    return df

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Hapus baris yang memiliki outliers
    df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_clean

def automate_preprocessing():
    # Load datasets
    garisKemiskinan_df = pd.read_csv("Dataset/garisKemiskinan.csv")
    minUpah_df = pd.read_csv("Dataset/minUpah.csv")
    pengeluaran_df = pd.read_csv("Dataset/pengeluaran.csv")
    avgUpah_df = pd.read_csv("Dataset/rataRataUpah.csv")

    # Penyesuaian nama kolom
    garisKemiskinan_df.rename(columns={'jenis': 'Jenis_Pengeluaran'}, inplace=True)
    garisKemiskinan_df.drop(columns=['periode'], inplace=True)
    pengeluaran_df.rename(columns={'jenis': 'Jenis_Pengeluaran'}, inplace=True)
    avgUpah_df.rename(columns={'upah': 'Upah_Rata_rata'}, inplace=True)

    # Gabungkan garisKemiskinan_df dengan minUpah_df
    merged_df = pd.merge(garisKemiskinan_df, minUpah_df, on=['provinsi', 'tahun'], how='outer')
    # Gabungkan hasil dengan pengeluaran_df
    merged_df = pd.merge(merged_df, pengeluaran_df, on=['provinsi', 'tahun', 'Jenis_Pengeluaran', 'daerah'], how='outer')
    # Gabungkan hasil dengan avgUpah_df
    merged_df = pd.merge(merged_df, avgUpah_df, on=['provinsi', 'tahun'], how='outer')
    
    # Menghapus semua data kosong(Apakah jumlah data akan berkurang drastis?)
    temp_df = merged_df
    temp_df = temp_df.dropna()
    print("Berhasil rename dan drop")

    # Mengubah nilai float menjadi bentuk integer
    # data gk, ump, peng, Upah_Rata_rata
    temp_df[['gk', 'ump', 'peng', 'Upah_Rata_rata']] = temp_df[['gk', 'ump', 'peng', 'Upah_Rata_rata']].astype(int)
    # Mengubah nama kolom agar lebih rapih
    temp_df.rename(columns={'gk': 'Garis Kemiskinan'}, inplace=True)
    temp_df.rename(columns={'ump': 'Upah Minimum Provinsi'}, inplace=True)
    temp_df.rename(columns={'peng': 'Pengeluaran'}, inplace=True)

    temp_df = replace_outliers_with_median(temp_df, "Garis Kemiskinan")
    temp_df = replace_outliers_with_median(temp_df, "Upah Minimum Provinsi")
    temp_df = replace_outliers_with_median(temp_df, "Upah_Rata_rata")

    # Pilih kolom yang akan distandarisasi
    columns_to_scale = ["Upah Minimum Provinsi", "Pengeluaran", "Upah_Rata_rata"]

    # Inisialisasi StandardScaler
    scaler = StandardScaler()

    # Transformasi data
    temp_df[columns_to_scale] = scaler.fit_transform(temp_df[columns_to_scale])
   
    temp_ohe = temp_df
    # Inisialisasi encoder
    ohe = OneHotEncoder(drop=None, sparse_output=False)

    # Pilih kolom yang ingin diencode
    categorical_cols = ["Jenis_Pengeluaran", "daerah"]

    # Transformasi data menjadi array
    encoded_array = ohe.fit_transform(temp_ohe[categorical_cols])

    # Konversi ke DataFrame
    encoded_df = pd.DataFrame(encoded_array, columns=ohe.get_feature_names_out(categorical_cols))

    # Gabungkan dengan dataframe asli (tanpa kolom yang diencode)
    temp_ohe = temp_ohe.drop(columns=categorical_cols).reset_index(drop=True)
    temp_ohe = pd.concat([temp_ohe, encoded_df], axis=1)

    # Melakukan One Hot Encoding pada Data Provinsi
    encoder = OneHotEncoder(sparse_output=False)
    temp_ohe_final = temp_ohe
    # Definisikan pembagian wilayah
    indonesia_timur = ["MALUKU", "MALUKU UTARA", "PAPUA", "PAPUA BARAT"]
    indonesia_tengah = ["KALIMANTAN BARAT", "KALIMANTAN SELATAN", "KALIMANTAN TENGAH", "KALIMANTAN TIMUR", "KALIMANTAN UTARA", 
                        "BALI", "NUSA TENGGARA BARAT", "NUSA TENGGARA TIMUR", "GORONTALO", 
                        "SULAWESI BARAT", "SULAWESI SELATAN", "SULAWESI TENGAH", "SULAWESI TENGGARA", "SULAWESI UTARA"]
    indonesia_barat = ["ACEH", "BENGKULU", "JAMBI", "LAMPUNG", "RIAU", "SUMATERA BARAT", "SUMATERA SELATAN", "SUMATERA UTARA", 
                    "KEP. BANGKA BELITUNG", "KEP. RIAU", "BANTEN", "DKI JAKARTA", "JAWA BARAT", "JAWA TENGAH", "JAWA TIMUR", "DI YOGYAKARTA"]

    # Buat fungsi untuk mengkategorikan provinsi
    def categorize_province(province):
        if province in indonesia_timur:
            return "Indonesia Timur"
        elif province in indonesia_tengah:
            return "Indonesia Tengah"
        elif province in indonesia_barat:
            return "Indonesia Barat"
        else:
            return "Indonesia"  # Untuk kategori "INDONESIA"

    # Terapkan fungsi pada kolom 'provinsi'
    temp_ohe_final["provinsi"] = temp_ohe_final["provinsi"].apply(categorize_province)

    # Cek hasilnya
    print(temp_ohe_final["provinsi"].value_counts())
    encoded_array = encoder.fit_transform(temp_ohe_final[["provinsi"]])

    # Mengonversi ke DataFrame
    encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(["provinsi"]))

    # Gabungkan dengan Data Awal
    df_encoded = pd.concat([temp_ohe_final, encoded_df], axis=1)

    # Melakukan drop pada bagian provinsi
    df_encoded = df_encoded.drop(columns=["provinsi"])
    df_encoded.head()

    
    return merged_df

if __name__ == "__main__":
    # Panggil fungsi untuk mendapatkan data yang sudah diproses secara lengkap
    df_final = automate_preprocessing()

    if df_final is not None:
        print("\n### Informasi DataFrame Hasil Preprocessing ###")
        df_final.info()

        print("\n### 5 Baris Pertama DataFrame Hasil Preprocessing ###")
        print(df_final.tail())

        print(f"\nDimensi DataFrame akhir: {df_final.shape}")