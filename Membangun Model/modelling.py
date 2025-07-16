import os
import sys
import warnings
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import mlflow
import mlflow.sklearn

# Cukup panggil autolog() satu kali di sini
mlflow.autolog()

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # 1. Load data Anda (tidak berubah)
    file_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "IndonesiaWorkfare_preprocessing.csv")
    data = pd.read_csv(file_path)

    X = data.copy()

    # 2. Ambil Parameter Clustering (tidak berubah)
    n_clusters = int(sys.argv[1]) if len(sys.argv) > 1 else 4

    # 3. Inisialisasi dan latih model
    # MLflow akan otomatis membuat 'run', mencatat parameter, metrik, dan model
    # saat Anda memanggil .fit()
    model = KMeans(n_clusters=n_clusters, random_state=40)
    model.fit(X)

    print(f"✅ Autologging selesai untuk model dengan {n_clusters} cluster.")
    print("Periksa MLflow UI Anda untuk melihat hasilnya.")