import os
import sys
import warnings
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import mlflow
import mlflow.sklearn

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # 1. Muat data Anda (tidak berubah)
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "IndonesiaWorkfare_preprocessing.csv")
    data = pd.read_csv(file_path)

    X = data.copy()
    input_example = X.head()

    # 2. Tentukan Rentang Hyperparameter untuk di-tuning
    # Kita akan mencoba nilai k (n_clusters) dari 2 hingga 10
    k_values_to_tune = range(2, 11)
    print(f"Memulai hyperparameter tuning untuk n_clusters dari {min(k_values_to_tune)} hingga {max(k_values_to_tune)}...")

    # 3. Looping untuk setiap nilai hyperparameter
    for k in k_values_to_tune:
        # Memulai Run baru untuk setiap nilai K dengan nama yang deskriptif
        with mlflow.start_run(run_name=f"KMeans_k={k}"):
            print(f"-- Mencoba k={k} --")

            # a. Log Parameter
            # Catat parameter yang sedang di-tuning (k) dan parameter tetap lainnya
            mlflow.log_param("n_clusters", k)
            mlflow.log_param("init", "k-means++")
            mlflow.log_param("n_init", "auto")
            mlflow.log_param("max_iter", 300)
            mlflow.log_param("random_state", 40)
            mlflow.log_param("algorithm", "lloyd")

            # b. Inisialisasi dan Latih Model
            model = KMeans(n_clusters=k, 
                           init="k-means++", 
                           n_init="auto", 
                           max_iter=300, 
                           random_state=40,
                           algorithm="lloyd")
            model.fit(X)

            cluster_labels = model.labels_

            # c. Log Metriks (sama seperti yang dicatat autolog)
            inertia = model.inertia_
            mlflow.log_metric("inertia", inertia)

            sil_score = silhouette_score(X, cluster_labels)
            mlflow.log_metric("silhouette_score", sil_score)

            # d. Log Model sebagai Artefak
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=f"model_k_{k}", # Path unik untuk setiap model
                input_example=input_example
            )

    print("\n✅ Proses hyperparameter tuning selesai.")

    # 4. (Opsional) Cari dan tampilkan hasil run terbaik berdasarkan Silhouette Score
    all_runs = mlflow.search_runs(order_by=["metrics.silhouette_score DESC"])
    best_run = all_runs.iloc[0]
    
    print("\n--- Hasil Tuning Terbaik ---")
    print(f"Run ID: {best_run['run_id']}")
    print(f"Parameter n_clusters terbaik: {best_run['params.n_clusters']}")
    print(f"Silhouette Score terbaik: {best_run['metrics.silhouette_score']:.4f}")
    print(f"Inertia: {best_run['metrics.inertia']:.2f}")