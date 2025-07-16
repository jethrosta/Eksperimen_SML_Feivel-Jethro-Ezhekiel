import os
import sys
import warnings
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import mlflow
import mlflow.sklearn
import dagshub

dagshub.init(repo_owner='jethrosta', repo_name='Eksperimen_SML_Feivel-Jethro-Ezhekiel', mlflow=True)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Muat data Anda
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "IndonesiaWorkfare_preprocessing.csv")
    data = pd.read_csv(file_path)

    X = data.copy()
    input_example = X.head()

    # Tentukan rentang hyperparameter
    k_values_to_tune = range(2, 11)
    print(f"Memulai hyperparameter tuning untuk n_clusters...")

    for k in k_values_to_tune:
        with mlflow.start_run(run_name=f"KMeans_k={k}"):
            print(f"-- Mencoba k={k} --")

            # a. Log Parameter (tidak berubah)
            mlflow.log_param("n_clusters", k)
            mlflow.log_param("init", "k-means++")
            mlflow.log_param("random_state", 40)
            
            # b. Latih Model (tidak berubah)
            model = KMeans(n_clusters=k, init="k-means++", n_init="auto", random_state=40)
            model.fit(X)
            cluster_labels = model.labels_

            # c. Log Metriks (dengan tambahan 2 metrik baru)
            # Metrik dari autolog
            inertia = model.inertia_
            sil_score = silhouette_score(X, cluster_labels)
            
            calinski_score = calinski_harabasz_score(X, cluster_labels)
            davies_score = davies_bouldin_score(X, cluster_labels)
            
            mlflow.log_metric("inertia", inertia)
            mlflow.log_metric("silhouette_score", sil_score)
            # Log metrik baru
            mlflow.log_metric("calinski_harabasz_score", calinski_score)
            mlflow.log_metric("davies_bouldin_score", davies_score)

            # d. Log Model (tidak berubah)
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=f"model_k_{k}",
                input_example=input_example
            )

    print("\n✅ Proses hyperparameter tuning selesai dan log dikirim ke DagsHub.")