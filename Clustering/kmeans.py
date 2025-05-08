import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, adjusted_rand_score, silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

def analyze_kmeans_clustering(n_clusters, dataframes, dataframe_names, true_labels, distance_type):
    
    # Encode true labels
    le = LabelEncoder()
    y_true = le.fit_transform(true_labels)

    # Prepare results storage
    results = {
        "CSV Name": [],
        "Accuracy": [],
        "ARI": [],
        "Silhouette Score": [],
        "Davies-Bouldin Score": []
    }

    # Calculate grid size for subplots
    num_plots = len(dataframes)
    num_cols = 2
    num_rows = int(np.ceil(num_plots / num_cols))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 6 * num_rows))
    axes = axes.flatten()

    for idx, df in enumerate(dataframes):
        # Drop non-feature columns
        df_proc = df.copy()
        for col in ['label', 'filename']:
            if col in df_proc:
                df_proc = df_proc.drop(columns=[col])

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_proc)

        # Run custom KMeans
        centroids, kmeans_labels = kmeans(
            X_scaled, k=n_clusters, distance_type=distance_type, random_state=42
        )

        # Align labels via Hungarian algorithm
        n_classes = len(np.unique(y_true))
        cont = np.zeros((n_clusters, n_classes), dtype=int)
        for j, label in enumerate(kmeans_labels):
            cont[label, y_true[j]] += 1
        row_ind, col_ind = linear_sum_assignment(-cont)
        aligned = np.zeros_like(kmeans_labels)
        for k in range(n_clusters):
            aligned[kmeans_labels == row_ind[k]] = col_ind[k]

        # Convert aligned numeric labels back to original names
        predicted_names = le.inverse_transform(aligned)

        # Compute metrics
        acc = accuracy_score(y_true, aligned)
        ari = adjusted_rand_score(y_true, kmeans_labels)
        sil = silhouette_score(X_scaled, kmeans_labels)
        dbs = davies_bouldin_score(X_scaled, kmeans_labels)

        # Store metrics
        results["CSV Name"].append(dataframe_names[idx])
        results["Accuracy"].append(acc)
        results["ARI"].append(ari)
        results["Silhouette Score"].append(sil)
        results["Davies-Bouldin Score"].append(dbs)

        # 2D PCA projection
        pca = PCA(n_components=2)
        X2 = pca.fit_transform(X_scaled)

        ax = axes[idx]
        # Plot each predicted label separately for legend
        unique_labels = np.unique(predicted_names)
        for label_name in unique_labels:
            mask = (predicted_names == label_name)
            ax.scatter(
                X2[mask, 0], X2[mask, 1],
                label=f'{label_name}',
                alpha=0.6
            )

        # Annotate each point with its predicted label name
        # for j, (x, y) in enumerate(X2):
        #     ax.text(x, y, predicted_names[j], fontsize=8, alpha=0.7)

        ax.set_title(f"{dataframe_names[idx]}")
        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        ax.legend(title='Predicted Label')

    # Hide unused subplots
    for j in range(num_plots, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

    # Return metrics DataFrame
    return pd.DataFrame(results)
