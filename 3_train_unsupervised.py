import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans, AgglomerativeClustering, HDBSCAN
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score,
    homogeneity_score, completeness_score,
    v_measure_score, silhouette_score,
)


if len(sys.argv) != 2:
    sys.exit("Usage: python 3_train_unsupervised.py <wavlm|ecapa>")
embedding_name = sys.argv[1]
print(embedding_name)


X = np.load(f"embeddings/{embedding_name}_embeddings.npy")

meta = pd.read_csv(f"embeddings/{embedding_name}_embedding_index.csv")


# keep only labeled data
mask = meta["speaker"] != "Unknown"

X = X[mask]
meta = meta[mask].reset_index(drop=True)

y_true = meta["speaker"].to_numpy(dtype=str)

# ensure unit-normalized embeddings
X = normalize(X)

n_speakers = len(np.unique(y_true))

print("Embedding:", embedding_name)
print("Samples:", len(X))
print("Speakers:", n_speakers)
print()
print(meta["speaker"].value_counts())


models = {
    "kmeans": KMeans(
        n_clusters=n_speakers,
        random_state=42,
        n_init="auto",
    ),

    "agglomerative": AgglomerativeClustering(
        n_clusters=n_speakers,
        metric="cosine",
        linkage="average",
    ),

    "hdbscan": HDBSCAN(
        min_cluster_size=5,
        min_samples=5,
        metric="euclidean",
    ),
}


def evaluate_clustering(name, model):

    output_dir = os.path.join("trained_models", "unsupervised", embedding_name, name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{name}")

    clusters = model.fit_predict(X)

    ari = adjusted_rand_score(y_true, clusters)

    nmi = normalized_mutual_info_score(y_true, clusters)

    homogeneity = homogeneity_score(y_true, clusters)

    completeness = completeness_score(y_true, clusters)

    v_measure = v_measure_score(y_true, clusters)

    # HDBSCAN uses -1 for noise
    valid_mask = clusters != -1
    valid_clusters = np.unique(
        clusters[valid_mask]
    )

    if (
        valid_mask.sum() > 1
        and len(valid_clusters) > 1
    ):
        silhouette = silhouette_score(
            X[valid_mask],
            clusters[valid_mask],
            metric="cosine",
        )
    else:
        silhouette = np.nan

    noise_count = (clusters == -1).sum()

    print("Clusters found:", len(valid_clusters))
    print("Noise points:", noise_count)
    print("ARI:", ari)
    print("NMI:", nmi)
    print("Homogeneity:", homogeneity)
    print("Completeness:", completeness)
    print("V-measure:", v_measure)
    print("Silhouette:", silhouette)

    assignments = meta[
        ["audio_id", "speaker"]
    ].copy()

    assignments["cluster"] = clusters

    assignments.to_csv(
        os.path.join(
            output_dir,
            "assignments.csv"
        ),
        index=False,
    )

    table = pd.crosstab(
        assignments["speaker"],
        assignments["cluster"],
    )

    print("\nSpeaker vs cluster:")
    print(table)

    # save report
    with open(
        os.path.join(
            output_dir,
            "report.txt"
        ),
        "w"
    ) as f:

        f.write(f"Embedding: {embedding_name}\n")
        f.write(f"Method: {name}\n")
        f.write(f"Samples: {len(X)}\n")
        f.write(f"True speakers: {n_speakers}\n")
        f.write(f"Clusters found: {len(valid_clusters)}\n")
        f.write(f"Noise points: {noise_count}\n\n")

        f.write(f"ARI: {ari:.6f}\n")
        f.write(f"NMI: {nmi:.6f}\n")
        f.write(f"Homogeneity: {homogeneity:.6f}\n")
        f.write(f"Completeness: {completeness:.6f}\n")
        f.write(f"V-measure: {v_measure:.6f}\n")
        f.write(f"Silhouette: {silhouette:.6f}\n")

        f.write("\nSpeaker vs cluster:\n")
        f.write(table.to_string())

    # heatmap
    plt.figure(figsize=(10, 6))

    sns.heatmap(
        table,
        annot=True,
        fmt="d",
        cmap="Blues",
    )

    plt.xlabel("Discovered cluster")
    plt.ylabel("True speaker")
    plt.title(
        f"{embedding_name.upper()} - {name}"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "cluster_vs_speaker.png"
        ),
        dpi=200,
    )

    plt.close()


for name, model in models.items():
    evaluate_clustering(
        name,
        model
    )
