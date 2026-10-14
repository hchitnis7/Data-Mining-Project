"""
Task 2: Tune cluster count.

Tests k values beyond 40 to see whether astro-ph.IM (and other
underperforming classes) form cleaner dedicated clusters with finer
granularity. Reuses the same TF-IDF + SVD pipeline, varying only k.

Run on GPU (cuML) if available, else falls back to sklearn automatically.
"""

import os
import pandas as pd
import numpy as np
import re
import time

os.makedirs("plots", exist_ok=True)

t0 = time.time()
def tick(msg):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

try:
    import cudf
    from cuml.feature_extraction.text import TfidfVectorizer
    from cuml.decomposition import TruncatedSVD
    from cuml.cluster import KMeans
    GPU = True
    tick("Using RAPIDS cuML (GPU)")
except ImportError:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.cluster import KMeans
    GPU = False
    tick("cuML not available, falling back to sklearn (CPU)")

df = pd.read_parquet("astroph_pre_with_future_labels.parquet")
df = df.dropna(subset=["future_subcat"]).copy().reset_index(drop=True)
tick(f"Working set: {len(df):,} papers")

ALL_CLASSES = ["astro-ph.CO", "astro-ph.SR", "astro-ph.GA",
               "astro-ph.IM", "astro-ph.HE", "astro-ph.EP"]
baseline_dist = df["future_subcat"].value_counts(normalize=True)


def clean_text(text):
    text = text.lower()
    text = re.sub(r"\$.*?\$", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


if "clean_abstract" not in df.columns:
    df["clean_abstract"] = df["abstract"].apply(clean_text)
tick("Text cleaned")

if GPU:
    gdf = cudf.from_pandas(df[["clean_abstract"]])
    vectorizer = TfidfVectorizer(max_features=8000, stop_words="english", min_df=5, max_df=0.5)
    X = vectorizer.fit_transform(gdf["clean_abstract"])
    X_dense = X.toarray()
    svd = TruncatedSVD(n_components=300, random_state=42)
    X_svd = svd.fit_transform(X_dense)
else:
    vectorizer = TfidfVectorizer(max_features=8000, stop_words="english", min_df=5, max_df=0.5)
    X = vectorizer.fit_transform(df["clean_abstract"])
    svd = TruncatedSVD(n_components=300, random_state=42)
    X_svd = svd.fit_transform(X)
tick("TF-IDF + SVD done, reused across all k values")


def get_svd_cpu(arr):
    return arr.get() if hasattr(arr, "get") else arr


K_VALUES = [20, 40, 60, 80, 100]
summary_rows = []

for k in K_VALUES:
    clusterer = KMeans(n_clusters=k, random_state=42, n_init=10 if not GPU else 1)
    labels = clusterer.fit_predict(X_svd)
    labels_cpu = labels.get() if hasattr(labels, "get") else (
        labels.to_numpy() if hasattr(labels, "to_numpy") else labels
    )
    df["cluster_k"] = labels_cpu
    tick(f"k={k}: KMeans done")

    # Best cluster per class, same methodology as before
    for cls in ALL_CLASSES:
        baseline_rate = baseline_dist.get(cls, 0)
        cluster_class_rate = df.groupby("cluster_k")["future_subcat"].apply(lambda s: (s == cls).mean())
        cluster_sizes = df.groupby("cluster_k").size()
        valid = cluster_sizes[cluster_sizes >= 30].index
        if len(valid) == 0:
            continue
        best_cluster = cluster_class_rate[valid].idxmax()
        best_rate = cluster_class_rate[best_cluster]
        best_size = cluster_sizes[best_cluster]
        summary_rows.append({
            "k": k, "class": cls, "best_cluster": best_cluster,
            "cluster_size": int(best_size), "purity": best_rate,
            "lift": best_rate - baseline_rate,
        })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("plots/tune_k_results.csv", index=False)
tick("Saved plots/tune_k_results.csv")

print("\nLift by class and k (pivot table):")
pivot = summary_df.pivot(index="class", columns="k", values="lift")
print(pivot.round(3))

print("\nBest k per class (by lift):")
best_k_per_class = summary_df.loc[summary_df.groupby("class")["lift"].idxmax()]
print(best_k_per_class[["class", "k", "lift", "purity", "cluster_size"]].to_string(index=False))
