"""
Step 3 (GPU version, full dataset): Cluster pre-2009 astro-ph abstracts
using RAPIDS cuML on GPU. Requires cudf + cuml installed and a CUDA GPU
available (check with nvidia-smi before running).

If cuml/cudf aren't available, this will raise ImportError immediately
with a clear message rather than silently falling back to CPU.
"""

import pandas as pd
import re
import time

t0 = time.time()
def tick(msg):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

try:
    import cudf
    from cuml.feature_extraction.text import TfidfVectorizer
    from cuml.decomposition import TruncatedSVD
    from cuml.cluster import KMeans
    tick("RAPIDS cuML imported successfully - running on GPU")
except ImportError as e:
    raise ImportError(
        "cudf/cuml not available. Install RAPIDS first "
        "(pip install cudf-cu12 cuml-cu12 --extra-index-url=https://pypi.nvidia.com, "
        "matching your CUDA version), or use the CPU version instead."
    ) from e

df = pd.read_parquet("astroph_pre_with_future_labels.parquet")
tick(f"Loaded parquet: {len(df):,} rows")

df = df.dropna(subset=["future_subcat"]).copy()
df = df.reset_index(drop=True)  # cuML's vectorizer needs a gap-free index
tick(f"After dropping unresolved: {len(df):,} papers (full dataset, no subsampling)")

baseline_dist = df["future_subcat"].value_counts(normalize=True)
print("\nBaseline class distribution:")
print(baseline_dist)


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\$.*?\$", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


df["clean_abstract"] = df["abstract"].apply(clean_text)
tick("Text cleaning done")

# Move to GPU dataframe
gdf = cudf.from_pandas(df[["clean_abstract"]])
tick("Moved to GPU (cudf)")

vectorizer = TfidfVectorizer(max_features=8000, stop_words="english", min_df=5, max_df=0.5)
X = vectorizer.fit_transform(gdf["clean_abstract"])
tick(f"TF-IDF done (GPU), shape {X.shape}")

# cuML's TruncatedSVD requires dense input, unlike sklearn's version which
# accepts sparse. ~103K x 8000 float32 is ~3.3GB - should fit on most GPUs,
# but watch for an out-of-memory error here if your GPU has limited VRAM.
X_dense = X.toarray()
tick(f"Converted TF-IDF to dense, shape {X_dense.shape}")

svd = TruncatedSVD(n_components=300, random_state=42)
X_svd = svd.fit_transform(X_dense)
tick(f"SVD done (GPU), explained variance = {float(svd.explained_variance_ratio_.sum()):.3f}")

clusterer = KMeans(n_clusters=40, random_state=42)
cluster_labels = clusterer.fit_predict(X_svd)
tick("KMeans done (GPU)")

# Bring results back to pandas/CPU for the rest of the analysis
# Bring results back to pandas/CPU for the rest of the analysis.
# cluster_labels may be a cupy array (.get()), a cudf Series (.to_numpy()),
# or already a plain array - handle all three.
if hasattr(cluster_labels, "get"):
    cluster_labels_cpu = cluster_labels.get()
elif hasattr(cluster_labels, "to_numpy"):
    cluster_labels_cpu = cluster_labels.to_numpy()
else:
    cluster_labels_cpu = cluster_labels
df["cluster"] = cluster_labels_cpu
print("\nCluster sizes:")
print(df["cluster"].value_counts())

baseline_majority_rate = baseline_dist.max()
print(f"\nBaseline majority-class rate: {baseline_majority_rate:.1%}")

results = []
for cluster_id, group in df.groupby("cluster"):
    counts = group["future_subcat"].value_counts()
    majority_class = counts.idxmax()
    purity = counts.max() / len(group)
    results.append({
        "cluster": cluster_id,
        "size": len(group),
        "majority_class": majority_class,
        "purity": purity,
        "lift_over_baseline": purity - baseline_majority_rate,
    })

results_df = pd.DataFrame(results).sort_values("lift_over_baseline", ascending=False)
print("\nCluster validation results:")
print(results_df.to_string(index=False))

strong_clusters = results_df[results_df["lift_over_baseline"] > 0.1]
print(f"\n{len(strong_clusters)} cluster(s) beat baseline purity by >10 points.")

df.to_parquet("astroph_clustered_full_gpu.parquet", index=False)
results_df.to_csv("cluster_validation_results_full_gpu.csv", index=False)
tick("Saved results")