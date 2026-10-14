"""
Task 4: Richer text representation.

Compares TF-IDF (current method) against sentence embeddings (SBERT)
for the astro-ph pre-2009 clustering task, to see whether a richer
representation produces cleaner cluster separation.

Needs: pip install sentence-transformers
GPU strongly recommended for encoding (CPU works but is much slower on
~100K abstracts).
"""

import os
import re
import time
import numpy as np
import pandas as pd

os.makedirs("plots", exist_ok=True)

t0 = time.time()
def tick(msg):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("Run: pip install sentence-transformers")

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

df = pd.read_parquet("astroph_pre_with_future_labels.parquet")
df = df.dropna(subset=["future_subcat"]).copy().reset_index(drop=True)
tick(f"Working set: {len(df):,} papers")

ALL_CLASSES = ["astro-ph.CO", "astro-ph.SR", "astro-ph.GA",
               "astro-ph.IM", "astro-ph.HE", "astro-ph.EP"]
baseline_dist = df["future_subcat"].value_counts(normalize=True)


def clean_text(text):
    text = text.lower()
    text = re.sub(r"\$.*?\$", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# SBERT wants more natural text than the aggressively stripped TF-IDF
# version - keep punctuation/case for embeddings, just collapse whitespace
# and drop raw LaTeX math blocks.
df["sbert_text"] = df["abstract"].apply(clean_text)

# all-MiniLM-L6-v2: fast, strong general-purpose baseline. Swap for
# allenai-specter or a SciBERT-based sentence model if you want something
# tuned for scientific text specifically.
model = SentenceTransformer("all-MiniLM-L6-v2")
tick("Model loaded")

embeddings = model.encode(
    df["sbert_text"].tolist(),
    batch_size=128,
    show_progress_bar=True,
    convert_to_numpy=True,
)
tick(f"Encoded {embeddings.shape[0]:,} abstracts into {embeddings.shape[1]}-dim vectors")

np.save("sbert_embeddings.npy", embeddings)

# Cluster directly on embeddings (already dense, no SVD needed, but a
# light PCA keeps KMeans distances well-behaved in very high dimensions)
pca = PCA(n_components=100, random_state=42)
emb_reduced = pca.fit_transform(embeddings)
tick(f"PCA explained variance (100 comps): {pca.explained_variance_ratio_.sum():.3f}")

k = 40
clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
df["cluster_sbert"] = clusterer.fit_predict(emb_reduced)
tick("KMeans on SBERT embeddings done")

results = []
for cls in ALL_CLASSES:
    baseline_rate = baseline_dist.get(cls, 0)
    cluster_class_rate = df.groupby("cluster_sbert")["future_subcat"].apply(lambda s: (s == cls).mean())
    cluster_sizes = df.groupby("cluster_sbert").size()
    valid = cluster_sizes[cluster_sizes >= 50].index
    best_cluster = cluster_class_rate[valid].idxmax()
    best_rate = cluster_class_rate[best_cluster]
    results.append({
        "class": cls, "baseline": baseline_rate, "best_cluster": best_cluster,
        "cluster_size": int(cluster_sizes[best_cluster]), "purity": best_rate,
        "lift": best_rate - baseline_rate,
    })

sbert_results = pd.DataFrame(results).sort_values("lift", ascending=False)
print("\nSBERT-based results (best cluster per class):")
print(sbert_results.to_string(index=False))

sbert_results.to_csv("plots/sbert_vs_tfidf_results.csv", index=False)

# Compare directly against the saved TF-IDF results if available
tfidf_path = "tableB9_master_results.csv"
if os.path.exists(tfidf_path):
    tfidf_results = pd.read_csv(tfidf_path)
    print("\nSide-by-side comparison (TF-IDF lift vs SBERT lift):")
    tfidf_results["class"] = tfidf_results["class"]
    tfidf_lift = tfidf_results.set_index("class")["lift"].str.rstrip("%").astype(float) / 100
    sbert_lift = sbert_results.set_index("class")["lift"]
    comparison = pd.DataFrame({"tfidf_lift": tfidf_lift, "sbert_lift": sbert_lift})
    comparison["winner"] = np.where(comparison["sbert_lift"] > comparison["tfidf_lift"], "SBERT", "TF-IDF")
    print(comparison.round(3))
    comparison.to_csv("plots/tfidf_vs_sbert_comparison.csv")

tick("Done")
