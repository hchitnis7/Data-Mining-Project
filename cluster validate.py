"""
Step 3: Cluster pre-2009 astro-ph abstracts, then check whether clusters
are "pure" with respect to future_subcat (the subcategory each paper's
authors went on to specialize in, post-2009) - beyond what you'd expect
from the overall class distribution alone.
"""

import pandas as pd
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

t0 = time.time()
def tick(msg):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

# Set to a number (e.g. 15000) for a fast first pass, or None for full run
SUBSAMPLE = 15000

df = pd.read_parquet("astroph_pre_with_future_labels.parquet")
tick(f"Loaded parquet: {len(df):,} rows")

# Only keep papers where we actually have a resolvable future label
df = df.dropna(subset=["future_subcat"]).copy()
tick(f"After dropping unresolved: {len(df):,} papers")

if SUBSAMPLE is not None and len(df) > SUBSAMPLE:
    df = df.sample(n=SUBSAMPLE, random_state=42).copy()
    tick(f"Subsampled down to {len(df):,} papers for fast iteration")

baseline_dist = df["future_subcat"].value_counts(normalize=True)
print("\nBaseline class distribution (what 'random guessing' looks like):")
print(baseline_dist)


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\$.*?\$", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


df["clean_abstract"] = df["abstract"].apply(clean_text)
tick("Text cleaning done")

vectorizer = TfidfVectorizer(
    max_features=8000,
    stop_words="english",
    min_df=5,
    max_df=0.5,
)
X = vectorizer.fit_transform(df["clean_abstract"])
tick(f"TF-IDF done, shape {X.shape}")

svd = TruncatedSVD(n_components=300, random_state=42)
X_svd = svd.fit_transform(X)
tick(f"SVD done, explained variance = {svd.explained_variance_ratio_.sum():.3f}")

clusterer = KMeans(n_clusters=20, random_state=42, n_init=10)
df["cluster"] = clusterer.fit_predict(X_svd)
tick("KMeans done")
print("\nCluster sizes:")
print(df["cluster"].value_counts())

# ---------------------------------------------------------------
# Validation: for each cluster, what's its majority future_subcat,
# and what fraction of the cluster agrees with that majority
# ("purity")? Compare against the baseline majority-class rate
# (what you'd get just by always guessing the single most common
# class overall, with no clustering at all).
# ---------------------------------------------------------------
baseline_majority_rate = baseline_dist.max()
print(f"\nBaseline majority-class rate (guessing most common class always): "
      f"{baseline_majority_rate:.1%}")

results = []
for cluster_id, group in df.groupby("cluster"):
    if cluster_id == -1:
        continue  # safety guard, KMeans shouldn't produce this
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

if not results:
    raise RuntimeError("No valid clusters produced - check clustering step above.")

results_df = pd.DataFrame(results).sort_values("lift_over_baseline", ascending=False)
print("\nCluster validation results:")
print(results_df.to_string(index=False))

# Flag clusters that meaningfully beat baseline
strong_clusters = results_df[results_df["lift_over_baseline"] > 0.1]
print(f"\n{len(strong_clusters)} cluster(s) beat baseline purity by >10 points - "
      f"these are your candidate 'early signal' clusters.")

# 2D visualization for the report
df["svd_x"] = X_svd[:, 0]
df["svd_y"] = X_svd[:, 1]

plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x="svd_x", y="svd_y", hue="future_subcat",
                 s=8, alpha=0.5, palette="tab10")
plt.title("Pre-2009 astro-ph abstracts, colored by authors' later specialization")
plt.tight_layout()
plt.savefig("fig7_svd_by_future_subcat.png", dpi=150)
plt.close()

df.to_parquet("astroph_clustered.parquet", index=False)
results_df.to_csv("cluster_validation_results.csv", index=False)
print("\nSaved astroph_clustered.parquet and cluster_validation_results.csv")