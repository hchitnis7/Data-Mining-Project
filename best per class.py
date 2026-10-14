"""
Instead of just the overall top-5 clusters by lift, find the BEST
cluster for EACH class (including the underrepresented astro-ph.IM
and astro-ph.EP), and show top terms + examples for those too.

This checks whether IM/EP signal exists but was just outranked by
bigger CO/SR clusters, vs genuinely absent.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_parquet("astroph_clustered_full_gpu.parquet")
results_df = pd.read_csv("cluster_validation_results_full_gpu.csv")

ALL_CLASSES = ["astro-ph.CO", "astro-ph.SR", "astro-ph.GA",
               "astro-ph.IM", "astro-ph.HE", "astro-ph.EP"]
baseline_dist = df["future_subcat"].value_counts(normalize=True)

vectorizer = TfidfVectorizer(max_features=8000, stop_words="english", min_df=5, max_df=0.5)
X = vectorizer.fit_transform(df["clean_abstract"])
terms = np.array(vectorizer.get_feature_names_out())

TOP_N_TERMS = 15
TOP_N_EXAMPLES = 3

print("="*70)
print("BEST CLUSTER FOR EACH CLASS (not just overall top-5)")
print("="*70)

for cls in ALL_CLASSES:
    baseline_rate = baseline_dist.get(cls, 0)

    # For each cluster, what fraction is THIS specific class (not just
    # the cluster's own majority class)?
    cluster_class_rate = df.groupby("cluster")["future_subcat"].apply(
        lambda s: (s == cls).mean()
    )
    cluster_sizes = df.groupby("cluster").size()

    # Only consider clusters with a reasonable size (avoid tiny noisy clusters)
    valid = cluster_sizes[cluster_sizes >= 50].index
    best_cluster = cluster_class_rate[valid].idxmax()
    best_rate = cluster_class_rate[best_cluster]
    best_size = cluster_sizes[best_cluster]

    print(f"\n--- {cls} (baseline rate: {baseline_rate:.1%}) ---")
    print(f"Best cluster: {best_cluster}  |  size={best_size}  |  "
          f"rate of {cls}={best_rate:.1%}  |  lift={best_rate - baseline_rate:+.1%}")

    mask = df["cluster"] == best_cluster
    cluster_tfidf_mean = X[mask.values].mean(axis=0).A1
    top_term_idx = cluster_tfidf_mean.argsort()[::-1][:TOP_N_TERMS]
    print(f"Top terms: {', '.join(terms[top_term_idx])}")

    # Full class breakdown of this cluster, not just the target class
    print("Full class breakdown of this cluster:")
    print(df[mask]["future_subcat"].value_counts(normalize=True).round(3))

    examples = df[mask & (df["future_subcat"] == cls)].sample(
        min(TOP_N_EXAMPLES, (mask & (df["future_subcat"] == cls)).sum()), random_state=1
    )
    print("Example abstracts (actually labeled this class):")
    for _, ex in examples.iterrows():
        print(f"  {ex['abstract'][:200]}...")