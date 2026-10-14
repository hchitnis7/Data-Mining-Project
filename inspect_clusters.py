"""
Inspect the top distinguishing terms for your strongest clusters, and
pull a few example abstracts, so you can explain *what* each cluster
is actually about in your report / presentation.

Run this right after 03b_cluster_validate.py in the same session
(reuses vectorizer, df, etc.) - or re-load from the saved parquet.
"""

import pandas as pd
import numpy as np

# If running standalone, reload what you need:
df = pd.read_parquet("astroph_clustered_full_gpu.parquet")
results_df = pd.read_csv("cluster_validation_results_full_gpu.csv")

TOP_N_CLUSTERS = 5
TOP_N_TERMS = 15
TOP_N_EXAMPLES = 3

top_clusters = results_df.sort_values("lift_over_baseline", ascending=False).head(TOP_N_CLUSTERS)

from sklearn.feature_extraction.text import TfidfVectorizer

# Re-fit vectorizer on this df's clean_abstract to get term names
# (quick re-fit since we need the vocabulary - cheap at this size)
vectorizer = TfidfVectorizer(max_features=8000, stop_words="english", min_df=5, max_df=0.5)
X = vectorizer.fit_transform(df["clean_abstract"])
terms = np.array(vectorizer.get_feature_names_out())

for _, row in top_clusters.iterrows():
    cid = row["cluster"]
    mask = df["cluster"] == cid
    cluster_tfidf_mean = X[mask.values].mean(axis=0).A1
    top_term_idx = cluster_tfidf_mean.argsort()[::-1][:TOP_N_TERMS]

    print(f"\n{'='*70}")
    print(f"Cluster {cid}  |  size={row['size']}  |  majority_class={row['majority_class']}  "
          f"|  purity={row['purity']:.1%}  |  lift={row['lift_over_baseline']:+.1%}")
    print(f"Top terms: {', '.join(terms[top_term_idx])}")
    print("\nExample abstracts:")
    examples = df[mask].sample(min(TOP_N_EXAMPLES, mask.sum()), random_state=1)
    for _, ex in examples.iterrows():
        print(f"  [{ex['future_subcat']}] {ex['abstract'][:200]}...")