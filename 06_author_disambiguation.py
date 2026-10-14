"""
Task 1: Author disambiguation.

Current matching uses last-name + first-initial ("smith_j"), which
collides distinct people. This rebuilds the author key using the full
first name when available (authors_parsed gives [last, first, suffix]),
falling back to initial only when the first name is itself just an
initial. This doesn't fully solve disambiguation (that needs institution/
co-author-graph matching) but meaningfully cuts collisions cheaply.

The point of doing this is NOT just a cleaner label - it's to check
whether the original cs.SY-style "author votes" labeling was distorted
by name collisions. So this script doesn't stop at relabeling: it
re-runs the SAME TF-IDF + SVD + KMeans pipeline on the v2 labels and
puts the v1 vs v2 purity/lift numbers side by side. If the numbers barely
move, disambiguation didn't matter here. If they move a lot, the original
result was partly an artifact of name collisions - also worth reporting.

Requires astroph_working_set.parquet to have been saved WITH the
authors_parsed column (01_load_filter_astroph.py was updated to do this -
if your saved file predates that, re-run the loader first).
"""

import re
import time
import pandas as pd
from collections import defaultdict

t0 = time.time()
def tick(msg):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

SUBCATS = {"astro-ph.CO", "astro-ph.EP", "astro-ph.GA",
           "astro-ph.HE", "astro-ph.IM", "astro-ph.SR"}

df = pd.read_parquet("astroph_working_set.parquet")

if "authors_parsed" not in df.columns:
    raise RuntimeError(
        "authors_parsed not found in astroph_working_set.parquet. "
        "Re-run the updated 01_load_filter_astroph.py (which now saves this "
        "column) before running this script - disambiguation can't work "
        "without the raw author names."
    )


def better_author_key(author_row):
    """
    author_row is [last, first, suffix] from authors_parsed.
    Use full first name (lowercased, stripped) when it looks like a real
    name (len > 1 and not just an initial like 'J' or 'J.'), else fall
    back to initial-only. This reduces the "Smith, J." collision problem
    without needing external author-ID resolution.
    """
    last = (author_row[0] or "").strip().lower()
    first = (author_row[1] or "").strip().lower().rstrip(".")
    if len(first) > 1:
        return f"{last}_{first}"
    return f"{last}_{first[:1]}"


df["all_author_keys_v2"] = df["authors_parsed"].apply(
    lambda authors: [better_author_key(a) for a in authors] if authors is not None else []
)
tick("Rebuilt author keys with full first names")

pre = df[df["pre_split"]].copy()
post = df[~df["pre_split"]].copy()

# Rebuild the post-2009 author -> subcategory lookup with the improved keys
author_subcats_v2 = defaultdict(set)
for _, row in post.iterrows():
    primary_cat = row["categories"].split()[0]
    if primary_cat not in SUBCATS:
        continue
    for author_key in row["all_author_keys_v2"]:
        if author_key:
            author_subcats_v2[author_key].add(primary_cat)

print(f"v2 lookup covers {len(author_subcats_v2):,} distinct authors")


def dominant_subcat_v2(all_author_keys):
    votes = defaultdict(int)
    for key in all_author_keys:
        for sc in author_subcats_v2.get(key, set()):
            votes[sc] += 1
    if not votes:
        return None
    return max(votes, key=votes.get)


pre["future_subcat_v2"] = pre["all_author_keys_v2"].apply(dominant_subcat_v2)
pre_v2 = pre.dropna(subset=["future_subcat_v2"]).copy().reset_index(drop=True)
tick(f"v2 resolvable pre-split papers: {len(pre_v2):,}")

print("\nfuture_subcat_v2 distribution:")
print(pre_v2["future_subcat_v2"].value_counts())

# Compare to the original v1 labels, where both exist, to see how much
# the relabeling itself changed before we even get to clustering
if "future_subcat" in pre.columns:
    both = pre.dropna(subset=["future_subcat", "future_subcat_v2"])
    agree = (both["future_subcat"] == both["future_subcat_v2"]).mean()
    print(f"\nAgreement between v1 (initial-only) and v2 (full-name) labels: {agree:.1%}")
    changed = both[both["future_subcat"] != both["future_subcat_v2"]]
    print(f"{len(changed):,} / {len(both):,} papers changed future_subcat ({len(changed) / len(both):.1%})")

pre_v2.to_parquet("astroph_pre_with_future_labels_v2.parquet", index=False)
tick("Saved astroph_pre_with_future_labels_v2.parquet")

# ---------------------------------------------------------------------
# Now the part that actually answers "did disambiguation matter?":
# re-run the same TF-IDF + SVD + KMeans pipeline on the v2 labels and
# compare purity/lift directly against the original (v1) results.
# ---------------------------------------------------------------------
try:
    import cudf
    from cuml.feature_extraction.text import TfidfVectorizer
    from cuml.decomposition import TruncatedSVD
    from cuml.cluster import KMeans
    GPU = True
    tick("Using RAPIDS cuML (GPU) for re-clustering")
except ImportError:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.cluster import KMeans
    GPU = False
    tick("cuML not available, falling back to sklearn (CPU) for re-clustering")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"\$.*?\$", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


pre_v2["clean_abstract"] = pre_v2["abstract"].apply(clean_text)
tick("Text cleaned")

if GPU:
    gdf = cudf.from_pandas(pre_v2[["clean_abstract"]])
    vectorizer = TfidfVectorizer(max_features=8000, stop_words="english", min_df=5, max_df=0.5)
    X = vectorizer.fit_transform(gdf["clean_abstract"])
    X_dense = X.toarray()
    svd = TruncatedSVD(n_components=300, random_state=42)
    X_svd = svd.fit_transform(X_dense)
    clusterer = KMeans(n_clusters=40, random_state=42)
    labels = clusterer.fit_predict(X_svd)
    labels_cpu = labels.get() if hasattr(labels, "get") else (
        labels.to_numpy() if hasattr(labels, "to_numpy") else labels
    )
else:
    vectorizer = TfidfVectorizer(max_features=8000, stop_words="english", min_df=5, max_df=0.5)
    X = vectorizer.fit_transform(pre_v2["clean_abstract"])
    svd = TruncatedSVD(n_components=300, random_state=42)
    X_svd = svd.fit_transform(X)
    clusterer = KMeans(n_clusters=40, random_state=42, n_init=10)
    labels_cpu = clusterer.fit_predict(X_svd)

pre_v2["cluster"] = labels_cpu
tick("Re-clustering on v2 labels done")

baseline_dist_v2 = pre_v2["future_subcat_v2"].value_counts(normalize=True)
ALL_CLASSES = sorted(SUBCATS)

v2_results = []
for cls in ALL_CLASSES:
    baseline_rate = baseline_dist_v2.get(cls, 0)
    cluster_class_rate = pre_v2.groupby("cluster")["future_subcat_v2"].apply(lambda s: (s == cls).mean())
    cluster_sizes = pre_v2.groupby("cluster").size()
    valid = cluster_sizes[cluster_sizes >= 30].index
    if len(valid) == 0:
        continue
    best_cluster = cluster_class_rate[valid].idxmax()
    best_rate = cluster_class_rate[best_cluster]
    v2_results.append({
        "class": cls, "baseline_v2": baseline_rate, "best_cluster_v2": best_cluster,
        "cluster_size_v2": int(cluster_sizes[best_cluster]), "purity_v2": best_rate,
        "lift_v2": best_rate - baseline_rate,
    })

v2_df = pd.DataFrame(v2_results).sort_values("lift_v2", ascending=False)
print("\nv2 (disambiguated) best-cluster-per-class results:")
print(v2_df.to_string(index=False))
v2_df.to_csv("plots/author_disambig_v2_results.csv", index=False)

# Side-by-side against the original v1 master results, if available
v1_path = "/teamspace/studios/this_studio/plots/tableB9_master_results.csv"
import os
if os.path.exists(v1_path):
    v1_df = pd.read_csv(v1_path)
    v1_df["lift_v1"] = v1_df["lift"].astype(str).str.replace("+", "", regex=False).str.rstrip("%").astype(float) / 100
    comparison = v1_df[["class", "lift_v1"]].merge(v2_df[["class", "lift_v2"]], on="class", how="outer")
    comparison["delta"] = comparison["lift_v2"] - comparison["lift_v1"]
    print("\n" + "=" * 70)
    print("v1 (initial-only keys) vs v2 (disambiguated keys) - lift comparison")
    print("=" * 70)
    print(comparison.round(3).to_string(index=False))
    comparison.to_csv("plots/author_disambig_v1_vs_v2_comparison.csv", index=False)
    print(
        "\nIf delta is small across classes, disambiguation did not "
        "meaningfully change the result - the original method was already "
        "robust to the initial-only collision problem. If delta is large "
        "(especially for common surnames), report the v2 numbers as the "
        "more trustworthy ones."
    )
else:
    print(f"\n({v1_path} not found - skipping direct v1-vs-v2 comparison, "
          "v2 standalone results are still saved above.)")

tick("Done")