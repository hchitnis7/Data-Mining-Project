"""
Full visualization suite for the report/presentation.
Organized in independent sections - comment out any section that
errors (e.g. missing wordcloud/plotly) without affecting the rest.

Run after 03c_cluster_gpu_full.py has produced:
  astroph_clustered_full_gpu.parquet
  cluster_validation_results_full_gpu.csv
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

sns.set_theme(style="whitegrid")

OUTDIR = "plots"
os.makedirs(OUTDIR, exist_ok=True)

def outpath(filename):
    return os.path.join(OUTDIR, filename)

df = pd.read_parquet("astroph_clustered_full_gpu.parquet")
results_df = pd.read_csv("cluster_validation_results_full_gpu.csv")

ALL_CLASSES = ["astro-ph.CO", "astro-ph.SR", "astro-ph.GA",
               "astro-ph.IM", "astro-ph.HE", "astro-ph.EP"]
baseline_dist = df["future_subcat"].value_counts(normalize=True)

vectorizer = TfidfVectorizer(max_features=8000, stop_words="english", min_df=5, max_df=0.5)
X = vectorizer.fit_transform(df["clean_abstract"])
terms = np.array(vectorizer.get_feature_names_out())

print("Loaded data, starting figure generation...\n")

# =================================================================
# SECTION A: remaining EDA figures
# =================================================================

# --- A6: Word cloud per class (raw class vocabulary, not cluster-specific) ---
try:
    from wordcloud import WordCloud

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, cls in zip(axes.flat, ALL_CLASSES):
        text = " ".join(df[df["future_subcat"] == cls]["clean_abstract"].sample(
            min(2000, (df["future_subcat"] == cls).sum()), random_state=1))
        wc = WordCloud(width=500, height=350, background_color="white",
                        max_words=60).generate(text)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(cls)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(outpath("figA6_wordcloud_by_class.png"), dpi=150)
    plt.close()
    print("Saved figA6_wordcloud_by_class.png")
except ImportError:
    print("SKIPPED figA6 - wordcloud not installed (pip install wordcloud)")

# --- A8: Papers per year by future_subcat (stacked area) ---
df["year"] = df["submitted"].dt.year
yearly_by_class = df.groupby(["year", "future_subcat"]).size().unstack(fill_value=0)
plt.figure(figsize=(12, 6))
yearly_by_class.plot(kind="area", stacked=True, ax=plt.gca(), colormap="tab10")
plt.title("Papers per year by eventual (future) subcategory")
plt.xlabel("Year")
plt.ylabel("Number of papers")
plt.legend(title="Future subcategory", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(outpath("figA8_papers_per_year_by_class.png"), dpi=150)
plt.close()
print("Saved figA8_papers_per_year_by_class.png")

# =================================================================
# SECTION B: tables
# =================================================================

# --- B9: Master results table ---
master_rows = []
for cls in ALL_CLASSES:
    baseline_rate = baseline_dist.get(cls, 0)
    cluster_class_rate = df.groupby("cluster")["future_subcat"].apply(lambda s: (s == cls).mean())
    cluster_sizes = df.groupby("cluster").size()
    valid = cluster_sizes[cluster_sizes >= 50].index
    best_cluster = cluster_class_rate[valid].idxmax()
    best_rate = cluster_class_rate[best_cluster]
    best_size = cluster_sizes[best_cluster]

    mask = df["cluster"] == best_cluster
    cluster_tfidf_mean = X[mask.values].mean(axis=0).A1
    top_term_idx = cluster_tfidf_mean.argsort()[::-1][:5]
    top_terms = ", ".join(terms[top_term_idx])

    master_rows.append({
        "class": cls,
        "baseline_rate": f"{baseline_rate:.1%}",
        "best_cluster": best_cluster,
        "cluster_size": best_size,
        "cluster_purity_for_class": f"{best_rate:.1%}",
        "lift": f"{best_rate - baseline_rate:+.1%}",
        "top_terms": top_terms,
    })

master_table = pd.DataFrame(master_rows)
master_table.to_csv(outpath("tableB9_master_results.csv"), index=False)
print("\nSaved tableB9_master_results.csv")
print(master_table.to_string(index=False))

# --- B10/C13: Confusion-style heatmap (clusters x classes) ---
# Restrict to the "strong" clusters (reasonable size, meaningfully above
# baseline for their majority class) so the heatmap stays readable.
strong_cluster_ids = results_df[
    (results_df["lift_over_baseline"] > 0.05) & (results_df["size"] >= 50)
].sort_values("lift_over_baseline", ascending=False)["cluster"].tolist()

conf_matrix = (
    df[df["cluster"].isin(strong_cluster_ids)]
    .groupby(["cluster", "future_subcat"]).size()
    .unstack(fill_value=0)
)
conf_matrix_pct = conf_matrix.div(conf_matrix.sum(axis=1), axis=0)
conf_matrix_pct = conf_matrix_pct.loc[strong_cluster_ids]  # keep sorted by lift
conf_matrix_pct.to_csv(outpath("tableB10_confusion_matrix.csv"))

plt.figure(figsize=(10, max(6, 0.4 * len(conf_matrix_pct))))
sns.heatmap(conf_matrix_pct, annot=True, fmt=".0%", cmap="YlOrRd", cbar_kws={"label": "% of cluster"})
plt.title("Cluster composition: % of each cluster by future subcategory")
plt.xlabel("Future subcategory")
plt.ylabel("Cluster ID")
plt.tight_layout()
plt.savefig(outpath("figC13_confusion_heatmap.png"), dpi=150)
plt.close()
print("Saved figC13_confusion_heatmap.png and tableB10_confusion_matrix.csv")

# --- B11: Method vs baseline comparison table ---
method_vs_baseline = pd.DataFrame({
    "approach": ["Random guessing (uniform)", "Always guess majority class",
                 "Best cluster per class (this method)"],
    "accuracy_or_rate": [
        f"{1/len(ALL_CLASSES):.1%}",
        f"{baseline_dist.max():.1%}",
        f"{master_table['cluster_purity_for_class'].str.rstrip('%').astype(float).mean():.1f}% (avg across classes)",
    ]
})
method_vs_baseline.to_csv(outpath("tableB11_method_vs_baseline.csv"), index=False)
print("\nSaved tableB11_method_vs_baseline.csv")
print(method_vs_baseline.to_string(index=False))

# =================================================================
# SECTION C: cluster visualizations
# =================================================================

# --- C14: Cluster size vs purity scatter ---
plot_data = results_df.copy()
plt.figure(figsize=(10, 7))
sns.scatterplot(data=plot_data, x="size", y="purity", hue="majority_class",
                 s=100, palette="tab10")
plt.axhline(baseline_dist.max(), color="gray", linestyle="--",
            label=f"baseline majority rate ({baseline_dist.max():.1%})")
plt.title("Cluster size vs. purity, colored by majority class")
plt.xlabel("Cluster size (papers)")
plt.ylabel("Purity (fraction = majority class)")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(outpath("figC14_size_vs_purity.png"), dpi=150)
plt.close()
print("Saved figC14_size_vs_purity.png")

# --- C15: Stacked bar, full class composition for top clusters ---
top_n = 8
top_clusters_for_bar = results_df.sort_values("lift_over_baseline", ascending=False).head(top_n)["cluster"].tolist()
comp = (
    df[df["cluster"].isin(top_clusters_for_bar)]
    .groupby(["cluster", "future_subcat"]).size()
    .unstack(fill_value=0)
)
comp_pct = comp.div(comp.sum(axis=1), axis=0).loc[top_clusters_for_bar]

plt.figure(figsize=(10, 6))
comp_pct.plot(kind="barh", stacked=True, ax=plt.gca(), colormap="tab10")
plt.title(f"Full class composition, top {top_n} clusters by lift")
plt.xlabel("Fraction of cluster")
plt.ylabel("Cluster ID")
plt.legend(title="Future subcategory", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(outpath("figC15_stacked_composition.png"), dpi=150)
plt.close()
print("Saved figC15_stacked_composition.png")

# --- C17b: Static SVD scatter, ground truth vs clusters (guaranteed to run,
# no optional dependencies) - the core cluster visualization. ---
from sklearn.decomposition import TruncatedSVD

if "svd_x" not in df.columns or "svd_y" not in df.columns:
    svd_2d = TruncatedSVD(n_components=2, random_state=42)
    X_2d = svd_2d.fit_transform(X)
    df["svd_x"] = X_2d[:, 0]
    df["svd_y"] = X_2d[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
sns.scatterplot(data=df, x="svd_x", y="svd_y", hue="future_subcat",
                 s=4, alpha=0.4, palette="tab10", ax=axes[0])
axes[0].set_title("Colored by authors' actual future subcategory (ground truth)")
sns.scatterplot(data=df, x="svd_x", y="svd_y", hue="cluster",
                 s=4, alpha=0.4, palette="tab20", legend=False, ax=axes[1])
axes[1].set_title("Colored by k-means cluster assignment")
plt.tight_layout()
plt.savefig(outpath("figC17b_svd_ground_truth_vs_clusters.png"), dpi=150)
plt.close()
print("Saved figC17b_svd_ground_truth_vs_clusters.png")

# --- C16: Word clouds per top cluster (using cluster-specific TF-IDF terms) ---
try:
    from wordcloud import WordCloud

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for ax, cid in zip(axes.flat, top_clusters_for_bar):
        mask = (df["cluster"] == cid).values
        cluster_tfidf_mean = X[mask].mean(axis=0).A1
        freqs = {terms[i]: cluster_tfidf_mean[i] for i in cluster_tfidf_mean.argsort()[::-1][:50]}
        wc = WordCloud(width=400, height=300, background_color="white").generate_from_frequencies(freqs)
        ax.imshow(wc, interpolation="bilinear")
        row = results_df[results_df["cluster"] == cid].iloc[0]
        ax.set_title(f"Cluster {cid} ({row['majority_class']}, {row['purity']:.0%} pure)")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(outpath("figC16_wordclouds_top_clusters.png"), dpi=150)
    plt.close()
    print("Saved figC16_wordclouds_top_clusters.png")
except ImportError:
    print("SKIPPED figC16 - wordcloud not installed (pip install wordcloud)")

# --- C17: Interactive SVD scatter (plotly, optional) ---
try:
    import plotly.express as px
    from sklearn.decomposition import TruncatedSVD

    # svd_x/svd_y may not be saved in the parquet (e.g. if it was produced
    # before this column was added upstream) - compute them here directly
    # from X, which we already have loaded, rather than depending on it.
    if "svd_x" not in df.columns or "svd_y" not in df.columns:
        print("svd_x/svd_y not found in parquet - computing 2D projection now...")
        svd_2d = TruncatedSVD(n_components=2, random_state=42)
        X_2d = svd_2d.fit_transform(X)
        df["svd_x"] = X_2d[:, 0]
        df["svd_y"] = X_2d[:, 1]
        print("Computed svd_x/svd_y.")

    fig = px.scatter(
        df, x="svd_x", y="svd_y", color="future_subcat",
        hover_data=["cluster", "abstract"],
        title="Interactive SVD projection (hover for abstract preview)",
        opacity=0.5,
    )
    fig.write_html(outpath("figC17_interactive_scatter.html"))
    print("Saved figC17_interactive_scatter.html (open in browser)")
except ImportError:
    print("SKIPPED figC17 - plotly not installed (pip install plotly)")

print(f"\nAll done. Check the '{OUTDIR}/' directory for all figures/tables.")