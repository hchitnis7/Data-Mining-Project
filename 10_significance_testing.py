"""
Task 5: Statistical significance testing.

Formalizes the "lift over baseline" results with a permutation test:
shuffle the future_subcat labels many times, recompute each cluster's
best-class purity under the null (no real relationship between cluster
assignment and future subfield), and see how often the shuffled purity
matches or exceeds the real result. This gives a p-value for each
cluster's lift, rather than just reporting the raw percentage.

Also adds bootstrap confidence intervals on purity for the headline
clusters, which is easy to quote in the report/presentation
("77.7% purity, 95% CI [x%, y%]").

Run after 03c_cluster_gpu_full.py / 05_full_visualization_suite.py.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

df = pd.read_parquet("astroph_clustered_full_gpu.parquet")
results_df = pd.read_csv("cluster_validation_results_full_gpu.csv")

N_PERMUTATIONS = 2000
N_BOOTSTRAP = 2000

# Focus on the headline clusters (top by lift) - running this on all 40
# clusters is fine too, just slower and less necessary for clusters you
# won't present.
TOP_N = 10
target_clusters = results_df.sort_values("lift_over_baseline", ascending=False).head(TOP_N)["cluster"].tolist()

labels = df["future_subcat"].values
cluster_assignments = df["cluster"].values


def cluster_purity(cluster_id, labels_arr):
    mask = cluster_assignments == cluster_id
    sub = labels_arr[mask]
    if len(sub) == 0:
        return 0.0, None
    vals, counts = np.unique(sub, return_counts=True)
    i = counts.argmax()
    return counts[i] / len(sub), vals[i]


print("=" * 70)
print("PERMUTATION TEST: is each cluster's purity better than chance?")
print("=" * 70)

perm_results = []
for cid in target_clusters:
    real_purity, majority_class = cluster_purity(cid, labels)

    null_purities = np.empty(N_PERMUTATIONS)
    for i in range(N_PERMUTATIONS):
        shuffled = np.random.permutation(labels)
        mask = cluster_assignments == cid
        sub = shuffled[mask]
        vals, counts = np.unique(sub, return_counts=True)
        null_purities[i] = counts.max() / len(sub)

    p_value = (null_purities >= real_purity).mean()
    # avoid reporting exactly 0 - with N permutations the smallest
    # measurable p-value is 1/N
    p_value = max(p_value, 1 / N_PERMUTATIONS)

    perm_results.append({
        "cluster": cid, "majority_class": majority_class,
        "real_purity": real_purity, "null_mean_purity": null_purities.mean(),
        "null_95th_pct": np.percentile(null_purities, 95),
        "p_value": p_value,
        "significant_at_0.01": p_value < 0.01,
    })
    print(f"Cluster {cid:>3} ({majority_class:>12}): purity={real_purity:.1%}, "
          f"null mean={null_purities.mean():.1%}, null 95th pct={np.percentile(null_purities, 95):.1%}, "
          f"p={p_value:.4f}")

perm_df = pd.DataFrame(perm_results)
perm_df.to_csv("plots/permutation_test_results.csv", index=False)

print("\n" + "=" * 70)
print("BOOTSTRAP CONFIDENCE INTERVALS on purity")
print("=" * 70)

boot_results = []
for cid in target_clusters:
    mask = cluster_assignments == cid
    sub_labels = labels[mask]
    n = len(sub_labels)

    boot_purities = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        sample = np.random.choice(sub_labels, size=n, replace=True)
        vals, counts = np.unique(sample, return_counts=True)
        boot_purities[i] = counts.max() / n

    ci_low, ci_high = np.percentile(boot_purities, [2.5, 97.5])
    real_purity, majority_class = cluster_purity(cid, labels)

    boot_results.append({
        "cluster": cid, "majority_class": majority_class,
        "purity": real_purity, "ci_95_low": ci_low, "ci_95_high": ci_high,
    })
    print(f"Cluster {cid:>3} ({majority_class:>12}): "
          f"{real_purity:.1%}  95% CI [{ci_low:.1%}, {ci_high:.1%}]")

boot_df = pd.DataFrame(boot_results)
boot_df.to_csv("plots/bootstrap_ci_results.csv", index=False)

print("\nSaved plots/permutation_test_results.csv and plots/bootstrap_ci_results.csv")
print(
    "\nFor the report: quote p-values from the permutation test "
    "(e.g. 'p < 0.0005, N=2000 permutations') alongside purity, and use "
    "the bootstrap CIs whenever you state a purity/lift number as a "
    "headline figure, not just the point estimate."
)
