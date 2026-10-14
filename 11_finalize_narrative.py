"""
Task 6: Finalize presentation narrative.

Not a new analysis - this pulls together everything from tasks 1-5 
plus the original results into one ranked shortlist, so
you can pick the 3-4 strongest clusters to center the oral exam on, and
prepares a one-line "defensible claim" plus a cheat-sheet of definitions
you may be asked to explain on the spot (purity, lift, baseline,
permutation test, Jaccard-adjacent clustering concepts).

Run last, after whichever of 06-10 you completed.
"""

import os
import pandas as pd

os.makedirs("plots", exist_ok=True)

print("=" * 70)
print("SHORTLIST: strongest clusters for the oral exam narrative")
print("=" * 70)

master = pd.read_csv("/teamspace/studios/this_studio/plots/tableB9_master_results.csv")
master["lift_numeric"] = master["lift"].str.replace("+", "", regex=False).str.rstrip("%").astype(float)
master_sorted = master.sort_values("lift_numeric", ascending=False)

print("\nRanked by lift over baseline (core evidence, present in this order):")
print(master_sorted[["class", "baseline_rate", "cluster_purity_for_class", "lift", "top_terms"]].to_string(index=False))

# If the permutation test was run, attach p-values for extra rigor
perm_path = "plots/permutation_test_results.csv"
if os.path.exists(perm_path):
    perm = pd.read_csv(perm_path)
    print("\nSignificance (from permutation test, if cluster IDs match):")
    print(perm[["cluster", "majority_class", "real_purity", "p_value"]].to_string(index=False))

print("\n" + "=" * 70)
print("SUGGESTED NARRATIVE ORDER (lead with the strongest, most defensible)")
print("=" * 70)
narrative = [
    "1. astro-ph.CO (cluster 17, brane-world/extra-dimensional vocabulary) "
    "- highest purity (77.7%), clean theoretical vocabulary, easy to explain to a non-technical audience.",
    "2. astro-ph.SR (cluster 20, white dwarf/compact object vocabulary) "
    "- second-strongest, large enough sample (1,046 papers) to feel robust.",
    "3. astro-ph.EP (cluster 36, exoplanet vocabulary) "
    "- best example of 'rare class, strong relative signal' (+30.5pp lift despite only 4.7% baseline).",
    "4. astro-ph.IM (weakest result) "
    "- present as the counter-example that sharpens the claim: methodological categories "
    "don't have their own vocabulary, so the method's failure mode is itself informative.",
]
for line in narrative:
    print(line)

print("\n" + "=" * 70)
print("CHEAT SHEET - likely questions in the oral exam")
print("=" * 70)
cheat_sheet = {
    "Purity": "Fraction of a cluster's papers belonging to that cluster's majority class. "
              "Higher = the cluster is more 'pure' (internally consistent).",
    "Lift over baseline": "Purity minus the overall base rate of that class in the full dataset. "
                           "Measures how much better than random guessing the cluster is.",
    "Baseline / majority-class guessing": "The accuracy you'd get by always predicting the most "
                                           "common class, with no model at all - the floor any real method must beat.",
    "TF-IDF": "Term Frequency-Inverse Document Frequency: weights words by how distinctive they are "
              "to a document relative to the whole corpus, down-weighting common words.",
    "TruncatedSVD": "Dimensionality reduction (like PCA but works on sparse data without centering) "
                    "used to compress the TF-IDF matrix before clustering.",
    "K-Means": "Clustering algorithm that partitions points into k groups by minimizing distance "
               "to each group's centroid. Requires choosing k in advance (see Task 2).",
    "Permutation test": "Null-hypothesis test: shuffle the labels many times, recompute the result "
                         "each time, and see how often chance alone produces something as extreme as "
                         "what was actually observed. Low p-value = unlikely to be chance.",
    "Bootstrap confidence interval": "Resampling the data (with replacement) many times to estimate "
                                      "how much a statistic (like purity) would vary if we'd drawn a "
                                      "different sample - gives a plausible range, not just one number.",
    "future_subcat (author-history proxy)": "Since old papers weren't retroactively relabeled, we infer "
                                             "the 'ground truth' subfield of a pre-2009 paper from what "
                                             "subcategory its authors predominantly published in after 2009.",
}
for term, definition in cheat_sheet.items():
    print(f"\n{term}:\n  {definition}")

with open("plots/narrative_cheat_sheet.txt", "w") as f:
    f.write("SHORTLIST AND CHEAT SHEET FOR ORAL EXAM\n")
    f.write("=" * 70 + "\n\n")
    f.write("Narrative order:\n")
    for line in narrative:
        f.write(line + "\n")
    f.write("\nDefinitions:\n")
    for term, definition in cheat_sheet.items():
        f.write(f"\n{term}:\n  {definition}\n")

print("\nSaved plots/narrative_cheat_sheet.txt")
