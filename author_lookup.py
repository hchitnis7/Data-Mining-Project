"""
Step 2: Two things before clustering:
  (a) Check whether pre-2009 papers were retroactively machine-classified
      into subcategories (arXiv said they might be, eventually).
  (b) Build a lookup: for each author, which astro-ph subcategories did
      they publish in AFTER the 2009 split? This is the ground truth
      we'll validate clusters against.
"""

import pandas as pd
from collections import defaultdict

SUBCATS = {"astro-ph.CO", "astro-ph.EP", "astro-ph.GA",
           "astro-ph.HE", "astro-ph.IM", "astro-ph.SR"}

df = pd.read_parquet("astroph_working_set.parquet")

pre = df[df["pre_split"]].copy()
post = df[~df["pre_split"]].copy()

print(f"Pre-split papers: {len(pre):,}")
print(f"Post-split papers: {len(post):,}")

# ---------------------------------------------------------------
# (a) Retroactive relabeling check
# ---------------------------------------------------------------
def has_any_subcat(categories_str):
    cats = set(categories_str.split())
    return len(cats & SUBCATS) > 0

pre["has_subcat_label"] = pre["categories"].apply(has_any_subcat)
print(f"\nPre-split papers that ALREADY carry a subcategory tag today: "
      f"{pre['has_subcat_label'].sum():,} / {len(pre):,} "
      f"({pre['has_subcat_label'].mean():.1%})")

if pre["has_subcat_label"].mean() > 0.3:
    print(">> Significant retroactive relabeling detected - you have a "
          "second, direct validation signal available in addition to "
          "the author-based one.")
else:
    print(">> Little/no retroactive relabeling - author-based validation "
          "is your primary signal, as planned.")

# ---------------------------------------------------------------
# (b) Author -> post-2009 subcategories lookup
# ---------------------------------------------------------------
author_subcats = defaultdict(set)

for _, row in post.iterrows():
    # Use only the PRIMARY category (first token) - cross-listed tags
    # like astro-ph.IM get added to almost every active author's
    # history regardless of their actual specialization, which swamps
    # the signal if we count all tags equally.
    primary_cat = row["categories"].split()[0]
    if primary_cat not in SUBCATS:
        continue
    for author_key in row["all_author_keys"]:
        if author_key:
            author_subcats[author_key].add(primary_cat)

print(f"\nBuilt lookup for {len(author_subcats):,} distinct authors "
      f"who published in a subcategory post-2009.")


def dominant_subcat_for_paper(all_author_keys):
    """
    For a pre-split paper, look up what subcategory(ies) its authors
    went on to publish in. Returns the most common one among the
    paper's authors, or None if no author has post-2009 subcat history.
    """
    votes = defaultdict(int)
    for key in all_author_keys:
        for sc in author_subcats.get(key, set()):
            votes[sc] += 1
    if not votes:
        return None
    return max(votes, key=votes.get)


pre["future_subcat"] = pre["all_author_keys"].apply(dominant_subcat_for_paper)

print("\nPre-split papers with a resolvable 'future subcategory' via authors:")
print(pre["future_subcat"].value_counts(dropna=False))

pre.to_parquet("astroph_pre_with_future_labels.parquet", index=False)
print("\nSaved astroph_pre_with_future_labels.parquet")