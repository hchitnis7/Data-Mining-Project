"""
Task 3: Validate on a second split.

Generalization check: repeat the full method (author-history labeling +
TF-IDF/SVD/KMeans + purity-vs-baseline validation) on a second, different
category-evolution case, to see whether the astro-ph result was a fluke
of that particular split or reflects something more general.

Candidate: cs.NA (Numerical Analysis) / math.NA. Unlike cs.SY/eess.SY,
these are NOT a 1:1 administrative pair (cs.NA = 51,534 papers vs
math.NA = 51,595 - close but not identical), so there's a chance of
real cross-category signal. This script is self-contained: it re-streams
the raw snapshot, builds the working set, and runs the same pipeline.

Adjust RAW_PATH and SPLIT_DATE/TARGET_CATEGORIES for whatever second
case you settle on.
"""

import json
import re
import time
import pandas as pd
from collections import defaultdict

t0 = time.time()
def tick(msg):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

# ---- CONFIG: adjust for your chosen second case ----
RAW_PATH = r"/teamspace/studios/this_studio/arxiv-metadata-oai-snapshot.json"
TARGET_CATEGORIES = {"cs.NA", "math.NA"}
# math.NA has existed throughout; cs.NA was introduced as a secondary/
# cross-listed tag later. If you can't find a clean split date for this
# pair, treat the "split date" as a midpoint or document it as a
# limitation - unlike eess (2017) and astro-ph (2009), cs.NA/math.NA
# doesn't have one official announced date, which is itself worth
# reporting in the generalization discussion.
SPLIT_DATE = "2010-01-01"  # placeholder - verify/adjust before trusting results


def get_earliest_date(versions):
    if not versions:
        return None
    return versions[0].get("created")


def author_key(a):
    last = (a[0] or "").strip().lower()
    first = (a[1] or "").strip().lower().rstrip(".")
    return f"{last}_{first[:1] if len(first) <= 1 else first}"


rows = []
total = 0
with open(RAW_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        total += 1
        try:
            paper = json.loads(line)
        except json.JSONDecodeError:
            continue
        cats = set(paper.get("categories", "").split())
        if not (cats & TARGET_CATEGORIES):
            continue
        abstract = paper.get("abstract", "")
        if not abstract.strip():
            continue
        earliest = get_earliest_date(paper.get("versions", []))
        if not earliest:
            continue
        authors_parsed = paper.get("authors_parsed", [])
        rows.append({
            "id": paper.get("id"),
            "abstract": abstract.strip(),
            "categories": paper.get("categories", ""),
            "submitted": earliest,
            "all_author_keys": [author_key(a) for a in authors_parsed],
        })
        if total % 500_000 == 0:
            tick(f"Scanned {total:,}...")

df = pd.DataFrame(rows)
df["submitted"] = pd.to_datetime(df["submitted"], errors="coerce", utc=True)
df = df.dropna(subset=["submitted"])
split_ts = pd.Timestamp(SPLIT_DATE, tz="UTC")
df["pre_split"] = df["submitted"] < split_ts
tick(f"Working set: {len(df):,} rows, {df['pre_split'].sum():,} pre-split")

pre = df[df["pre_split"]].copy()
post = df[~df["pre_split"]].copy()

# Author -> primary post-split category lookup
author_cats = defaultdict(set)
for _, row in post.iterrows():
    primary = row["categories"].split()[0]
    if primary not in TARGET_CATEGORIES:
        continue
    for k in row["all_author_keys"]:
        if k:
            author_cats[k].add(primary)


def dominant_cat(keys):
    votes = defaultdict(int)
    for k in keys:
        for c in author_cats.get(k, set()):
            votes[c] += 1
    return max(votes, key=votes.get) if votes else None


pre["future_cat"] = pre["all_author_keys"].apply(dominant_cat)
pre = pre.dropna(subset=["future_cat"]).reset_index(drop=True)
tick(f"Resolvable pre-split papers: {len(pre):,}")
print(pre["future_cat"].value_counts())

baseline = pre["future_cat"].value_counts(normalize=True)
print(f"\nBaseline majority rate: {baseline.max():.1%}")

# ---- Clustering: same TF-IDF + SVD + KMeans pipeline ----
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans


def clean_text(text):
    text = text.lower()
    text = re.sub(r"\$.*?\$", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


pre["clean_abstract"] = pre["abstract"].apply(clean_text)
vectorizer = TfidfVectorizer(max_features=8000, stop_words="english", min_df=5, max_df=0.5)
X = vectorizer.fit_transform(pre["clean_abstract"])
svd = TruncatedSVD(n_components=min(300, X.shape[0] - 1, X.shape[1] - 1), random_state=42)
X_svd = svd.fit_transform(X)
tick(f"SVD explained variance: {svd.explained_variance_ratio_.sum():.3f}")

k = min(40, max(5, len(pre) // 200))  # scale k to dataset size
clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
pre["cluster"] = clusterer.fit_predict(X_svd)
tick(f"KMeans done with k={k}")

results = []
for cid, group in pre.groupby("cluster"):
    counts = group["future_cat"].value_counts()
    majority = counts.idxmax()
    purity = counts.max() / len(group)
    results.append({"cluster": cid, "size": len(group), "majority_class": majority,
                     "purity": purity, "lift": purity - baseline.get(majority, 0)})

results_df = pd.DataFrame(results).sort_values("lift", ascending=False)
print("\nSecond-split validation results (top 10 by lift):")
print(results_df.head(10).to_string(index=False))

results_df.to_csv("plots/second_split_validation_results.csv", index=False)
pre.to_parquet("second_split_clustered.parquet", index=False)
tick("Saved results")

print(
    "\nInterpretation guide: if several clusters beat baseline by a "
    "meaningful margin (similar order of magnitude to the astro-ph "
    "result), that supports generalization. If lift is near zero "
    "everywhere, report that honestly - it suggests the astro-ph result "
    "may depend on astro-ph's unusually distinct subfield vocabulary "
    "rather than being a general property of the method."
)
