"""
EDA for the astro-ph pre-2009 project. Produces the visualizations for
your preliminary report / presentation:
- papers per year, with the 2009 split marked
- abstract length distribution
- abstract length by eventual (future) subcategory
- class balance of future_subcat (your validation target)
- number of co-authors distribution
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

df = pd.read_parquet("astroph_pre_with_future_labels.parquet")
df["year"] = df["submitted"].dt.year
df["abstract_len_words"] = df["abstract"].str.split().apply(len)

print(df[["year", "abstract_len_words"]].describe())
print("\nDate range:", df["submitted"].min(), "to", df["submitted"].max())
print("\nfuture_subcat distribution:\n", df["future_subcat"].value_counts(dropna=False))

# ---------------------------------------------------------------
# 1. Papers per year, full astro-ph corpus (pre-split set), 2009 marked
# ---------------------------------------------------------------
plt.figure(figsize=(10, 5))
df.groupby("year").size().plot(kind="line", marker="o")
plt.axvline(2009, color="red", linestyle="--", label="2009 subcategory split")
plt.title("astro-ph papers per year (pre-split working set)")
plt.xlabel("Year")
plt.ylabel("Number of papers")
plt.legend()
plt.tight_layout()
plt.savefig("fig1_astroph_papers_per_year.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 2. Abstract length distribution
# ---------------------------------------------------------------
plt.figure(figsize=(10, 5))
sns.histplot(df["abstract_len_words"], bins=50)
plt.title("Distribution of abstract length (words)")
plt.xlabel("Words in abstract")
plt.tight_layout()
plt.savefig("fig2_astroph_abstract_length_hist.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 3. Abstract length by future_subcat (box plot)
# ---------------------------------------------------------------
plt.figure(figsize=(10, 5))
plot_df = df.dropna(subset=["future_subcat"])
sns.boxplot(data=plot_df, x="future_subcat", y="abstract_len_words")
plt.title("Abstract length by eventual (future) subcategory")
plt.xlabel("Future subcategory (via author history)")
plt.ylabel("Words in abstract")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("fig3_astroph_length_by_future_subcat.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 4. Class balance - your validation target distribution
# ---------------------------------------------------------------
plt.figure(figsize=(8, 5))
plot_df["future_subcat"].value_counts().plot(kind="bar")
plt.title("Class balance: future subcategory (validation target)")
plt.xlabel("Subcategory")
plt.ylabel("Number of papers")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("fig4_astroph_future_subcat_balance.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 5. Co-author count distribution (context for author-based validation)
# ---------------------------------------------------------------
df["n_authors"] = df["all_author_keys"].apply(len)
plt.figure(figsize=(8, 5))
sns.histplot(df["n_authors"], bins=range(0, df["n_authors"].max() + 2), discrete=True)
plt.title("Number of co-authors per paper")
plt.xlabel("Co-authors")
plt.xlim(0, 20)  # long tail trimmed for readability
plt.tight_layout()
plt.savefig("fig5_astroph_n_authors_hist.png", dpi=150)
plt.close()

print("\nAll figures saved.")