"""
Step 1 (revised target): astro-ph pre-2009 subcategory split.

On 20 Jan 2009, arXiv split astro-ph into six subcategories
(astro-ph.CO, .EP, .GA, .HE, .IM, .SR), mandatory going forward.
Unlike cs.SY -> eess.SY, this was NOT retroactively relabeled onto
old papers (astro-ph still exists standalone today), so this is a
genuine "can we detect the subfield before it was named" case.

Validation strategy: author-based. For each pre-2009 paper, check
whether its authors later (post-2009) published papers tagged with
a specific astro-ph subcategory. If a cluster's authors disproportionately
went on to one subcategory, that's the signal.

NOTE: this version saves the raw `authors_parsed` field into the
output parquet (as a plain list-of-lists, not a numpy/pandas-specific
type) so that 06_author_disambiguation.py can rebuild better author
keys later using full first names instead of just initials. Without
this column saved, task 1 (author disambiguation) can't run without
re-scanning the raw JSON from scratch.
"""

import json
import pandas as pd

RAW_PATH = r"arxiv-metadata-oai-snapshot.json"
OUT_PATH = "astroph_working_set.parquet"

SPLIT_DATE = "2009-01-20"
SUBCATS = {"astro-ph.CO", "astro-ph.EP", "astro-ph.GA",
           "astro-ph.HE", "astro-ph.IM", "astro-ph.SR"}
RELEVANT = SUBCATS | {"astro-ph"}


def get_earliest_date(versions):
    if not versions:
        return None
    return versions[0].get("created")


def first_author_key(authors_parsed):
    """Crude author identity key: last name + first initial of first author."""
    if not authors_parsed:
        return None
    a = authors_parsed[0]
    last = (a[0] or "").strip().lower()
    first_initial = (a[1] or "").strip().lower()[:1]
    return f"{last}_{first_initial}"


def clean_authors_parsed(authors_parsed):
    """
    Normalize authors_parsed into a plain list of 3-element lists of
    plain strings (never None), so it round-trips cleanly through
    parquet (pyarrow is picky about nested lists with mixed/None
    entries and non-string types).
    """
    cleaned = []
    for a in authors_parsed or []:
        # Raw entries are typically [last, first, suffix] - pad/truncate
        # defensively in case a row has a different shape.
        last = str(a[0]) if len(a) > 0 and a[0] is not None else ""
        first = str(a[1]) if len(a) > 1 and a[1] is not None else ""
        suffix = str(a[2]) if len(a) > 2 and a[2] is not None else ""
        cleaned.append([last, first, suffix])
    return cleaned


def main():
    rows = []
    total_scanned = 0

    with open(RAW_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_scanned += 1

            try:
                paper = json.loads(line)
            except json.JSONDecodeError:
                continue

            categories = paper.get("categories", "")
            cats = set(categories.split())
            if not (cats & RELEVANT):
                continue

            abstract = paper.get("abstract", "")
            if not abstract or not abstract.strip():
                continue

            earliest_date = get_earliest_date(paper.get("versions", []))
            if not earliest_date:
                continue

            authors_parsed = paper.get("authors_parsed", [])

            rows.append({
                "id": paper.get("id"),
                "abstract": abstract.strip(),
                "categories": categories,
                "submitted": earliest_date,
                "first_author_key": first_author_key(authors_parsed),
                "all_author_keys": [
                    f"{(a[0] or '').strip().lower()}_{(a[1] or '').strip().lower()[:1]}"
                    for a in authors_parsed
                ],
                # Raw field, kept so later scripts (author disambiguation)
                # can rebuild keys using full first names instead of just
                # initials, without re-scanning the raw JSON.
                "authors_parsed": clean_authors_parsed(authors_parsed),
            })

            if total_scanned % 500_000 == 0:
                print(f"Scanned {total_scanned:,}...")

    df = pd.DataFrame(rows)
    df["submitted"] = pd.to_datetime(df["submitted"], errors="coerce", utc=True)
    df = df.dropna(subset=["submitted"])

    split_ts = pd.Timestamp(SPLIT_DATE, tz="UTC")
    df["pre_split"] = df["submitted"] < split_ts

    for sc in SUBCATS:
        df[f"has_{sc.replace('.', '_')}"] = df["categories"].apply(
            lambda c, sc=sc: sc in c.split()
        )

    print(f"\nTotal rows: {len(df):,}")
    print(df["pre_split"].value_counts())

    df.to_parquet(OUT_PATH, index=False)
    print(f"Saved to {OUT_PATH}")
    print(f"Columns saved: {list(df.columns)}")


if __name__ == "__main__":
    main()