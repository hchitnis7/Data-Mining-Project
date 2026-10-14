"""Build improved proxy labels
Use --raw PATH to recover full names from the original metadata snapshot.
Without --raw, only voting is improved; identity collisions remain.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
if __package__:
    from .author_identity import full_author_key
else:
    from author_identity import full_author_key

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(__file__).resolve().parent / 'results'

SUBCATS = sorted({'astro-ph.CO', 'astro-ph.EP', 'astro-ph.GA',
                  'astro-ph.HE', 'astro-ph.IM', 'astro-ph.SR'})


def label_paper(keys, history):
    matched = [history[k] for k in sorted(set(keys)) if k and k in history]
    if not matched:
        return None, 0.0, 0.0, 0, False
    scores = np.array([sum(h.get(c, 0) / sum(h.values()) for h in matched)
                       for c in SUBCATS]) / len(matched)
    order = np.argsort(-scores, kind='stable')
    confidence = float(scores[order[0]])
    margin = float(scores[order[0]] - scores[order[1]])
    tied = bool(np.isclose(margin, 0, atol=1e-12, rtol=0))
    return (None if tied else SUBCATS[order[0]], confidence, margin, len(matched), tied)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path)
    parser.add_argument('--input', type=Path, default=ROOT / 'astroph_working_set.parquet')
    parser.add_argument('--output', type=Path, default=RESULTS, help='Directory for new labels and comparisons')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.input)
    key_column = 'full_author_keys' if 'full_author_keys' in df else 'all_author_keys'
    method = ('weighted_votes_exact_full_names' if key_column == 'full_author_keys'
              else 'weighted_votes_existing_initial_keys')
    if args.raw:
        names = {}
        wanted = set(df['id'].astype(str))
        with args.raw.open(encoding='utf-8') as stream:
            for line in stream:
                if not line.strip():
                    continue
                paper = json.loads(line)
                pid = str(paper.get('id', ''))
                if pid in wanted:
                    names[pid] = [full_author_key(a) for a in paper.get('authors_parsed', [])]
        missing = wanted - names.keys()
        if missing:
            raise ValueError(f'Raw snapshot lacks {len(missing)} working-set IDs; use the matching snapshot.')
        df['full_author_keys'] = df['id'].astype(str).map(names)
        key_column = 'full_author_keys'
        method = 'weighted_votes_exact_full_names'
    history = defaultdict(Counter)
    for cats, keys in df.loc[~df.pre_split, ['categories', key_column]].itertuples(index=False, name=None):
        tokens = cats.split()
        if tokens and tokens[0] in SUBCATS:
            for key in set(keys):
                if key:
                    history[key][tokens[0]] += 1
    pre = df.loc[df.pre_split].copy()
    values = [label_paper(keys, history) for keys in pre[key_column]]
    columns = ['future_subcat', 'label_confidence', 'label_margin', 'matched_authors', 'label_tied']
    pre[columns] = pd.DataFrame(values, index=pre.index)
    pre['label_method'] = method
    pre['high_confidence'] = (pre.future_subcat.notna() & (pre.label_confidence >= .7)
                              & (pre.label_margin >= .2) & (pre.matched_authors >= 2))
    pre['has_subcat_label'] = pre.categories.map(lambda c: bool(set(c.split()) & set(SUBCATS)))
    label_path = args.output / 'astroph_pre_with_future_labels_new.parquet'
    pre.to_parquet(label_path, index=False)
    print(f'Method: {method}; resolved: {pre.future_subcat.notna().sum():,}/{len(pre):,}')
    print(f'Saved {label_path}; original labels preserved.')
    if __package__:
        from .compare_author_labels import compare
    else:
        from compare_author_labels import compare
    compare(new_path=label_path, out=args.output)


if __name__ == '__main__':
    main()
