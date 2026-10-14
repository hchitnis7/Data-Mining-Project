"""Compare saved original and improved labels on identical paper IDs and fixed clusters."""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(__file__).resolve().parent / 'results'


def class_lift_comparison(common, min_size=50):
    """Compare class-specific enrichment on the SAME old-selected clusters."""
    sizes = common.groupby('cluster').size()
    valid = sizes[sizes >= min_size].index
    classes = sorted(set(common.future_subcat_old) | set(common.future_subcat_new))
    rows = []
    for cls in classes:
        rates = {}
        baselines = {}
        for method in ['old', 'new']:
            column = f'future_subcat_{method}'
            baselines[method] = float(common[column].eq(cls).mean())
            rates[method] = common[column].eq(cls).groupby(common.cluster).mean().reindex(valid)
        row = {'class': cls, 'comparison_papers': len(common),
               'old_baseline': baselines['old'], 'new_baseline': baselines['new']}
        if valid.empty:
            row.update(cluster=None, cluster_size=0, old_concentration=np.nan,
                       new_concentration=np.nan, old_lift=np.nan, new_lift=np.nan,
                       lift_change_pp=np.nan, new_best_cluster=None, new_best_lift=np.nan)
        else:
            cid = int(rates['old'].idxmax())
            old_lift = float(rates['old'].loc[cid]-baselines['old'])
            new_lift = float(rates['new'].loc[cid]-baselines['new'])
            new_best = int(rates['new'].idxmax())
            row.update(cluster=cid, cluster_size=int(sizes.loc[cid]),
                       old_concentration=float(rates['old'].loc[cid]),
                       new_concentration=float(rates['new'].loc[cid]),
                       old_lift=old_lift, new_lift=new_lift, lift_change_pp=100*(new_lift-old_lift),
                       new_best_cluster=new_best,
                       new_best_lift=float(rates['new'].loc[new_best]-baselines['new']))
        rows.append(row)
    return pd.DataFrame(rows)


def write_final_report(out, summary, changes, category_results, method):
    full_names = method == 'weighted_votes_exact_full_names'
    lines = ['# Final author-label comparison', '',
             f'Saved method: `{method}`.', '',
             'The original future_subcat uses surname + first-initial author keys (for example, smith_j), which can merge distinct people.']
    if full_names:
        lines += ['The new labels use normalized full given names where available and weighted publication-history voting.',
                  'This experiment changes both identity matching and voting; it does not isolate the effect of full names alone.']
    else:
        lines += ['The saved new labels still use the original initial keys. They change voting to equal-author, publication-frequency-weighted votes with unresolved ties.',
                  '**Full-name author validation has not been run on these saved results.** The raw matching metadata is needed to rebuild full-name keys.']
    lines += ['', '| Method | Resolved papers | Unresolved papers |', '|---|---:|---:|']
    for row in summary.itertuples(index=False):
        lines.append(f'| {row.method} | {row.resolved:,} | {row.unresolved:,} |')
    lines += ['', f"{int(changes['changed_resolved_label']):,} commonly resolved papers changed label; {int(changes['lost_label']):,} lost their label.",
              '', '## Category lift on identical papers and clusters', '']
    if category_results is None:
        lines += ['No saved GPU cluster dataset was available; category lift could not be calculated.']
    else:
        lines += [f'Comparison uses {int(category_results.comparison_papers.iloc[0]):,} papers with both labels and saved GPU assignments.',
                  'For each category, select its best original-label cluster (minimum 50 papers), then compare both labels on that same cluster.',
                  'Lift = category concentration in the cluster minus that category\'s baseline rate across the identical comparison papers.',
                  '', '| Category | Cluster | Papers | Old concentration | New concentration | Old lift (pp) | New lift (pp) | Change (pp) |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in category_results.itertuples(index=False):
            lines.append(f'| {r._0} | {r.cluster} | {r.cluster_size:,} | {r.old_concentration:.1%} | {r.new_concentration:.1%} | {100*r.old_lift:+.1f} | {100*r.new_lift:+.1f} | {r.lift_change_pp:+.1f} |')
        increases = category_results.loc[category_results.lift_change_pp > 0, 'class'].tolist()
        decreases = category_results.loc[category_results.lift_change_pp < 0, 'class'].tolist()
        lines += ['', f"Lift increased for: {', '.join(increases) or 'none'}.",
                  f"Lift decreased for: {', '.join(decreases) or 'none'}.", '', '![Category lift](class_lift_comparison.png)',
                  '', '`class_lift_comparison.csv` also reports the best cluster selected using new labels, separately from the paired comparison.']
    lines += ['', '## Interpretation', '',
              'These are descriptive, in-sample enrichment results against author-history proxies, not verified classification accuracy.',
              'The text representation and K-means assignments are fixed; clustering was not rerun. Baseline rates are recalculated for each labeling method.',
              'Higher lift against changed proxy labels is not proof that author identities or paper categories became more accurate.',
              'For an identity-only comparison, rerun initial-key and full-name matching with identical voting, then compare the same paper IDs and cluster assignments.',
              'Full-name matching can still merge identical names and split abbreviated/name variants. Statistical significance has not been tested.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def compare(new_path=RESULTS / 'astroph_pre_with_future_labels_new.parquet', out=RESULTS):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    old = pd.read_parquet(ROOT / 'astroph_pre_with_future_labels.parquet', columns=['id', 'future_subcat'])
    new = pd.read_parquet(new_path)
    methods = new.label_method.dropna().unique()
    if len(methods) != 1:
        raise ValueError('Expected a single label_method for the comparison.')
    method = methods[0]
    df = old.merge(new, on='id', suffixes=('_old', '_new'), validate='one_to_one')
    if len(df) != len(old) or len(df) != len(new):
        raise ValueError('Comparison requires identical paper IDs.')
    a, b = df.future_subcat_old, df.future_subcat_new
    summary = pd.DataFrame([
        {'method': 'original_saved', 'papers': len(df), 'resolved': a.notna().sum(), 'unresolved': a.isna().sum(), 'high_confidence': None},
        {'method': method, 'papers': len(df), 'resolved': b.notna().sum(), 'unresolved': b.isna().sum(), 'high_confidence': df.high_confidence.sum()},
    ])
    summary.to_csv(out / 'summary.csv', index=False)
    changes = pd.Series({'same_resolved_label': ((a == b) & a.notna()).sum(),
        'changed_resolved_label': ((a != b) & a.notna() & b.notna()).sum(),
        'lost_label': (a.notna() & b.isna()).sum(), 'gained_label': (a.isna() & b.notna()).sum(),
        'both_unresolved': (a.isna() & b.isna()).sum()}, name='papers')
    changes.to_csv(out / 'label_changes.csv')
    transition = pd.crosstab(a.fillna('Unresolved'), b.fillna('Unresolved'))
    transition.to_csv(out / 'label_transition.csv')
    classes = sorted(set(a.dropna()) | set(b.dropna())) + ['Unresolved']
    counts = pd.DataFrame({'Original': a.fillna('Unresolved').value_counts(),
                           'New': b.fillna('Unresolved').value_counts()}).reindex(classes).fillna(0)
    counts.to_csv(out / 'class_counts.csv')
    ax = counts.plot.bar(figsize=(11, 6))
    ax.set_ylabel('Papers'); ax.set_title('Author proxy label distribution (same papers)')
    plt.xticks(rotation=30, ha='right'); plt.tight_layout()
    plt.savefig(out / 'class_counts.png', dpi=160); plt.close()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(transition, cmap='Blues')
    ax.set_xticks(range(len(transition.columns)), transition.columns, rotation=35, ha='right')
    ax.set_yticks(range(len(transition.index)), transition.index)
    for i in range(len(transition)):
        for j in range(len(transition.columns)):
            ax.text(j, i, str(transition.iloc[i, j]), ha='center', va='center', fontsize=8)
    ax.set_xlabel('New label'); ax.set_ylabel('Original label')
    ax.set_title('Label transitions: counts, not accuracy'); fig.tight_layout()
    fig.savefig(out / 'label_transition.png', dpi=160); plt.close(fig)
    path = ROOT / 'astroph_clustered_full_gpu.parquet'
    category_results = None
    if path.exists():
        clusters = pd.read_parquet(path, columns=['id', 'cluster'])
        merged = clusters.merge(df, on='id', validate='one_to_one')
        common = merged.dropna(subset=['future_subcat_old', 'future_subcat_new'])
        if common.empty:
            raise ValueError('No commonly labeled papers with saved clusters.')
        category_results = class_lift_comparison(common)
        category_results.to_csv(out/'class_lift_comparison.csv', index=False)
        plot = category_results.set_index('class')[['old_lift', 'new_lift']].mul(100)
        ax = plot.rename(columns={'old_lift': 'Original labels', 'new_lift': 'New labels'}).plot.bar(
            figsize=(10, 6), color=['#64748b', '#0f766e'])
        ax.set_ylabel('Class-specific lift (percentage points)')
        ax.set_xlabel('Category')
        ax.set_title('Same papers and original-selected clusters; minimum size 50')
        ax.axhline(0, color='black', linewidth=.8)
        plt.xticks(rotation=0); plt.tight_layout()
        plt.savefig(out/'class_lift_comparison.png', dpi=180); plt.close()
        rows = []
        for cid, group in common.groupby('cluster'):
            row = {'cluster': cid, 'common_labeled_papers': len(group)}
            for name in ['old', 'new']:
                dist = group[f'future_subcat_{name}'].value_counts(normalize=True)
                row[f'{name}_majority'] = dist.idxmax()
                row[f'{name}_purity'] = dist.max()
            row['purity_change'] = row['new_purity'] - row['old_purity']
            rows.append(row)
        results = pd.DataFrame(rows)
        results.to_csv(out / 'fixed_cluster_comparison.csv', index=False)
        ax = results.set_index('cluster')[['old_purity', 'new_purity']].plot.bar(figsize=(14, 6))
        ax.set_ylabel('Majority-label fraction'); ax.set_ylim(0, 1)
        ax.set_title('Fixed GPU clusters, identical commonly labeled papers')
        plt.tight_layout(); plt.savefig(out / 'fixed_cluster_purity.png', dpi=160); plt.close()
    write_final_report(out, summary, changes, category_results, method)
    print(summary.to_string(index=False)); print(changes.to_string())
    print(f'Comparison tables and plots saved in {out}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, default=RESULTS / 'astroph_pre_with_future_labels_new.parquet')
    parser.add_argument('--output', type=Path, default=RESULTS)
    args = parser.parse_args()
    compare(new_path=args.labels, out=args.output)
