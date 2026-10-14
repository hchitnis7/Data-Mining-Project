"""Compare TF-IDF/SVD with sentence embeddings on a shared held-out split."""
import argparse
import hashlib
import importlib.metadata
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
CLASSES = sorted(['astro-ph.CO', 'astro-ph.EP', 'astro-ph.GA', 'astro-ph.HE', 'astro-ph.IM', 'astro-ph.SR'])


def clean_text(text):
    text = re.sub(r'\$.*?\$', ' ', text.lower())
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z\s]', ' ', text)).strip()


def evaluate(y_train, y_test, train_clusters, test_clusters, min_size):
    """Learn cluster labels/choose class clusters on training data only."""
    table = pd.crosstab(train_clusters, y_train).reindex(columns=CLASSES, fill_value=0)
    mapping = table.idxmax(axis=1)
    predictions = mapping.loc[test_clusters].to_numpy()
    summary = {
        'test_ami': adjusted_mutual_info_score(y_test, test_clusters),
        'test_ari': adjusted_rand_score(y_test, test_clusters),
        'test_accuracy': accuracy_score(y_test, predictions),
        'test_macro_f1': f1_score(y_test, predictions, labels=CLASSES, average='macro', zero_division=0),
    }
    rows = []
    valid = table.loc[table.sum(axis=1) >= min_size]
    for cls in CLASSES:
        row = {'class': cls, 'test_baseline_rate': float(np.mean(y_test == cls))}
        if valid.empty:
            row.update(cluster=None, train_size=0, test_size=0, test_concentration=np.nan,
                       test_lift=np.nan, test_class_coverage=np.nan)
        else:
            cid = int(valid[cls].div(valid.sum(axis=1)).idxmax())
            mask = test_clusters == cid
            concentration = float(np.mean(y_test[mask] == cls)) if mask.any() else np.nan
            row.update(cluster=cid, train_size=int(table.loc[cid].sum()), test_size=int(mask.sum()),
                       test_concentration=concentration, test_lift=concentration-row['test_baseline_rate'],
                       test_class_coverage=(float(np.sum(mask & (y_test == cls))/np.sum(y_test == cls))
                                            if np.any(y_test == cls) else np.nan))
        rows.append(row)
    return summary, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=ROOT/'astroph_pre_with_future_labels.parquet')
    p.add_argument('--output', type=Path, default=Path(__file__).parent/'results')
    p.add_argument('--sample', type=int, default=15000, help='0 uses all resolved papers')
    p.add_argument('--k', type=int, default=40)
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    p.add_argument('--min-cluster-size', type=int, default=50)
    p.add_argument('--model', default='sentence-transformers/all-MiniLM-L6-v2')
    p.add_argument('--revision', default=None, help='Optional Hugging Face commit for reproducibility')
    p.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    if args.sample < 0 or args.k < 2 or args.min_cluster_size < 1 or args.threads < 1:
        p.error('sample must be >= 0; k >= 2; min-cluster-size and threads >= 1')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.input).sort_values('id').reset_index(drop=True)
    df = df.loc[df.future_subcat.isin(CLASSES) & df.abstract.fillna('').str.strip().ne('')].copy()
    if df.id.duplicated().any():
        raise ValueError('Paper IDs must be unique')
    if args.sample and args.sample < len(df):
        selected, _ = train_test_split(np.arange(len(df)), train_size=args.sample,
                                      stratify=df.future_subcat, random_state=42)
        df = df.iloc[selected].sort_values('id').reset_index(drop=True)
    train, test = train_test_split(np.arange(len(df)), test_size=.2, stratify=df.future_subcat, random_state=42)
    if args.k >= len(train):
        p.error('k must be smaller than the training set')
    split = np.full(len(df), 'train', dtype=object)
    split[test] = 'test'
    df[['id', 'future_subcat']].assign(split=split).to_csv(out/'sample_split.csv', index=False)
    labels = df.future_subcat.to_numpy()
    y_train, y_test = labels[train], labels[test]
    majority = pd.Series(y_train).value_counts().idxmax()
    baseline_accuracy = float(np.mean(y_test == majority))
    texts = df.abstract.str.replace(r'\s+', ' ', regex=True).str.strip().tolist()
    print(f'{len(df):,} papers: {len(train):,} train / {len(test):,} test', flush=True)
    from sentence_transformers import SentenceTransformer
    import torch
    torch.set_num_threads(args.threads)
    model = SentenceTransformer(args.model, revision=args.revision, device=args.device,
                                cache_folder=str(out/'model_cache'), trust_remote_code=False)
    resolved_revision = getattr(model[0].auto_model.config, '_commit_hash', None) or args.revision
    if resolved_revision is None:
        reference = out/'model_cache'/('models--'+args.model.replace('/', '--'))/'refs'/'main'
        if reference.exists():
            resolved_revision = reference.read_text(encoding='utf-8').strip()
    token_lengths = [len(t) for t in model.tokenizer(texts, truncation=False, verbose=False)['input_ids']]
    truncated = sum(n > model.max_seq_length for n in token_lengths)
    signature = hashlib.sha256(json.dumps({'texts': texts, 'model': args.model,
        'revision': resolved_revision, 'max_seq_length': model.max_seq_length}, sort_keys=True).encode()).hexdigest()
    cache = out/f'embeddings_{signature}.npy'
    start = time.perf_counter()
    if cache.exists():
        embeddings = np.load(cache, allow_pickle=False)
        print('Loaded cached embeddings', flush=True)
    else:
        embeddings = model.encode(texts, batch_size=args.batch_size, show_progress_bar=True,
                                  normalize_embeddings=True, convert_to_numpy=True)
        np.save(cache, embeddings)
    embedding_seconds = time.perf_counter()-start
    if embeddings.shape[0] != len(df) or not np.isfinite(embeddings).all():
        raise ValueError('Invalid embedding cache')
    with threadpool_limits(limits=args.threads):
        start = time.perf_counter()
        vectorizer = TfidfVectorizer(max_features=8000, stop_words='english', min_df=5, max_df=.5, dtype=np.float32)
        cleaned = [clean_text(t) for t in texts]
        x_train = vectorizer.fit_transform([cleaned[i] for i in train])
        x_test = vectorizer.transform([cleaned[i] for i in test])
        components = min(300, x_train.shape[0]-1, x_train.shape[1]-1)
        if components < 2:
            raise ValueError('Insufficient vocabulary for SVD')
        svd = TruncatedSVD(n_components=components, random_state=42)
        svd_train, svd_test = svd.fit_transform(x_train), svd.transform(x_test)
        tfidf_seconds = time.perf_counter()-start
        representations = {
            'tfidf_svd': (svd_train, svd_test),
            'tfidf_svd_normalized': (normalize(svd_train), normalize(svd_test)),
            'sentence_embeddings': (embeddings[train], embeddings[test]),
        }
        metrics, class_rows, assignments = [], [], []
        for method, (a, b) in representations.items():
            for seed in args.seeds:
                start = time.perf_counter()
                km = KMeans(n_clusters=args.k, random_state=seed, n_init=10)
                train_clusters, test_clusters = km.fit_predict(a), km.predict(b)
                scores, rows = evaluate(y_train, y_test, train_clusters, test_clusters, args.min_cluster_size)
                scores.update(method=method, seed=seed, k=args.k, baseline_accuracy=baseline_accuracy,
                              accuracy_lift=scores['test_accuracy']-baseline_accuracy,
                              clustering_seconds=time.perf_counter()-start)
                metrics.append(scores)
                class_rows.extend(dict(row, method=method, seed=seed) for row in rows)
                clusters = np.empty(len(df), dtype=int)
                clusters[train], clusters[test] = train_clusters, test_clusters
                assignments.append(df[['id', 'future_subcat']].assign(split=split, method=method, seed=seed, cluster=clusters))
                print(f"{method}, seed {seed}: AMI={scores['test_ami']:.4f}, accuracy={scores['test_accuracy']:.1%}", flush=True)
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(out/'metrics_by_seed.csv', index=False)
    pd.DataFrame(class_rows).to_csv(out/'class_comparison.csv', index=False)
    pd.concat(assignments, ignore_index=True).to_csv(out/'cluster_assignments.csv', index=False)
    numeric = ['test_ami', 'test_ari', 'test_accuracy', 'test_macro_f1', 'accuracy_lift']
    averages = metrics.groupby('method')[numeric].mean()
    averages.to_csv(out/'comparison_mean.csv')
    deviations = metrics.groupby('method')[numeric].std()
    deviations.to_csv(out/'comparison_seed_std.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    names = ['TF-IDF/SVD', 'TF-IDF/SVD\nnormalized', 'MiniLM']
    order = list(representations)
    for ax, metric, title in zip(axes, ['test_ami', 'test_accuracy', 'test_macro_f1'],
                                  ['Adjusted mutual information', 'Held-out accuracy', 'Macro F1']):
        ax.bar(names, averages.loc[order, metric], yerr=deviations.loc[order, metric].fillna(0),
               capsize=4, color=['#64748b', '#94a3b8', '#0f766e'])
        ax.set_title(title)
        ax.set_ylim(bottom=min(0, float(averages[metric].min())-.02))
        if metric == 'test_accuracy':
            ax.axhline(baseline_accuracy, color='#b45309', linestyle='--', label='Majority baseline')
            ax.legend(fontsize=8)
    fig.suptitle('Same papers and k; mean across seeds, error bars = seed standard deviation')
    fig.tight_layout()
    fig.savefig(out/'comparison.png', dpi=180)
    plt.close(fig)
    manifest = vars(args).copy()
    manifest.update(input=str(args.input.resolve()), output=str(out), sample_actual=len(df),
                    train_size=len(train), test_size=len(test), truncated_abstracts=truncated,
                    max_seq_length=model.max_seq_length, resolved_model_revision=resolved_revision, embedding_cache=cache.name,
                    embedding_seconds=embedding_seconds, tfidf_seconds=tfidf_seconds,
                    svd_components=components, svd_explained_variance=float(svd.explained_variance_ratio_.sum()),
                    packages={n: importlib.metadata.version(n) for n in ['sentence-transformers', 'torch', 'scikit-learn', 'numpy', 'pandas']})
    (out/'run_config.json').write_text(json.dumps(manifest, indent=2, default=str), encoding='utf-8')
    report = ['# Text representation comparison', '',
              f'{len(df):,} papers; {len(train):,} training and {len(test):,} held-out papers. k={args.k}.',
              f'Input: `{args.input.name}`. Majority-class test accuracy: {baseline_accuracy:.1%}.',
              '', '![Comparison](comparison.png)', '', '| Method | AMI | ARI | Accuracy | Macro F1 | Accuracy lift |',
              '|---|---:|---:|---:|---:|---:|']
    for method, row in averages.iterrows():
        report.append(f"| {method} | {row.test_ami:.4f} | {row.test_ari:.4f} | {row.test_accuracy:.1%} | {row.test_macro_f1:.4f} | {100*row.accuracy_lift:+.1f} pp |")
    delta_original = averages.loc['sentence_embeddings', 'test_accuracy']-averages.loc['tfidf_svd', 'test_accuracy']
    delta_control = averages.loc['sentence_embeddings', 'test_accuracy']-averages.loc['tfidf_svd_normalized', 'test_accuracy']
    report += ['', f'MiniLM accuracy difference: {100*delta_original:+.2f} percentage points versus original TF-IDF/SVD, '
               f'and {100*delta_control:+.2f} points versus normalized TF-IDF/SVD. No significance test has been performed.']
    report += ['', 'Means across seeds; see comparison_seed_std.csv for variability. Seed variation is not a confidence interval.',
               '', 'TF-IDF vocabulary, SVD, K-means, cluster-to-label mapping, and best class-cluster selection use training papers only.',
               'The normalized TF-IDF control helps distinguish the effect of vector normalization from the embedding method.',
               'Class concentration and coverage in class_comparison.csv are measured on test papers for clusters selected on training papers.',
               f'{truncated:,}/{len(df):,} abstracts exceed the embedding model limit of {model.max_seq_length} tokens and were truncated.',
               '', 'Limitations: author-history labels are proxies; shared authors can connect train/test papers. This is a paper holdout, not author-disjoint validation.',
               'Modern pretrained embeddings may contain post-2009 knowledge. This comparison is retrospective, not evidence of historical deployment or field emergence.',
               'This subset comparison does not directly reproduce the full-dataset RAPIDS experiment.']
    (out/'RESULTS.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    print(averages.to_string(), flush=True)
    print(f'Results saved to {out}', flush=True)


if __name__ == '__main__':
    main()
