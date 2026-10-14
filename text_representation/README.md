# Issue 4: richer text representations

This independent experiment compares three representations on identical papers:

1. Existing TF-IDF (8,000 features) → SVD (300 components) approach.
2. The same approach with unit-normalized vectors, as a normalization control.
3. `sentence-transformers/all-MiniLM-L6-v2` embeddings of lightly cleaned original abstracts.

All methods use scikit-learn K-means, k=40, n_init=10, and seeds 42/43/44.
The default uses a reproducible, stratified 15,000-paper sample of the ORIGINAL
author labels and an 80/20 paper split. Keeping labels fixed isolates the text
representation comparison from the separate author-label experiment.

Run from the project root (Windows PowerShell):

```powershell
.\.venv\Scripts\python.exe -m pip install -r text_representation/requirements.txt
.\.venv\Scripts\python.exe text_representation/compare.py
```

The first run downloads the model into this experiment's results/model_cache.
CPU is the default; WSL2/RAPIDS are unnecessary. Embeddings are cached by text,
model name, revision, and token limit. For reproducible reruns across machines,
pass `--revision` with a Hugging Face model commit hash.

To compare improved author labels separately:

```powershell
.\.venv\Scripts\python.exe text_representation/compare.py --input author_validation/results/astroph_pre_with_future_labels_new.parquet --output text_representation/results_new_labels
```

Options: `--sample 0` for the full resolved set, `--k 60`, `--device cuda` for a
CUDA-enabled PyTorch installation, and `--seeds 42 43 44`.
Use a separate `--output` directory for each distinct experiment.

Outputs include RESULTS.md, comparison.png, per-seed metrics, mean/std metrics, per-class test
concentration/lift/coverage, paper split, assignments, and run configuration.
Accuracy learns a cluster-to-proxy-label mapping on training data and evaluates
it on held-out papers. AMI/ARI compare held-out clusters with proxy labels.
Best per-class clusters are selected on training papers only (minimum 50 papers).

This is not a comparison with the report's selected-cluster 44.3% statistic.
The original full GPU outputs and project scripts remain untouched.

Limitations: proxy-label noise, shared authors across the paper split, embedding
truncation (counted in run_config.json), and modern model pretraining. The result
tests retrospective topic recovery, not whether a field was emerging historically.

Validation:

```powershell
.\.venv\Scripts\python.exe text_representation/test_compare.py
```

The tests verify that held-out labels cannot choose a cluster's predicted label
or the best cluster for a class, and that size filters are respected.

Model references:
- https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
