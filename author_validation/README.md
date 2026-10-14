# Author-label improvement and comparison

Code and generated outputs are isolated in `author_validation/`. Original labels,
working data, and clustering files remain in the project root. Run from the project root:

```powershell
.\.venv\Scripts\python.exe author_validation/author_lookup.py
```

With the current working set, this improves voting only. Each distinct author contributes one total vote, distributed according to their post-split primary-category publication counts. Ties are unresolved. Confidence is vote concentration, not a calibrated probability. High-confidence means confidence >= 0.7, margin >= 0.2 and at least two matched authors; these are heuristic thresholds.

To enable exact full-name matching, obtain the original arXiv metadata snapshot and run:

```powershell
.\.venv\Scripts\python.exe author_validation/author_lookup.py --raw "C:\path\arxiv-metadata-oai-snapshot.json"
```

Names use normalized surname, given names and suffix. Initial-only names are excluded rather than guessed. Different spellings or abbreviated variants may fragment identities; identical full names can still collide. This reduces some collisions but is not complete author disambiguation. The raw snapshot must contain every ID in the saved working set. Future runs of load_filter_astroph.py also retain full_author_keys automatically.

Output: `author_validation/results/astroph_pre_with_future_labels_new.parquet`.
Existing clustering scripts still read the original file; change their input explicitly
when you want a new clustering experiment. Keep outputs separate. Use `--output`
to choose another results directory; `--input` selects another working-set file.

Comparison is generated automatically, or rerun:

```powershell
.\.venv\Scripts\python.exe author_validation/compare_author_labels.py
```

The `results/` folder contains summary.csv, label_changes.csv, class_counts.csv/png, label_transition.csv/png and fixed_cluster_comparison.csv/fixed_cluster_purity.png. Distribution comparisons use identical paper IDs. Cluster comparisons use fixed existing GPU assignments and the same papers with labels under both methods; they isolate label changes, not new clustering performance. Higher purity against a different proxy is not evidence of greater accuracy. Full-name and voting effects should be evaluated separately when raw metadata becomes available.

The final [results report](results/RESULTS.md) includes `class_lift_comparison.csv`
and its chart. Each category is compared on its original-label-selected best
cluster, using identical commonly labeled papers and a minimum cluster size of 50.
Lift means cluster category concentration minus that category's baseline rate;
the change is reported in percentage points. New-label-selected best clusters
are separate columns, so reselection does not inflate the paired comparison.
The report reads `label_method` and explicitly distinguishes voting-only runs
from runs that actually used full-name keys. It does not claim clustering was rerun.

Check the lift calculation from the project root:

```powershell
.\.venv\Scripts\python.exe -m unittest author_validation.test_class_lift
```

Both scripts locate their default data relative to the code, so they also work
when launched from inside this folder. Module invocation from the project root
is supported: `python -m author_validation.author_lookup`.

To compare text representations using these improved labels:

```powershell
.\.venv\Scripts\python.exe text_representation/compare.py --input author_validation/results/astroph_pre_with_future_labels_new.parquet --output text_representation/results_new_labels
```
