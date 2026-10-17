# desc 3.0 vs 2.1.1 parity test

Runs the TensorFlow release (desc 2.1.1, commit `e4116d6`) next to the PyTorch port (desc 3.0) on the same inputs and compares the results. It checks three things.

1. **Deterministic parity.** Both versions load the autoencoder 2.1.1 pretrained (`ae_weights.h5`), and 3.0 is given 2.1.1's Louvain partition. Clustering then runs with `epochs_fit=0.4`, which trains on fixed, unshuffled batches, so no randomness is left. The script compares encoder outputs, soft assignments `q` before and after one SGD step, the `delta_label` trace and the final labels. The `long` variant sets `tol=-1` so that all 1,000 iterations run (200 target updates).
2. **Louvain substitution.** desc 3.0 replaces the `louvain` package, which no longer installs, with igraph's `community_multilevel`. Both partition the same kNN graphs over 8 resolutions × 5 seeds. The script compares cluster counts, modularity, and ARI within and across implementations.
3. **End-to-end statistics.** `desc.train` runs 3 times per version and dataset (seeds 201809, 1, 2). The script compares ARI between versions against ARI between seeds of one version, and ARI/NMI against the published Paul15 labels.

Datasets: PBMC 3k (`desc.datasets.pbmc_processed`, 2,700 × 100, dims `[100, 64, 16]`) and Paul15 (2,730 cells, preprocessed as in `desc_2.1.1_paul.ipynb`, dims `[1000, 64, 32]`). Paul15 is read from `desc/original/data/paul15/paul15.h5` in the repository, so nothing is downloaded.

## Run it

```bash
# from the repository root; behind a TLS-intercepting proxy, export UV_NATIVE_TLS=1 first
benchmarks/tf_parity/run_all.sh            # sequential
benchmarks/tf_parity/run_all.sh --parallel # 12 end-to-end trainings at once
```

The script creates a Python 3.11 environment for 2.1.1 (TensorFlow 2.15.1, louvain 0.8.2, scanpy 1.9.8) and syncs the repository environment for 3.0. It extracts the 2.1.1 code with `git archive` and writes everything under `$DESC_PARITY_DIR` (default `./parity_work`).

On a 12-core CPU without a GPU, a full sequential run takes roughly 20 to 40 minutes after the environments are installed. Each end-to-end training takes 20 to 45 s, and each long deterministic run takes a few minutes.

Results land in `det/compare_det.json`, `louvain.json` and `e2e.json` in the work directory. The results from the original run (16 Oct 2026) are in `results/`.

| Script | Environment | Purpose |
|---|---|---|
| `paths.py` | both | work-directory layout (`DESC_PARITY_DIR`, `DESC_TF_SRC`) |
| `prep_data.py` | 3.0 | writes the shared float32 input matrices and Paul15 labels |
| `run_train.py` | both | one `desc.train` run (resolutions 0.8 and 1.0, `num_Cores=1`) |
| `common_side.py`, `tf_side.py`, `torch_side.py` | 2.1.1 / 3.0 | deterministic comparison drivers |
| `louvain_tf.py`, `louvain_torch.py` | 2.1.1 / 3.0 | Louvain partitions on shared graphs |
| `compare_det.py`, `compare_louvain.py`, `compare_e2e.py` | 3.0 | summary statistics as JSON |

## Reference results

**Deterministic parity.** Values are the largest absolute difference between versions.

| Run | Encoder output | Final q | Target updates | Same final label |
|---|---|---|---|---|
| PBMC | 1.8e-7 | 3.0e-8 | 2 / 2 | 100% |
| PBMC, long | 1.8e-7 | 1.5e-6 | 200 / 200 | 100% |
| Paul15 | 8.0e-7 | 1.2e-7 | 2 / 2 | 100% |
| Paul15, long | 8.0e-7 | 4.8e-7 | 200 / 200 | 100% |

**Louvain substitution.** Across 32 settings (2 datasets × 2 graph types × 8 resolutions), igraph reached higher modularity in 29, tied in 2 and was lower in 1, by 0.001. Partitions from the two implementations agree less with each other than each agrees with itself, by 0.07 ARI on average and at most 0.13.

**End-to-end runs.** Mean pairwise ARI between runs:

| Dataset, resolution | 2.1.1 vs 2.1.1 | 3.0 vs 3.0 | 2.1.1 vs 3.0 |
|---|---|---|---|
| PBMC 0.8 | 0.686 | 0.574 | 0.598 |
| PBMC 1.0 | 0.733 | 0.706 | 0.716 |
| Paul15 0.8 | 0.807 | 0.814 | 0.784 |
| Paul15 1.0 | 0.876 | 0.847 | 0.857 |

Against the 19 published Paul15 clusters, mean ARI is 0.316 (2.1.1) vs 0.341 (3.0) at resolution 0.8, and 0.336 vs 0.343 at 1.0.

## Limits

CPU only. Two datasets and three seeds per version. 2.1.1 runs on TensorFlow 2.15 and louvain 0.8.2, not the TensorFlow 2.1 and louvain 0.6.1 of the paper's notebook. The default training path (integer `epochs_fit`) shuffles batches, so it is compared statistically only.

The `runtime_s` values in `results/e2e.json` were measured under different load and aren't comparable. The 2.1.1 runs shared the machine with 11 other trainings. The 3.0 runs were repeated afterwards, 6 at a time, once a thread-count bug in 3.0 was fixed. On an idle machine with one core, one run took 34.8 s (2.1.1) vs 22.8 s (3.0) on PBMC, and 41.8 s vs 36.3 s on Paul15.
