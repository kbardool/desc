
## Deep Embedding for Single-cell Clustering (DESC)

DESC is an unsupervised deep learning algorithm for clustering scRNA-seq data. The algorithm constructs a non-linear mapping function from the original scRNA-seq data space to a low-dimensional feature space by iteratively learning cluster-specific gene expression representation and cluster assignment based on a deep neural network. This iterative procedure moves each cell to its nearest cluster, balances biological and technical differences between clusters, and reduces the influence of batch effect. DESC also enables soft clustering by assigning cluster-specific probabilities to each cell, which facilitates the identification of cells clustered with high-confidence and interpretation of results. 

![DESC workflow](docs/assets/images/desc_workflow.png)

For thorough details, see our paper: [https://www.nature.com/articles/s41467-020-15851-3](https://www.nature.com/articles/s41467-020-15851-3)
<br>


## Usage

The [**desc**](https://github.com/eleozzr/desc) package is an implementation of deep embedding for single-cell clustering. With desc, you can:

- Preprocess single cell gene expression data from various formats.
- Build a low-dimensional representation of the single-cell gene expression data.
- Obtain soft-clustering assignments of cells.
- Visualize the cell clustering results and  the  gene expression patterns.

Starting with `desc 3.0`, DESC is implemented in [PyTorch](https://pytorch.org) and requires Python 3.14 or newer. The public API is unchanged: existing calls such as `desc.train(adata, ...)` keep working (see [What changed in 3.0](#what-changed-in-30)). The older releases remain available for their TensorFlow versions:

1. For  `tensorflow 1*`, use `desc(2.0.3)`. See the notebook [desc_2.0.3_paul.ipynb](./desc_2.0.3_paul.ipynb), recorded with that version.
2. For  `tensorflow 2*`, use `desc(2.1.1)`. See the notebook [desc_2.1.1_paul.ipynb](./desc_2.1.1_paul.ipynb), recorded with that version.


<br>

# Installation

`desc 3.0` requires Python `3.14+`. We recommend [uv](https://docs.astral.sh/uv/) for installation:

```bash
# add desc to your uv project
uv add desc
# or install it into the current environment
uv pip install desc
# CPU-only machines can avoid downloading the CUDA build of PyTorch
uv pip install desc --torch-backend cpu
```

`pip install desc` works as well.

To work on desc itself, clone the repository and let uv create the environment. Pick a PyTorch build with an extra: `cpu`, `cu126` or `cu130`. Without an extra, PyTorch comes from PyPI (the CUDA build on Linux).

```bash
git clone https://github.com/eleozzr/desc
cd desc
uv sync --extra cpu
uv run pytest
uv run python -c "import desc; desc.run_desc_test()"
```

## What changed in 3.0

* **PyTorch instead of TensorFlow/Keras.** The architecture, initializers, SGD schedules, losses and early stopping follow the 2.x Keras implementation. Results are not bit-identical to 2.x, because the random number generators differ.
* **GPU.** `use_GPU=True` together with `GPU_id` trains on that CUDA device (or Apple MPS). Without a GPU, desc falls back to the CPU and prints a warning.
* **Files in `save_dir`** are now PyTorch files: `ae_weights.pt`, `encoder_weights.pt`, `autoencoder_model.pt` and `encoder_model.pt`. You can reload a saved model with `desc.models.network.load_model`. With `use_ae_weights=True`, an `ae_weights.h5` written by desc 2.x is still loaded if no `ae_weights.pt` exists.
* **Louvain initialization** now uses igraph's Louvain implementation (`community_multilevel`) with the given `louvain_resolution`. The `louvain` package used before cannot be installed on current Python versions.
* **Bug fixes.**
  * `alpha` is now passed to the t-distribution kernel. Before, it was always 1.0.
  * Sparse and `pandas.DataFrame` input to `desc.train` now works.
  * With `is_stacked=False`, each learning-rate step of the autoencoder pretraining runs 50 epochs, as in the stacked path. Before, each step ran `pretrain_epochs` epochs.
  * When the louvain step finds only one cluster, desc raises a `ValueError` instead of exiting the interpreter.
  * Training no longer changes `CUDA_VISIBLE_DEVICES` or the matplotlib backend.

## Tutorial

The notebook [desc_3.0_paul.ipynb](./desc_3.0_paul.ipynb) walks through clustering the Paul et al. (2015) data with desc 3.0, from preprocessing to t-SNE plots and a comparison with the published labels. It runs on a CPU in about a minute. The same tutorial is available as Markdown in [desc_paul.md](./desc_paul.md).

The [online tutorial](https://eleozzr.github.io/desc/tutorial.html) covers more datasets. Its recorded outputs come from desc 2.x, but its `desc.train` calls work unchanged in 3.0.

[benchmarks/tf_parity](./benchmarks/tf_parity) compares desc 3.0 with desc 2.1.1 side by side.


<br>

## Contributing

Souce code: [Github](https://github.com/eleozzr/desc)  

We are continuing adding new features. Bug reports or feature requests are welcome.

<br>


## References

Please consider citing the following reference:

- Xiangjie Li, Yafei Lyu, Jihwan Park, Jingxiao Zhang, Dwight Stambolian, Katalin Susztak, Gang Hu, Mingyao Li. Deep learning enables accurate clustering and batch effect removal in single-cell RNA-seq analysis. 2019. bioRxiv 530378; doi: [https://doi.org/10.1101/530378](https://www.biorxiv.org/content/10.1101/530378v1?rss=1)
<br>
