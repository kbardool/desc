import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import desc

FAST = dict(dims=[100, 32, 8], pretrain_epochs=10, max_iter=3, num_Cores=2, verbose=False)


@pytest.fixture(scope="module")
def pbmc():
    adata = desc.datasets.pbmc_processed()
    return adata[::4].copy()


def test_version_matches_pyproject():
    pyproject = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    assert desc.__version__ == pyproject["project"]["version"] == "3.0.0"


def test_train_anndata_multiple_resolutions(pbmc, tmp_path):
    adata = desc.train(pbmc.copy(), louvain_resolution="0.8,1.0", save_dir=str(tmp_path), use_ae_weights=False, **FAST)
    for res in ("0.8", "1.0"):
        labels = adata.obs["desc_" + res]
        assert isinstance(labels.dtype, pd.CategoricalDtype)
        assert list(labels.cat.categories) == list(range(len(labels.cat.categories)))
        assert adata.obsm["X_Embeded_z" + res].shape == (pbmc.n_obs, 8)
        prob = adata.uns["prob_matrix" + res]
        assert prob.shape[0] == pbmc.n_obs and prob.shape[1] >= labels.cat.categories.size
        np.testing.assert_allclose(prob.sum(1), 1.0, rtol=1e-5)
    assert (tmp_path / "ae_weights.pt").is_file()
    adata.write_h5ad(tmp_path / "adata_desc.h5ad")


def test_train_reuses_pretrained_weights(pbmc, tmp_path):
    first = desc.train(pbmc.copy(), louvain_resolution=0.8, save_dir=str(tmp_path), use_ae_weights=False, **FAST)
    second = desc.train(pbmc.copy(), louvain_resolution=0.8, save_dir=str(tmp_path), use_ae_weights=True, **FAST)
    np.testing.assert_array_equal(first.obs["desc_0.8"], second.obs["desc_0.8"])


@pytest.mark.parametrize("kind", ["ndarray", "sparse", "dataframe"])
def test_train_other_inputs(pbmc, tmp_path, kind):
    x = {
        "ndarray": lambda: np.asarray(pbmc.X),
        "sparse": lambda: sp.csr_matrix(pbmc.X),
        "dataframe": lambda: pd.DataFrame(pbmc.X, index=pbmc.obs_names, columns=pbmc.var_names),
    }[kind]()
    adata = desc.train(x, louvain_resolution=1.0, save_dir=str(tmp_path), use_ae_weights=False, kernel_clustering="gaussian", **FAST)
    assert adata.n_obs == pbmc.n_obs and "desc_1.0" in adata.obs
    if kind == "dataframe":
        assert list(adata.obs_names) == list(pbmc.obs_names)
    if kind == "sparse":
        assert sp.issparse(adata.X)


def test_get_device_falls_back_to_cpu():
    from desc.models.desc import get_device

    assert get_device(use_GPU=False, GPU_id=0).type == "cpu"
    assert get_device(use_GPU=True, GPU_id=None).type == "cpu"


def test_read_10X():
    adata = desc.read_10X(str(Path(desc.__file__).parent / "datasets" / "pbmc"))
    assert adata.shape == (2700, 32738)
    assert adata.var_names.is_unique and adata.var_names.name is None
    assert adata.obs_names[0] == adata.obs["barcode"].iloc[0]
