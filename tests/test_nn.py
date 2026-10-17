import numpy as np
import pytest
import torch

from desc.models import _nn
from desc.models.network import ClusteringLayer, ClusteringLayerGaussian, louvain

rng = np.random.default_rng(0)


def test_kld_matches_keras_formula():
    p = rng.dirichlet(np.ones(5), size=40).astype(np.float32)
    q = rng.dirichlet(np.ones(5), size=40).astype(np.float32)
    pc, qc = np.clip(p, 1e-7, 1), np.clip(q, 1e-7, 1)
    expected = np.mean(np.sum(pc * np.log(pc / qc), axis=-1))
    assert _nn.kld(torch.tensor(p), torch.tensor(q)).item() == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize("alpha", [1.0, 2.5])
def test_student_t_clustering_layer(alpha):
    x = rng.normal(size=(20, 4)).astype(np.float32)
    mu = rng.normal(size=(3, 4)).astype(np.float32)
    q = 1.0 / (1.0 + ((x[:, None, :] - mu) ** 2).sum(2) / alpha)
    q **= (alpha + 1.0) / 2.0
    q /= q.sum(1, keepdims=True)
    layer = ClusteringLayer(3, weights=[mu], alpha=alpha)
    np.testing.assert_allclose(layer(torch.tensor(x)).detach().numpy(), q, atol=1e-6)
    np.testing.assert_array_equal(layer.get_weights()[0], mu)


def test_gaussian_clustering_layer():
    x = rng.normal(size=(20, 4)).astype(np.float32)
    mu = rng.normal(size=(3, 4)).astype(np.float32)
    q = np.exp(-((x[:, None, :] - mu) ** 2) / 2.0).sum(2)
    q /= q.sum(1, keepdims=True)
    out = ClusteringLayerGaussian(3, weights=[mu])(torch.tensor(x)).detach().numpy()
    np.testing.assert_allclose(out, q, atol=1e-6)


def test_clustering_layer_needs_a_dimension():
    with pytest.raises(ValueError):
        ClusteringLayer(3)
    assert ClusteringLayer(3, input_dim=7).clusters.shape == (3, 7)


def test_dense_weights_use_keras_layout():
    layer = _nn.Dense(5, 3, activation="relu", name="encoder_0")
    kernel, bias = rng.normal(size=(5, 3)), rng.normal(size=3)
    layer.set_weights([kernel, bias])
    got_kernel, got_bias = layer.get_weights()
    np.testing.assert_allclose(got_kernel, kernel, atol=1e-6)
    np.testing.assert_allclose(got_bias, bias, atol=1e-6)
    x = rng.normal(size=(4, 5)).astype(np.float32)
    np.testing.assert_allclose(layer(torch.tensor(x)).detach().numpy(), np.maximum(x @ kernel + bias, 0), atol=1e-5)


def test_initializers():
    for init in _nn.INITIALIZERS:
        w = torch.empty(64, 32)
        _nn.init_kernel_(w, init)
        assert w.std() > 0
    with pytest.raises(ValueError):
        _nn.init_kernel_(torch.empty(2, 2), "orthogonal")


def test_unknown_activation():
    with pytest.raises(ValueError):
        _nn.Dense(3, 2, activation="relu6")


def test_early_stopping_follows_keras():
    es = _nn.EarlyStopping(min_delta=1e-4, patience=2)
    assert not es.should_stop(1.0)
    assert not es.should_stop(0.99995)  # improvement smaller than min_delta
    assert es.should_stop(0.99999)
    es = _nn.EarlyStopping(min_delta=1e-4, patience=2)
    assert [es.should_stop(v) for v in (1.0, 0.9, 0.95, 0.8, 0.85, 0.85)] == [False, False, False, False, False, True]


def test_get_optimizer_and_loss():
    params = list(_nn.Dense(3, 2).parameters())
    opt = _nn.get_optimizer(_nn.SGD(0.01, 0.9), params)
    assert isinstance(opt, torch.optim.SGD) and opt.defaults["momentum"] == 0.9
    assert _nn.get_optimizer("sgd", params).defaults["lr"] == 0.01
    assert _nn.SGD(lr=0.5).learning_rate == 0.5
    assert _nn.get_loss("kld") is _nn.kld and _nn.get_loss("mse") is _nn.mse
    with pytest.raises(ValueError):
        _nn.get_optimizer("adam", params)


def test_louvain_separates_blocks_and_sorts_by_size():
    import scipy.sparse as sp

    big, small = np.ones((6, 6)), np.ones((3, 3))
    adjacency = sp.block_diag([big, small]).tolil()
    adjacency.setdiag(0)
    adjacency[0, 6] = adjacency[6, 0] = 0.01  # weak bridge
    labels = louvain(adjacency.tocsr(), resolution=1.0)
    np.testing.assert_array_equal(labels, [0] * 6 + [1] * 3)
