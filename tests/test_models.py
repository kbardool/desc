import h5py
import numpy as np
import pytest
import torch

from desc.models.network import DescModel, load_model
from desc.models.SAE import SAE

rng = np.random.default_rng(0)


def test_sae_structure_and_shared_layers(tmp_path):
    sae = SAE(dims=[20, 8, 4], save_dir=str(tmp_path))
    assert list(sae.autoencoders.layers) == ["encoder_0", "encoder_1", "decoder_1", "decoder_0"]
    assert list(sae.encoder.layers) == ["encoder_0", "encoder_1"]
    assert sae.stacks[1].get_layer("encoder_1") is sae.autoencoders.get_layer("encoder_1")
    # hidden and output layers use actincenter, internal layers use act
    acts = {name: layer.activation_name for name, layer in sae.autoencoders.layers.items()}
    assert acts == {"encoder_0": "relu", "encoder_1": "tanh", "decoder_1": "relu", "decoder_0": "tanh"}
    x = rng.normal(size=(50, 20)).astype(np.float32)
    assert sae.extract_feature(x).shape == (50, 4)
    assert sae.autoencoders.predict(x).shape == (50, 20)


def test_sae_fit_reduces_loss(tmp_path):
    x = rng.normal(size=(200, 10)).astype(np.float32) * 0.3
    sae = SAE(dims=[10, 6, 3], save_dir=str(tmp_path), batch_size=32)
    before = np.mean((sae.autoencoders.predict(x) - x) ** 2)
    sae.fit(x, epochs=20)
    assert np.mean((sae.autoencoders.predict(x) - x) ** 2) < before


def test_sae_is_reproducible(tmp_path):
    x = rng.normal(size=(100, 10)).astype(np.float32)
    feats = []
    for _ in range(2):
        sae = SAE(dims=[10, 4], save_dir=str(tmp_path), random_seed=7)
        sae.fit(x, epochs=6)
        feats.append(sae.extract_feature(x))
    np.testing.assert_array_equal(*feats)


def test_invalid_init(tmp_path):
    with pytest.raises(ValueError):
        SAE(dims=[10, 4], init="orthogonal", save_dir=str(tmp_path))


def test_save_and_load_model(tmp_path):
    sae = SAE(dims=[12, 6, 3], save_dir=str(tmp_path))
    x = rng.normal(size=(5, 12)).astype(np.float32)
    sae.autoencoders.save(tmp_path / "autoencoder_model.pt")
    np.testing.assert_array_equal(load_model(tmp_path / "autoencoder_model.pt").predict(x), sae.autoencoders.predict(x))
    sae.encoder.save_weights(tmp_path / "encoder_weights.pt")
    other = SAE(dims=[12, 6, 3], save_dir=str(tmp_path), random_seed=1)
    other.encoder.load_weights(tmp_path / "encoder_weights.pt")
    np.testing.assert_array_equal(other.extract_feature(x), sae.extract_feature(x))


def write_keras_weights(path, layers):
    """Write `layers` ({name: [kernel, bias]}) in the HDF5 layout of tf.keras `Model.save_weights`."""
    with h5py.File(path, "w") as f:
        f.attrs["layer_names"] = [b"input"] + [name.encode() for name in layers]
        f.create_group("input").attrs["weight_names"] = np.array([], dtype="S")
        for name, (kernel, bias) in layers.items():
            g = f.create_group(name)
            g.attrs["weight_names"] = [("%s/kernel:0" % name).encode(), ("%s/bias:0" % name).encode()]
            g.create_dataset("%s/kernel:0" % name, data=kernel)
            g.create_dataset("%s/bias:0" % name, data=bias)


def test_load_desc2_keras_weights(tmp_path):
    dims = [12, 6, 3]
    shapes = {"encoder_0": (12, 6), "encoder_1": (6, 3), "decoder_1": (3, 6), "decoder_0": (6, 12)}
    layers = {n: [rng.normal(size=s).astype(np.float32), rng.normal(size=s[1]).astype(np.float32)] for n, s in shapes.items()}
    write_keras_weights(tmp_path / "ae_weights.h5", layers)
    sae = SAE(dims=dims, save_dir=str(tmp_path))
    sae.autoencoders.load_weights(str(tmp_path / "ae_weights.h5"))
    x = rng.normal(size=(4, 12)).astype(np.float32)
    h = np.maximum(x @ layers["encoder_0"][0] + layers["encoder_0"][1], 0)
    h = np.tanh(h @ layers["encoder_1"][0] + layers["encoder_1"][1])
    np.testing.assert_allclose(sae.extract_feature(x), h, atol=1e-5)


def blobs(n_per_cluster=60, n_features=20, n_clusters=3):
    centers = rng.normal(scale=3.0, size=(n_clusters, n_features))
    x = np.concatenate([c + rng.normal(size=(n_per_cluster, n_features)) for c in centers])
    return (x / np.abs(x).max()).astype(np.float32), np.repeat(np.arange(n_clusters), n_per_cluster)


def test_desc_model_end_to_end(tmp_path):
    x, _ = blobs()
    desc = DescModel(dims=[20, 8], x=x, pretrain_epochs=10, epochs_fit=2, batch_size=32, save_dir=str(tmp_path), n_neighbors=10)
    assert desc.n_clusters >= 2
    with pytest.raises(RuntimeError):
        desc.fit(maxiter=2)
    desc.compile(optimizer="sgd", loss="kld")
    z, q = desc.fit(maxiter=3)
    assert z.shape == (x.shape[0], 8)
    assert q.shape == (x.shape[0], desc.n_clusters)
    np.testing.assert_allclose(q.sum(1), 1.0, rtol=1e-5)
    assert set(np.unique(desc.predict(x))) <= set(range(desc.n_clusters))
    for f in ("ae_weights.pt", "encoder_weights.pt", "autoencoder_model.pt", "encoder_model.pt"):
        assert (tmp_path / f).is_file()


def test_desc_model_fit_on_batch(tmp_path):
    x, _ = blobs()
    desc = DescModel(dims=[20, 8], x=x, pretrain_epochs=10, epochs_fit=0.5, batch_size=32, save_dir=str(tmp_path))
    desc.compile()
    z, q = desc.fit(maxiter=20)
    assert q.shape == (x.shape[0], desc.n_clusters)


def test_desc_model_keeps_torch_thread_count(tmp_path):
    # scanpy's neighbors can reset the OpenMP thread count torch shares (num_Cores must hold for training)
    x, _ = blobs()
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        DescModel(dims=[20, 8], x=x, pretrain_epochs=2, batch_size=32, save_dir=str(tmp_path))
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(before)
