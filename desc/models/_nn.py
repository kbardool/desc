"""
PyTorch building blocks reproducing the Keras behaviour DESC 2.x relied on:
weight initializers, named Dense/Dropout layers, a Sequential model with the
Keras methods DESC used (predict, summary, get_layer, save_weights, ...), the
SGD optimizer, the 'mse' and 'kld' losses, EarlyStopping and the fit loop.
"""
import math
import random

import numpy as np
import torch
from scipy.sparse import issparse
from torch import nn

INITIALIZERS = (
    "glorot_uniform",
    "glorot_normal",
    "he_normal",
    "he_uniform",
    "lecun_normal",
    "lecun_uniform",
    "RandomNormal",
    "RandomUniform",
    "TruncatedNormal",
)

# Keras' *_normal variance-scaling initializers draw from a normal truncated at
# two standard deviations, with stddev rescaled to keep the intended variance.
_TRUNCATED_STDDEV_CORRECTION = 0.87962566103423978

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "elu": nn.ELU,
    "selu": nn.SELU,
    "softplus": nn.Softplus,
    "softsign": nn.Softsign,
    "gelu": nn.GELU,
    "swish": nn.SiLU,
    "silu": nn.SiLU,
    "linear": nn.Identity,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def as_float32_array(x):
    """Dense float32 numpy copy (or view) of a numpy array, sparse matrix or DataFrame."""
    if issparse(x):
        x = x.toarray()
    return np.ascontiguousarray(np.asarray(x, dtype=np.float32))


def model_device(model):
    for p in model.parameters():
        return p.device
    return torch.device("cpu")


def get_activation(activation):
    if activation is None:
        return nn.Identity()
    if activation not in _ACTIVATIONS:
        raise ValueError(
            "Invalid activation %r: expected one of %s" % (activation, sorted(_ACTIVATIONS))
        )
    return _ACTIVATIONS[activation]()


def _truncated_normal_(w, stddev):
    return nn.init.trunc_normal_(w, mean=0.0, std=stddev, a=-2.0 * stddev, b=2.0 * stddev)


def init_kernel_(w, init="glorot_uniform"):
    """Initialize a torch (out_features, in_features) weight like the named Keras initializer."""
    fan_out, fan_in = w.shape
    if init == "glorot_uniform":
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        return nn.init.uniform_(w, -limit, limit)
    if init == "glorot_normal":
        return _truncated_normal_(w, math.sqrt(2.0 / (fan_in + fan_out)) / _TRUNCATED_STDDEV_CORRECTION)
    if init == "he_normal":
        return _truncated_normal_(w, math.sqrt(2.0 / fan_in) / _TRUNCATED_STDDEV_CORRECTION)
    if init == "he_uniform":
        limit = math.sqrt(6.0 / fan_in)
        return nn.init.uniform_(w, -limit, limit)
    if init == "lecun_normal":
        return _truncated_normal_(w, math.sqrt(1.0 / fan_in) / _TRUNCATED_STDDEV_CORRECTION)
    if init == "lecun_uniform":
        limit = math.sqrt(3.0 / fan_in)
        return nn.init.uniform_(w, -limit, limit)
    if init == "RandomNormal":
        return nn.init.normal_(w, mean=0.0, std=0.05)
    if init == "RandomUniform":
        return nn.init.uniform_(w, -0.05, 0.05)
    if init == "TruncatedNormal":
        return _truncated_normal_(w, 0.05)
    raise ValueError("Invalid `init` argument: expected one of %s but got %r" % (INITIALIZERS, init))


class Dense(nn.Module):
    """Fully connected layer followed by an activation, initialized like a Keras Dense layer."""

    def __init__(self, in_features, units, activation=None, kernel_initializer="glorot_uniform", name=None):
        super().__init__()
        self.name = name
        self.in_features = in_features
        self.units = units
        self.activation_name = activation
        self.kernel_initializer = kernel_initializer
        self.linear = nn.Linear(in_features, units)
        self.activation = get_activation(activation)
        init_kernel_(self.linear.weight, kernel_initializer)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.activation(self.linear(x))

    def get_weights(self):
        """[kernel, bias] in the Keras layout, kernel shape (in_features, units)."""
        return [
            self.linear.weight.detach().cpu().numpy().T.copy(),
            self.linear.bias.detach().cpu().numpy().copy(),
        ]

    def set_weights(self, weights):
        kernel, bias = weights
        with torch.no_grad():
            self.linear.weight.copy_(torch.as_tensor(np.asarray(kernel).T))
            self.linear.bias.copy_(torch.as_tensor(np.asarray(bias)))

    def get_config(self):
        return {
            "in_features": self.in_features,
            "units": self.units,
            "activation": self.activation_name,
            "kernel_initializer": self.kernel_initializer,
            "name": self.name,
        }

    def output_dim(self):
        return self.units


class Dropout(nn.Dropout):
    def __init__(self, rate, name=None):
        super().__init__(p=rate)
        self.name = name

    def get_config(self):
        return {"rate": self.p, "name": self.name}

    def output_dim(self):
        return None


class Sequential(nn.Module):
    """A stack of named layers with the subset of the Keras Model API used by DESC."""

    def __init__(self, layers, name=None):
        super().__init__()
        self.name = name
        self.layers = nn.ModuleDict((layer.name, layer) for layer in layers)

    def forward(self, x):
        for layer in self.layers.values():
            x = layer(x)
        return x

    def get_layer(self, name):
        if name in self.layers:
            return self.layers[name]
        raise ValueError("No such layer: %s. Existing layers are: %s" % (name, list(self.layers)))

    def predict(self, x, batch_size=256, verbose=0):
        return predict(self, x, batch_size=batch_size)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def summary(self):
        rows = []
        dim = None
        for layer in self.layers.values():
            dim = layer.output_dim() or dim
            n_params = sum(p.numel() for p in layer.parameters())
            rows.append(("%s (%s)" % (layer.name, type(layer).__name__), "(None, %s)" % dim, str(n_params)))
        widths = [max(len(h), *(len(r[i]) for r in rows)) + 4 for i, h in enumerate(("Layer (type)", "Output Shape", "Param #"))]
        line = "".join(h.ljust(w) for h, w in zip(("Layer (type)", "Output Shape", "Param #"), widths))
        print('Model: "%s"' % self.name)
        print("_" * len(line))
        print(line)
        print("=" * len(line))
        for r in rows:
            print("".join(c.ljust(w) for c, w in zip(r, widths)))
        print("=" * len(line))
        print("Total params: %d" % self.count_params())

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        """Load weights saved by `save_weights`, or a Keras HDF5 weights file written by DESC 2.x."""
        if str(path).endswith(".h5"):
            load_keras_weights(self, path)
        else:
            self.load_state_dict(torch.load(path, map_location=model_device(self), weights_only=True))

    def save(self, path):
        """Save the architecture and weights so `desc.models.network.load_model` can rebuild the model."""
        torch.save(
            {
                "name": self.name,
                "layers": [(type(layer).__name__, layer.get_config()) for layer in self.layers.values()],
                "state_dict": self.state_dict(),
            },
            path,
        )


def _decode_names(names):
    return [n.decode("utf8") if isinstance(n, bytes) else str(n) for n in names]


def load_keras_weights(model, path):
    """Copy weights from a Keras HDF5 file (`save_weights` or `save`) into `model`, matching layers by name."""
    import h5py

    with h5py.File(path, "r") as f:
        group = f["model_weights"] if "model_weights" in f else f
        for layer_name in _decode_names(group.attrs["layer_names"]):
            weight_names = _decode_names(group[layer_name].attrs["weight_names"])
            if weight_names:
                model.get_layer(layer_name).set_weights([np.asarray(group[layer_name][w]) for w in weight_names])


class SGD(object):
    """
    Keras-style stochastic gradient descent, e.g. ``SGD(0.01, 0.9)``.

    PyTorch optimizers need the parameters at construction time, so this only
    records the hyper-parameters; `build` creates the `torch.optim.SGD` once the
    model is known (this happens in `DescModel.compile`). The momentum update is
    equivalent to Keras' for a constant learning rate.
    """

    def __init__(self, learning_rate=0.01, momentum=0.0, nesterov=False, lr=None):
        self.learning_rate = lr if lr is not None else learning_rate
        self.momentum = momentum
        self.nesterov = nesterov

    def build(self, params):
        return torch.optim.SGD(params, lr=self.learning_rate, momentum=self.momentum, nesterov=self.nesterov)

    def __repr__(self):
        return "SGD(learning_rate=%g, momentum=%g, nesterov=%s)" % (self.learning_rate, self.momentum, self.nesterov)


def get_optimizer(optimizer, params):
    """Resolve 'sgd', an `SGD` spec, a callable params -> optimizer, or a ready torch optimizer."""
    if isinstance(optimizer, torch.optim.Optimizer):
        return optimizer
    if isinstance(optimizer, str):
        if optimizer.lower() != "sgd":
            raise ValueError("Unknown optimizer %r; pass 'sgd', desc.models.network.SGD(...) or a torch optimizer" % optimizer)
        optimizer = SGD()
    if isinstance(optimizer, SGD):
        return optimizer.build(params)
    if callable(optimizer):
        return optimizer(params)
    raise TypeError("Unsupported optimizer: %r" % (optimizer,))


_EPSILON = 1e-7  # keras.backend.epsilon()


def mse(y_true, y_pred):
    return torch.mean((y_pred - y_true) ** 2)


def kld(y_true, y_pred):
    """Keras' kullback_leibler_divergence: clipped to [eps, 1], summed over clusters, averaged over samples."""
    y_true = torch.clamp(y_true, _EPSILON, 1.0)
    y_pred = torch.clamp(y_pred, _EPSILON, 1.0)
    return torch.mean(torch.sum(y_true * torch.log(y_true / y_pred), dim=-1))


_LOSSES = {
    "mse": mse,
    "mean_squared_error": mse,
    "kld": kld,
    "kl_divergence": kld,
    "kullback_leibler_divergence": kld,
}


def get_loss(loss):
    """Resolve a Keras loss name or a callable loss(y_true, y_pred)."""
    if isinstance(loss, str):
        if loss.lower() not in _LOSSES:
            raise ValueError("Unknown loss %r: expected one of %s or a callable" % (loss, sorted(_LOSSES)))
        return _LOSSES[loss.lower()]
    if callable(loss):
        return loss
    raise TypeError("Unsupported loss: %r" % (loss,))


class EarlyStopping(object):
    """Keras EarlyStopping(monitor='loss', mode='min') for one call to `fit`."""

    def __init__(self, min_delta=1e-4, patience=10, verbose=1):
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.best = math.inf
        self.wait = 0

    def should_stop(self, loss):
        if loss + self.min_delta < self.best:
            self.best = loss
            self.wait = 0
            return False
        self.wait += 1
        return self.wait >= self.patience


def train_on_batch(model, optimizer, loss_fn, x, y):
    model.train()
    optimizer.zero_grad()
    loss = loss_fn(y, model(x))
    loss.backward()
    optimizer.step()
    return loss.item()


def fit(model, x, y, optimizer, loss_fn, batch_size=32, epochs=1, shuffle=True, early_stopping=None, verbose=True):
    """Mini-batch training like keras `Model.fit`; returns the per-epoch (sample-weighted mean) losses."""
    device = model_device(model)
    x = torch.as_tensor(as_float32_array(x), device=device)
    y = torch.as_tensor(as_float32_array(y), device=device)
    n = x.shape[0]
    history = []
    for epoch in range(epochs):
        order = torch.randperm(n).to(device) if shuffle else torch.arange(n, device=device)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            total += train_on_batch(model, optimizer, loss_fn, x[idx], y[idx]) * len(idx)
        history.append(total / n)
        if verbose:
            print("Epoch %d/%d - loss: %.4f" % (epoch + 1, epochs, history[-1]))
        if early_stopping is not None and early_stopping.should_stop(history[-1]):
            if early_stopping.verbose:
                print("Epoch %05d: early stopping" % (epoch + 1))
            break
    return history


@torch.no_grad()
def predict(model, x, batch_size=256):
    was_training = model.training
    model.eval()
    device = model_device(model)
    x = as_float32_array(x)
    out = [
        model(torch.as_tensor(x[start:start + batch_size], device=device)).cpu()
        for start in range(0, x.shape[0], batch_size)
    ]
    model.train(was_training)
    return torch.cat(out).numpy()
