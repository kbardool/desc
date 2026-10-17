.. role:: small
.. role:: smaller
.. role:: noteversion

Version 3.0.0 :small:`October 16, 2026`
-------------------------------------

New functionality:

- DESC runs on PyTorch instead of TensorFlow/Keras and requires Python 3.14 or newer.
- The package is built from ``pyproject.toml`` and managed with uv.
- ``use_GPU=True`` with ``GPU_id`` trains on that CUDA device, or on Apple MPS.
- ``desc.models.network.load_model`` reloads the models saved in ``save_dir``.

Updates:

- The public API (``desc.train``, ``DescModel``, ``SAE``, ...) is unchanged.
- Weights in ``save_dir`` are saved as ``.pt`` files. ``ae_weights.h5`` from desc 2.x can still be loaded with ``use_ae_weights=True``.
- The Louvain initialization uses igraph, because the ``louvain`` package no longer installs on current Python versions.
- ``alpha`` is now passed to the clustering kernel.
- Sparse and ``pandas.DataFrame`` input to ``desc.train`` works.
- A single detected cluster raises ``ValueError`` instead of exiting Python.
- Training no longer changes ``CUDA_VISIBLE_DEVICES`` or the matplotlib backend.

Version 2.1.1 :small:`July 25, 2020`
-------------------------------------

Updates:

- Compatible with TensorFlow 2 and scanpy 1.4 or newer.

Version 2.0.3 :small:`July 25, 2020`
-------------------------------------

Updates:

- Compatible with scanpy 1.4 or newer, still with TensorFlow 1.

Version 1.0.0.3 :small:`July 10, 2019`
-------------------------------------

New functionality:
- here add some notes about new funcs details

Updates:

- here add some notes for updating detail
