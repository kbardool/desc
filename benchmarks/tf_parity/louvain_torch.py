"""Partition the same graphs with desc 3's igraph-based louvain()."""
from paths import DET
import numpy as np, scipy.sparse as sp
from desc.models.network import louvain
RES = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.6, 2.0]; SEEDS = range(5)
for dataset in ("pbmc", "paul"):
    for gname in ("embedding", "pca"):
        g = sp.load_npz(f"{DET}/graph_{dataset}_{gname}.npz")
        out = {f"labels_{r}_{s}": louvain(g, resolution=r, random_state=s) for r in RES for s in SEEDS}
        np.savez(f"{DET}/igraph_{dataset}_{gname}.npz", **out)
        print(dataset, gname, {r: len(np.unique(out[f"labels_{r}_0"])) for r in RES})
