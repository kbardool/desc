"""Build kNN graphs (scanpy 1.9.8, as desc 2.1.1 did) and partition them with the `louvain` package."""
from paths import DATA, DET
import sys, numpy as np, scipy.sparse as sp, scanpy as sc
RES = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.6, 2.0]; SEEDS = range(5)
for dataset in ("pbmc", "paul"):
    graphs = {"embedding": np.load(f"{DET}/tf_{dataset}.npz")["features"],   # DESC's actual input to louvain
              "pca": sc.tl.pca(np.load(f"{DATA}/{dataset}.npy"), n_comps=30, random_state=0)}
    for gname, rep in graphs.items():
        a = sc.AnnData(rep.astype(np.float32))
        sc.pp.neighbors(a, n_neighbors=10, use_rep="X")
        out = {"res": np.array(RES)}
        for r in RES:
            for s in SEEDS:
                sc.tl.louvain(a, resolution=r, random_state=s, key_added="l")
                out[f"labels_{r}_{s}"] = np.asarray(a.obs["l"].astype(int))
        sp.save_npz(f"{DET}/graph_{dataset}_{gname}.npz", sp.csr_matrix(a.obsp["connectivities"]))
        np.savez(f"{DET}/louvainpkg_{dataset}_{gname}.npz", **out)
        print(dataset, gname, {r: len(np.unique(out[f"labels_{r}_0"])) for r in RES})
