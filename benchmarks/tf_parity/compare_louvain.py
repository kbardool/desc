from paths import WORK, DET
import itertools, json, numpy as np, scipy.sparse as sp, igraph
from sklearn.metrics import adjusted_rand_score as ari
RES = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.6, 2.0]; SEEDS = range(5)
def modularity(adj, labels, res):
    e = sp.triu(adj.maximum(adj.T), k=1).tocoo()
    g = igraph.Graph(n=adj.shape[0], edges=np.column_stack((e.row, e.col)).tolist())
    return g.modularity(labels.tolist(), weights=e.data.tolist(), resolution=res)
out = {}
for ds in ("pbmc", "paul"):
    for gname in ("embedding", "pca"):
        adj = sp.load_npz(f"{DET}/graph_{ds}_{gname}.npz")
        pk, ig = np.load(f"{DET}/louvainpkg_{ds}_{gname}.npz"), np.load(f"{DET}/igraph_{ds}_{gname}.npz")
        rows = []
        for r in RES:
            P = [pk[f"labels_{r}_{s}"] for s in SEEDS]; I = [ig[f"labels_{r}_{s}"] for s in SEEDS]
            rows.append({"resolution": r,
                "n_clusters_louvainpkg": float(np.mean([len(np.unique(l)) for l in P])), "n_clusters_igraph": float(np.mean([len(np.unique(l)) for l in I])),
                "ari_cross": float(np.mean([ari(a, b) for a in P for b in I])),
                "ari_within_louvainpkg": float(np.mean([ari(a, b) for a, b in itertools.combinations(P, 2)])),
                "ari_within_igraph": float(np.mean([ari(a, b) for a, b in itertools.combinations(I, 2)])),
                "modularity_louvainpkg": float(np.mean([modularity(adj, l, r) for l in P])),
                "modularity_igraph": float(np.mean([modularity(adj, l, r) for l in I]))})
        out[f"{ds}_{gname}"] = rows
        print(ds, gname); [print("  ", {k: round(v, 3) for k, v in row.items()}) for row in rows]
json.dump(out, open(f"{WORK}/louvain.json", "w"), indent=1)
