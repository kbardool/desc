from paths import WORK, DATA, RUNS
import itertools, json, re, numpy as np
from sklearn.metrics import adjusted_rand_score as ari, normalized_mutual_info_score as nmi
SEEDS = (201809, 1, 2); R = RUNS
truth = {"paul_clusters": np.load(f"{DATA}/paul_clusters.npy", allow_pickle=True).astype(str), "paul_celltype": np.load(f"{DATA}/paul_celltype.npy", allow_pickle=True).astype(str)}
stat = lambda v: {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))}
out = {}
for ds in ("pbmc", "paul"):
    runs = {impl: [np.load(f"{R}/{impl}_{ds}_{s}/result.npz") for s in SEEDS] for impl in ("tf", "torch")}
    logs = {impl: [open(f"{R}/{impl}_{ds}_{s}/log.txt").read() for s in SEEDS] for impl in ("tf", "torch")}
    out[ds] = {"runtime_s": {i: [float(r["runtime"]) for r in runs[i]] for i in runs},
               "converged_runs": {i: [l.count("Reached tolerance threshold") for l in logs[i]] for i in logs},
               "init_clusters": {i: [[int(n) for n in re.findall(r"clustering \(ClusteringLayer\)\s+\(None, (\d+)\)", l)] for l in logs[i]] for i in logs}}
    for res in ("0.8", "1.0"):
        lab = {i: [r["labels_" + res] for r in runs[i]] for i in runs}
        e = {"n_clusters": {i: [int(len(np.unique(l))) for l in lab[i]] for i in lab},
             "ari_within_tf": stat([ari(a, b) for a, b in itertools.combinations(lab["tf"], 2)]),
             "ari_within_torch": stat([ari(a, b) for a, b in itertools.combinations(lab["torch"], 2)]),
             "ari_tf_vs_torch": stat([ari(a, b) for a in lab["tf"] for b in lab["torch"]]),
             "nmi_tf_vs_torch": stat([nmi(a, b) for a in lab["tf"] for b in lab["torch"]]),
             "max_prob_median": {i: [float(np.median(r["prob_" + res].max(1))) for r in runs[i]] for i in runs}}
        if ds == "paul":
            for t, y in truth.items():
                e[f"ari_vs_{t}"] = {i: [float(ari(y, l)) for l in lab[i]] for i in lab}
                e[f"nmi_vs_{t}"] = {i: [float(nmi(y, l)) for l in lab[i]] for i in lab}
        out[ds]["res_" + res] = e
json.dump(out, open(f"{WORK}/e2e.json", "w"), indent=1)
print(json.dumps(out, indent=1))
