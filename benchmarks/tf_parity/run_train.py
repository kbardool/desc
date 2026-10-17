"""desc.train end to end. Usage: run_train.py <desc_src_dir> <dataset> <seed> <out_dir>. Runs under either implementation."""
import os, sys, time, numpy as np
src, dataset, seed, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
sys.path.insert(0, src)
import desc, scanpy as sc
from paths import DATA
x = np.load(f"{DATA}/{dataset}.npy")
dims = [x.shape[1], 64, 32] if dataset == "paul" else [x.shape[1], 64, 16]
os.makedirs(out, exist_ok=True)
t0 = time.time()
adata = desc.train(sc.AnnData(x), dims=dims, tol=0.005, n_neighbors=10, batch_size=256, louvain_resolution=[0.8, 1.0],
                   save_dir=os.path.join(out, "save"), use_GPU=False, num_Cores=1, use_ae_weights=False,
                   random_seed=seed, do_tsne=False, do_umap=False, verbose=True)
res = {"runtime": time.time() - t0, "version": desc.__version__}
for r in ("0.8", "1.0"):
    res["labels_" + r] = np.asarray(adata.obs["desc_" + r].astype(int))
    res["prob_" + r] = np.asarray(adata.uns["prob_matrix" + r])
    res["z_" + r] = np.asarray(adata.obsm["X_Embeded_z" + r])
np.savez(os.path.join(out, "result.npz"), **res)
print("DONE", desc.__version__, res["runtime"])
