"""Shared driver for the deterministic comparison; `impl` hooks provide framework specifics."""
import io, os, re, shutil, contextlib, numpy as np
from paths import DATA

def run(impl, DescModel, dataset, weights_dir, out, tol=0.005, tag="", maxiter=1000):
    x = np.load(f"{DATA}/{dataset}.npy")
    dims = [x.shape[1], 64, 32] if dataset == "paul" else [x.shape[1], 64, 16]
    work = os.path.join(out, f"{impl['name']}_{dataset}{tag}_work"); shutil.rmtree(work, ignore_errors=True); os.makedirs(work)
    shutil.copy(os.path.join(weights_dir, "ae_weights.h5"), work)       # TF-pretrained autoencoder from desc 2.1.1
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        d = DescModel(dims=dims, x=x, louvain_resolution=0.8, n_neighbors=10, batch_size=256, epochs_fit=0.4, tol=tol,
                      use_ae_weights=True, save_dir=work, random_seed=201809)
        r = {"features": d.extract_features(x), "init_pred": np.asarray(d.init_pred), "init_centroid": d.init_centroid[0],
             "q0": d.model.predict(x)}
        p0 = d.target_distribution(r["q0"]).astype(np.float32)
        snapshot = impl["snapshot"](d)
        impl["compile"](d)
        impl["step"](d, x[:256], p0[:256])                                  # one SGD step on the first batch
        r["q1"] = d.model.predict(x)
        impl["restore"](d, snapshot)
        impl["compile"](d)                                                 # fresh optimizer, as train_single does
        r["z"], r["q"] = d.fit(maxiter=maxiter)
    text = log.getvalue()
    open(os.path.join(out, f"{impl['name']}_{dataset}{tag}_log.txt"), "w").write(text)
    r["delta"] = np.array([float(v) for v in re.findall(r"delta_label of current \d+ th iteration is ([0-9.e-]+)", text)])
    final = re.search(r"delta_label\s+([0-9.e-]+)\s+< tol", text)
    r["delta_final"] = float(final.group(1)) if final else np.nan
    np.savez(os.path.join(out, f"{impl['name']}_{dataset}{tag}.npz"), **r)
    print(impl["name"], dataset + tag, "clusters", r["init_centroid"].shape[0], "updates", len(r["delta"]), "final delta", r["delta_final"])
