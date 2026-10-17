from paths import DET
import sys, json, numpy as np
from sklearn.metrics import adjusted_rand_score as ari
out = {}
for ds in sys.argv[1:]:
    t, p = np.load(f"{DET}/tf_{ds}.npz"), np.load(f"{DET}/torch_{ds}.npz")  # ds may carry a _long suffix
    md = lambda k: float(np.abs(t[k] - p[k]).max())
    rel = lambda k: float(np.abs(t[k] - p[k]).max() / np.abs(t[k]).max())
    lt, lp = t["q"].argmax(1), p["q"].argmax(1)
    out[ds] = {"n_cells": int(t["features"].shape[0]), "n_clusters": int(t["init_centroid"].shape[0]),
        "encoder_max_abs_diff": md("features"), "encoder_max_rel_diff": rel("features"),
        "centroid_max_abs_diff": md("init_centroid"), "q0_max_abs_diff": md("q0"), "q_after_1_step_max_abs_diff": md("q1"),
        "updates_tf": int(len(t["delta"])), "updates_torch": int(len(p["delta"])),
        "delta_trace_tf": t["delta"].tolist() + [float(t["delta_final"])], "delta_trace_torch": p["delta"].tolist() + [float(p["delta_final"])],
        "final_q_max_abs_diff": md("q"), "final_z_max_abs_diff": md("z"),
        "final_labels_identical_frac": float((lt == lp).mean()), "final_labels_ari": float(ari(lt, lp))}
print(json.dumps(out, indent=1))
json.dump(out, open(f"{DET}/compare_det.json", "w"), indent=1)
