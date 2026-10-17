import sys
from paths import RUNS, DET, TF_SRC
sys.path.insert(0, TF_SRC)
import numpy as np
from desc.models.network import DescModel
from tensorflow.keras.optimizers import SGD
from common_side import run
impl = {"name": "tf",
        "snapshot": lambda d: d.model.get_weights(),
        "restore": lambda d, w: d.model.set_weights(w),
        "compile": lambda d: d.compile(optimizer=SGD(0.01, 0.9), loss="kld"),
        "step": lambda d, xb, pb: d.model.train_on_batch(xb, pb)}
dataset = sys.argv[1]
long = len(sys.argv) > 2 and sys.argv[2] == "long"   # tol=-1: never stop early, all 1000 iterations
run(impl, DescModel, dataset, f"{RUNS}/tf_{dataset}_201809/save", DET, tol=-1.0 if long else 0.005, tag="_long" if long else "")
