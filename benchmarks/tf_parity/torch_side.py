import copy, sys
from paths import RUNS, DET
import numpy as np, torch
import desc.models.network as net
from desc.models._nn import train_on_batch
from common_side import run
dataset = sys.argv[1]
long = len(sys.argv) > 2 and sys.argv[2] == "long"   # tol=-1: never stop early, all 1000 iterations
tf = np.load(f"{DET}/tf_{dataset}.npz")
net.louvain = lambda adjacency, resolution=1.0, random_state=0: tf["init_pred"]   # same initial partition as TF
impl = {"name": "torch",
        "snapshot": lambda d: copy.deepcopy(d.model.state_dict()),
        "restore": lambda d, s: d.model.load_state_dict(s),
        "compile": lambda d: d.compile(optimizer=net.SGD(0.01, 0.9), loss="kld"),
        "step": lambda d, xb, pb: train_on_batch(d.model, d.optimizer, d.loss, torch.as_tensor(xb), torch.as_tensor(pb))}
run(impl, net.DescModel, dataset, f"{RUNS}/tf_{dataset}_201809/save", DET, tol=-1.0 if long else 0.005, tag="_long" if long else "")
