"""Write identical float32 inputs for both implementations (run in the desc 3 env)."""
import re, sys, numpy as np, scanpy as sc
repo = sys.argv[1]
from paths import DATA as out
sys.path.insert(0, repo)
import desc
pbmc = desc.datasets.pbmc_processed()
np.save(f"{out}/pbmc.npy", np.asarray(pbmc.X, dtype=np.float32))
# Paul15 exactly as in desc_2.1.1_paul.ipynb
sc.settings.datasetdir = f"{repo}/desc/original/data"
paul = sc.datasets.paul15()
clusters = paul.obs["paul15_clusters"].astype(str).values
celltype = np.array([re.split(r"[0-9]{1,2}", c, maxsplit=1)[1] for c in clusters])
sc.pp.log1p(paul)
sc.pp.highly_variable_genes(paul, n_top_genes=1000, subset=True)
sc.pp.scale(paul, max_value=6)
np.save(f"{out}/paul.npy", np.asarray(paul.X, dtype=np.float32))
np.save(f"{out}/paul_clusters.npy", clusters); np.save(f"{out}/paul_celltype.npy", celltype)
print("pbmc", pbmc.shape, "paul", paul.shape, "clusters", len(set(clusters)), "celltypes", sorted(set(celltype)))
