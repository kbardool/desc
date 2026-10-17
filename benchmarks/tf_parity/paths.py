"""Locations shared by all parity scripts. DESC_PARITY_DIR holds inputs, runs and results (default ./parity_work);
DESC_TF_SRC is the directory containing the desc 2.1.1 package (default $DESC_PARITY_DIR/desc-2.1.1)."""
import os
WORK = os.path.abspath(os.environ.get("DESC_PARITY_DIR", "parity_work"))
DATA, RUNS, DET = (os.path.join(WORK, d) for d in ("data", "runs", "det"))
TF_SRC = os.path.abspath(os.environ.get("DESC_TF_SRC", os.path.join(WORK, "desc-2.1.1")))
for d in (DATA, RUNS, DET):
    os.makedirs(d, exist_ok=True)
