AI VIET NAM – Assignment 4 (Omega Rule toy implementation)


Description
-----------
This repository contains a small, self-contained PyTorch toy implementation to explore the
"Omega Rule" memory-updating idea (from ATLAS) at small scale. The code implements two
updaters that share the same input/output contract:


- OmegaUpdater: performs a few gradient-descent steps to optimize the external memory
so that M(k) ≈ v for keys/values in a local context window (matches Omega Rule).
- DeltaUpdater: a one-shot, analytical-like aggregated token-by-token update (baseline).


The code is intentionally small, readable, and easy to extend. It includes a synthetic
memorization dataset, evaluation helpers and scripts to run grids of experiments.


Repository structure
--------------------
```
project_root/
├─ experiments/run_grid.sh # runs a small grid and writes results/grid_results.csv
├─ results/ # experiment outputs
├─ src/
│ ├─ datasets/synthetic_dataset.py
│ ├─ models/omega_updater.py
│ ├─ models/delta_updater.py
│ ├─ eval.py
│ ├─ train.py
│ ├─ exp_runner.py
│ └─ utils.py
├─ README.md
├─ requirements.txt
```


Quickstart (CPU)
-----------------
1. Create & activate a virtual environment (recommended):


```bash
python3 -m venv .venv
source .venv/bin/activate
```


2. Install dependencies:


```bash
pip install -r requirements.txt
```


3. Run a single test (Omega updater):


```bash
# from project root
python3 -m src.train --updater omega --mem_size 64 --dim 64 --window 50 --steps 8 --lr 0.01 --seed 0
```


4. Run the baseline (Delta updater):


```bash
python3 -m src.train --updater delta --mem_size 64 --dim 64 --window 50 --lr 0.01 --seed 0
```


Grid experiments
----------------
Run the small grid (writes results/grid_results.csv):


message the author.