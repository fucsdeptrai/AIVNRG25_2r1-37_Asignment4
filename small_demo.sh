#!/usr/bin/env bash
set -euo pipefail

# small_demo.sh
# - tạo venv (nếu chưa có)
# - cài dependencies tối thiểu
# - đảm bảo __init__.py để Python nhận src như package
# - chạy 1 single run (Omega)
# - chạy exp_runner (small grid)
# - chạy analysis script để xuất plots
#
# Run from project root: ./small_demo.sh

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "Project root: $ROOT_DIR"
cd "$ROOT_DIR"

PYTHON_CMD=${PYTHON_CMD:-python3}
echo "Using python: $($PYTHON_CMD --version 2>&1)"

# 1) virtualenv
if [ ! -d ".venv" ]; then
  echo "Creating virtualenv .venv ..."
  $PYTHON_CMD -m venv .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate

pip install --upgrade pip
if [ -f "requirements.txt" ]; then
  echo "Installing requirements.txt ..."
  pip install -r requirements.txt || echo "pip install -r requirements.txt failed; please check file"
else
  echo "No requirements.txt found — installing minimal packages"
  pip install torch numpy matplotlib pandas || echo "Try installing torch manually if pip failed (CUDA vs CPU builds)"
fi

# 2) ensure package importability
echo "Ensuring src/ is a package (creating __init__ files if missing)..."
mkdir -p src models src/models src/datasets
touch src/__init__.py
touch src/models/__init__.py
touch src/datasets/__init__.py

# 3) ensure results + analysis folders exist
mkdir -p results
mkdir -p analysis

# 4) Single quick test (Omega)
echo "Running a quick single test (Omega) ..."
SINGLE_OUT="results/single_omega.txt"
set +e
$PYTHON_CMD -m src.train --updater Omega --mem_size 20 --dim 10 --window 5 --steps 200 --lr 0.01 --reg 0.0 --seed 0 > "$SINGLE_OUT" 2>&1
RET=$?
set -e
if [ $RET -ne 0 ]; then
  echo "Single run failed. See $SINGLE_OUT for output. Aborting."
  echo "Tail of output:"
  tail -n 40 "$SINGLE_OUT" || true
  exit 1
else
  echo "Single run succeeded. Output saved to $SINGLE_OUT"
  tail -n 10 "$SINGLE_OUT" || true
fi

# 5) Run small grid experiments via exp_runner
echo "Running small grid experiments via exp_runner..."
# Prefer module run if available, else run script
if python -c "import importlib, sys, pkgutil; print('ok')" >/dev/null 2>&1; then
  if python -c "import src.exp_runner" >/dev/null 2>&1; then
    echo "Running as module: python -m src.exp_runner --out_csv results/exp_results.csv"
    # if exp_runner module expects CLI args, run default
    python -m src.exp_runner --out_csv results/exp_results.csv
  else
    if [ -f "src/exp_runner.py" ]; then
      echo "Running script: python src/exp_runner.py --out_csv results/exp_results.csv"
      python src/exp_runner.py --out_csv results/exp_results.csv
    else
      echo "Could not find src/exp_runner.py or src.exp_runner. Please ensure exp_runner exists."
      exit 1
    fi
  fi
else
  echo "Python execution test failed. Aborting."
  exit 1
fi

echo "Exp runner finished. Results written to results/exp_results.csv"
ls -l results/exp_results.csv || true
head -n 8 results/exp_results.csv || true

# 6) Run analysis (plot)
echo "Running analysis script to produce plots..."
if [ -f "analysis/analysis.py" ]; then
  python analysis/analysis.py
  echo "Analysis finished. Check generated plots (mse_comparison.png etc.)"
  ls -l analysis/*.png || true
else
  echo "No analysis/analysis.py found — skip plotting step."
fi

echo "Demo run complete. Inspect results/ and analysis/ directories."
