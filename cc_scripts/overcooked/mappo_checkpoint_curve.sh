#!/bin/bash
#SBATCH --account=aip-jrwright
#SBATCH --job-name=mappo_checkpoint_curve
#SBATCH --array=0-29                  # 30 seeds, one per array task -- EXACT same pool the AIF
                                       # paradigms (ind/ic/fc) and the existing MAPPO comparator
                                       # were run on: ep=76..105, a0=1000..1029, a1=2000..2029
                                       # (see _default_seed_lists in
                                       # run_scripts_overcooked/run_mappo_semantic_action_level_sweep.py).
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=0-03:00
#SBATCH --output=mappo_ckpt_curve_%A_%a.out

# New, standalone MAPPO sample-efficiency-curve launcher -- runs
# run_scripts_overcooked/run_mappo_checkpoint_curve.py (a new module; does
# NOT edit agents/PPO/MA_PPO/mappo_simple.py, run_mappo_semantic_action_level_sweep.py,
# or any Active Inference file -- see that module's own docstring and
# ai/02-debug.md's 2026-08-24 "MAPPO sample-efficiency curve" entry).
#
# One array task = one training seed = one EXACT (episode_seed, agent0_seed,
# agent1_seed) triple, identical to the ind/ic/fc/mappo launchers' own pool,
# so every checkpoint budget's result is directly, seed-for-seed paired
# against the already-computed 30-seed AIF numbers (thesis_logs/03_ma_overcooked/
# sal_{ind,ic,fc}_30seed*). Each task trains ONE MAPPO policy from scratch and
# evaluates the live policy at a ladder of step budgets (0 up to 500k, same
# ladder validated in the local 3-seed pilot -- see ai/02-debug.md), each
# checkpoint scored on exactly this task's own episode_seed (one 1500-step
# trajectory), matching the AIF paradigms' own one-trajectory-per-seed
# evaluation protocol exactly rather than averaging over extra eval seeds
# per checkpoint the way the local pilot did.
#
# Override at submit time, e.g.:
#   MAX_STEPS=1000 sbatch cc_scripts/overcooked/mappo_checkpoint_curve.sh
#   BUDGETS="0 50000 150000 300000 500000" sbatch cc_scripts/overcooked/mappo_checkpoint_curve.sh
set -uo pipefail                      # no -e: we still copy logs on failure

MAX_STEPS=${MAX_STEPS:-1500}
BUDGETS=${BUDGETS:-"0 25000 50000 75000 100000 150000 200000 225000 250000 275000 300000 350000 400000 500000"}
# Budget ladder embedded in DEST_BASE (not just the job name) so a future
# resubmission with a different BUDGETS/MAX_STEPS never silently overwrites
# this run's copied-back JSON -- same don't-silently-collide principle
# already applied to every other launcher in this project (see e.g.
# cc_scripts/overcooked/mappo_semantic_action_level.sh's MAX_TRAIN_STEPS tag).
BUDGET_TAG=$(echo "${BUDGETS}" | tr ' ' '_')
DEST_BASE=${DEST_BASE_OVERRIDE:-"/home/toulabin/projects/aip-jrwright/toulabin/logs/mappo_checkpoint_curve_step${MAX_STEPS}_budgets${BUDGET_TAG}_30seed"}

module purge
module load python/3.11.4 scipy-stack
# Ray/RLlib depends on pyarrow; see cc_scripts/overcooked/mappo_semantic_action_level.sh
# for the same fix (real pyarrow only comes from the Arrow module on Alliance).
module load gcc arrow

if [ "${SLURM_TMPDIR:-}" = "" ]; then
    echo "Error: SLURM_TMPDIR not defined"
    exit 1
fi
echo "Working in SLURM_TMPDIR: $SLURM_TMPDIR"
cd "$SLURM_TMPDIR"
mkdir -p project virtualenvs

echo "Cloning repository..."
cd project
git clone --quiet https://github.com/ellietoulabi/Fn_ActiveInference.git
echo "Repository cloned."

echo "Creating virtual environment (system-site-packages for scipy-stack)..."
cd ../virtualenvs
python3.11 -m venv --system-site-packages .venv
source .venv/bin/activate
echo "Activated virtualenv."

echo "Installing dependencies (cc_scripts/overcooked/requirements-cc-sal.txt, plus ray/RLlib/torch for MAPPO)..."
cd ../project/Fn_ActiveInference/
if ! pip install --no-input -r cc_scripts/overcooked/requirements-cc-sal.txt; then
    echo "ERROR: pip install failed. Do not use requirements.txt on Alliance (opencv-python dummy wheel)."
    exit 1
fi

echo "Checking pyarrow from Arrow module (must work before ray install)..."
python -c "import pyarrow; print('pyarrow OK', getattr(pyarrow, '__version__', '?'))" || {
    echo "ERROR: pyarrow not importable. Load 'gcc arrow' before activating the venv."
    exit 1
}

echo "Installing ray + RLlib deps (not in requirements-cc-sal.txt)..."
RAY_VER="2.55.1"
if ! pip install --no-input --prefer-binary "ray==${RAY_VER}"; then
    echo "ERROR: pip install ray==${RAY_VER} failed."
    exit 1
fi
if ! pip install --no-input --prefer-binary "torch" "dm-tree" "lz4" "tensorboardX" "pandas" "pydantic"; then
    echo "ERROR: pip install of RLlib dependencies failed."
    exit 1
fi
echo "ray + RLlib deps installed (pyarrow from Arrow module)."

python -c "import pyarrow; import ray; from ray.rllib.algorithms.ppo import PPOConfig; import torch; print('ray/rllib/torch/pyarrow import OK')" || {
    echo "ERROR: ray/rllib/torch/pyarrow import check failed after install."
    exit 1
}
echo "ray/rllib/torch/pyarrow import check OK."

# shellcheck source=overcooked/_sal_common.sh
source cc_scripts/overcooked/_sal_common.sh
sal_ensure_venv_runtime_deps || exit 1
sal_verify_imports || exit 1

# sal_ensure_venv_runtime_deps (above) force-reinstalls numpy/scipy directly
# into this venv's own site-packages via --ignore-installed -- confirmed by a
# real CC job failure that pandas (never itself force-installed, only ever
# resolved via --system-site-packages inheritance) becomes invisible to
# ray.tune's own `import pandas` right after that reinstall, even though the
# identical import succeeded moments earlier before it ran. Force pandas the
# same way, after the numpy/scipy reinstall, so it's guaranteed present
# regardless of whatever that workaround does to site-packages visibility.
echo "Re-ensuring pandas is visible after the numpy/scipy venv workaround..."
pip install --no-input --prefer-binary --ignore-installed pandas || {
    echo "ERROR: pip install --ignore-installed pandas failed."
    exit 1
}

echo "Preflight (mappo checkpoint curve): checking imports..."
sal_setup_pythonpath
python -c "
import numpy as np  # noqa: F401
import ray  # noqa: F401
import torch  # noqa: F401
from ray.rllib.algorithms.ppo import PPOConfig  # noqa: F401
from agents.PPO.MA_PPO.mappo_simple import (
    AIFObsOvercookedMAEnv, CentralizedCriticPPOTorchRLModule, build_config,
)  # noqa: F401
import run_scripts_overcooked.run_mappo_checkpoint_curve as _m  # noqa: F401
print('Preflight OK: mappo_checkpoint_curve')
" || {
    echo "ERROR: preflight imports failed for mappo_checkpoint_curve (see traceback above)."
    exit 1
}

SEED_IDX=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set (submit with sbatch --array)}
EP_SEED=$((76 + SEED_IDX))
A0_SEED=$((1000 + SEED_IDX))
A1_SEED=$((2000 + SEED_IDX))
echo "---- mappo_checkpoint_curve seed_idx=${SEED_IDX} ep=${EP_SEED} a0=${A0_SEED} a1=${A1_SEED} max_steps=${MAX_STEPS} budgets=[${BUDGETS}] ----"

mkdir -p "$DEST_BASE"
OUT_DIR="$SLURM_TMPDIR/mappo_checkpoint_curve_out"
mkdir -p "$OUT_DIR"
LOG_FILE="$SLURM_TMPDIR/mappo_ckpt_curve_ep${EP_SEED}_a0_${A0_SEED}_a1_${A1_SEED}.log"

# One exact (episode_seed, agent0_seed, agent1_seed) triple per array task,
# --train-seeds ${EP_SEED} so "seed N" means the same integer for the
# training-time RLlib seed as for the AIF-matched evaluation identity.
python -u run_scripts_overcooked/run_mappo_checkpoint_curve.py \
  --train-seeds ${EP_SEED} \
  --budgets ${BUDGETS} \
  --eval-episode-seeds ${EP_SEED} \
  --agent0-seed ${A0_SEED} \
  --agent1-seed ${A1_SEED} \
  --eval-max-steps ${MAX_STEPS} \
  --num-workers 8 --envs-per-worker 4 \
  --out-dir "$OUT_DIR" > "$LOG_FILE" 2>&1
EXIT_CODE=$?

echo "Copying logs and per-seed curve JSON..."
cp "$LOG_FILE" "${DEST_BASE}/" 2>/dev/null || echo "Warning: log file not found"
cp "$OUT_DIR"/mappo_curve_trainseed*.json "${DEST_BASE}/" 2>/dev/null || echo "Warning: curve JSON not found"
# Per-task single-seed summary (n_train_seeds=1) -- redundant with the
# per-seed file above but renamed per-task so 30 tasks' summaries don't
# collide when copied into the same DEST_BASE.
cp "$OUT_DIR/mappo_curve_summary.json" "${DEST_BASE}/mappo_curve_summary_seed${EP_SEED}.json" 2>/dev/null || echo "Warning: summary JSON not found"
echo "Copy done"

if [ $EXIT_CODE -ne 0 ]; then
    echo "mappo_checkpoint_curve failed (seed_idx=${SEED_IDX}) exit=${EXIT_CODE}"
    tail -80 "$LOG_FILE"
    exit $EXIT_CODE
fi
echo "---- mappo_checkpoint_curve seed_idx=${SEED_IDX} complete ----"
