#!/bin/bash
#SBATCH --account=aip-jrwright
#SBATCH --job-name=mappo_sal_centralized_critic
#SBATCH --array=0-29                  # 30 seeds (one episode per task: train + eval) -- matches the ind/ic/fc launchers' fixed pool: ep=76..105, a0=1000..1029, a1=2000..2029
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=4-00:00
#SBATCH --output=mappo_sal_%A_%a.out

# MAPPO paradigm, semantic-action level, one seed per array task: trains a
# fresh policy for this seed (default 1M env steps), then evaluates it and
# writes the per-step CSV, matching the ind/ic/fc launchers' one-seed-per-task
# convention. Uses the centralized-critic RLModule (see ai/02-debug.md section
# L, and agents/PPO/MA_PPO/mappo_simple.py::CentralizedCriticPPOTorchRLModule)
# -- the actor still only ever sees its own local observation; only the value
# function used during training gets the joint/global state (standard CTDE).
#
# Override episode length / training budget at submit time, e.g.:
#   MAX_TRAIN_STEPS=200000 MAX_STEPS=500 sbatch cc_scripts/overcooked/mappo_semantic_action_level.sh
set -uo pipefail                      # no -e: we still copy logs on failure

MAX_STEPS=${MAX_STEPS:-1500}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-1000000}
DEST_BASE=${DEST_BASE_OVERRIDE:-"/home/toulabin/projects/aip-jrwright/toulabin/logs/sal_mappo"}

module purge
module load python/3.11.4 scipy-stack
# Ray/RLlib depends on pyarrow. On Alliance, `pip install pyarrow` (including via the
# `ray[rllib]` extra, which pulls it in transitively) hits a dummy wheel that always
# fails; the real package comes from the Arrow module. Load it BEFORE creating the
# venv (see https://docs.alliancecan.ca/wiki/Arrow), same fix already applied to
# cc_scripts/redbluebutton/{mappo,three_plus_ppo}.sh.
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
# Do NOT use pip's ray[rllib] extra on Alliance: it pulls pyarrow, and Alliance's
# wheelhouse only has a dummy pyarrow that always fails (real pyarrow comes from
# 'module load gcc arrow' above). Install ray + the other RLlib deps explicitly,
# pinned to a version actually present in Alliance's wheelhouse -- ray==2.40.0
# previously failed here with "Could not find a version that satisfies the
# requirement ray==2.40.0"; 2.55.1 matches what this code was developed/tested
# against and is available in the wheelhouse.
RAY_VER="2.55.1"
if ! pip install --no-input --prefer-binary "ray==${RAY_VER}"; then
    echo "ERROR: pip install ray==${RAY_VER} failed."
    exit 1
fi
if ! pip install --no-input --prefer-binary "torch" "dm-tree" "lz4" "tensorboardX" "pandas"; then
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

# MAPPO-specific preflight, kept local to this script (not added to
# _sal_common.sh's shared sal_preflight(), which only covers ind/ic/fc) so
# this launcher never needs to touch any active-inference-related file.
echo "Preflight (mappo): checking imports..."
sal_setup_pythonpath
python -c "
import numpy as np  # noqa: F401
import ray  # noqa: F401
import torch  # noqa: F401
from ray.rllib.algorithms.ppo import PPOConfig  # noqa: F401
import sal_step_csv_log  # noqa: F401
from agents.PPO.MA_PPO.mappo_simple import (
    AIFObsOvercookedMAEnv, CentralizedCriticPPOTorchRLModule, build_config,
)  # noqa: F401
print('Preflight OK: mappo')
" || {
    echo "ERROR: preflight imports failed for mappo (see traceback above)."
    exit 1
}

SEED_IDX=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set (submit with sbatch --array)}
EP_SEED=$((76 + SEED_IDX))
A0_SEED=$((1000 + SEED_IDX))
A1_SEED=$((2000 + SEED_IDX))
echo "---- mappo seed_idx=${SEED_IDX} ep=${EP_SEED} a0=${A0_SEED} a1=${A1_SEED} max_steps=${MAX_STEPS} max_train_steps=${MAX_TRAIN_STEPS} ----"

mkdir -p "$DEST_BASE"
CSV_DIR="$SLURM_TMPDIR/logs_sal"
mkdir -p "$CSV_DIR"
CKPT_DIR="$SLURM_TMPDIR/checkpoints_sal"
LOG_FILE="$SLURM_TMPDIR/mappo_sal_ep${EP_SEED}_a0_${A0_SEED}_a1_${A1_SEED}.log"

# One seed per array task: --n-runs 1 --episode-start ${EP_SEED} derives
# agent0/agent1 seeds as 1000+SEED_IDX / 2000+SEED_IDX internally (see
# _default_seed_lists in the sweep script), matching every other paradigm's
# seed convention.
python -u run_scripts_overcooked/run_mappo_semantic_action_level_sweep.py \
  --n-runs 1 \
  --episode-start ${EP_SEED} \
  --max-steps ${MAX_STEPS} \
  --max-train-steps ${MAX_TRAIN_STEPS} \
  --checkpoint-dir "$CKPT_DIR" \
  --log-csv --log-dir "$CSV_DIR" \
  --num-workers 8 --envs-per-worker 4 > "$LOG_FILE" 2>&1
EXIT_CODE=$?

sal_copy_artifacts "$DEST_BASE" "$LOG_FILE" "$CSV_DIR"

if [ $EXIT_CODE -ne 0 ]; then
    echo "mappo sweep failed (seed_idx=${SEED_IDX}) exit=${EXIT_CODE}"
    sal_report_failure "$LOG_FILE"
    exit $EXIT_CODE
fi
echo "---- mappo seed_idx=${SEED_IDX} complete ----"
