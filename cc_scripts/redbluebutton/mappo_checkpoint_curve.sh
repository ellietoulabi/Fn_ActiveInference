#!/bin/bash
#SBATCH --account=aip-jrwright
#SBATCH --job-name=ma_mappo_checkpoint_curve
#SBATCH --array=0-29                  # 30 seeds (0..29), one per array task -- same seed pool as
                                       # nine.sh / two_aif_{independent,fully_collective,
                                       # individually_collective}.sh / mappo.sh / two_opsrl.sh
                                       # (SEED_IDX used directly as --seed, no offset).
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0-01:00
#SBATCH --output=ma_mappo_ckpt_curve_%A_%a.out

# New, standalone MAPPO sample-efficiency-curve launcher for MA Red-Blue-Button
# -- runs run_scripts_red_blue_doors/multi_agent/run_mappo_checkpoint_curve.py
# (a new module; does NOT edit run_two_ppo_agents.py or any Active Inference
# file -- see that module's own docstring and ai/02-debug.md's 2026-08-24
# "MAPPO sample-efficiency curve" entries).
#
# Unlike the Overcooked version of this same idea, this environment has NO
# teleportation -- every action is a real primitive step through the same
# TwoAgentRedBlueButtonEnv the AIF paradigms and OPSRL play against, so
# there's no environment-dynamics asymmetry to disclose for this comparison.
#
# One array task = one seed = one training run, using the exact same
# SEED_IDX-as-seed convention as every other MA Red-Blue-Button launcher.
# Each task trains ONE MAPPO policy from scratch and evaluates the live
# policy at a ladder of step budgets (0 up to 100k, the scale already
# empirically shown -- in this session's own 5-seed sanity check -- to span
# from near-random (~20-40% success at ~5k steps) to near-ceiling (~95-100%
# at ~100k steps) on this task), each checkpoint scored via the SAME
# 100-episode/20-per-config/max-steps scored protocol every AIF paradigm and
# OPSRL are evaluated on (run_seed_experiment, imported unmodified from
# run_two_ppo_agents.py).
#
# MODE mirrors run_two_ppo_agents.py's own --mode exactly: "pretrained"
# (default) trains on domain-randomized maps every episode; "online" trains
# on the exact same map schedule evaluation uses for this seed. Submit BOTH
# to get both curves -- they write to different DEST_BASE folders (MODE is
# embedded below) so they can never collide:
#   sbatch cc_scripts/redbluebutton/mappo_checkpoint_curve.sh                  # MODE=pretrained (default)
#   MODE=online sbatch cc_scripts/redbluebutton/mappo_checkpoint_curve.sh
#
# Other overrides at submit time, e.g.:
#   MAX_STEPS=30 sbatch cc_scripts/redbluebutton/mappo_checkpoint_curve.sh
#   BUDGETS="0 5000 25000 50000 100000" sbatch cc_scripts/redbluebutton/mappo_checkpoint_curve.sh
set -uo pipefail                      # no -e: we still copy logs on failure

MODE=${MODE:-pretrained}
MAX_STEPS=${MAX_STEPS:-50}
BUDGETS=${BUDGETS:-"0 2000 4000 6000 10000 15000 20000 30000 40000 50000 75000 100000"}
EVAL_EPISODES=${EVAL_EPISODES:-100}
EVAL_EPISODES_PER_CONFIG=${EVAL_EPISODES_PER_CONFIG:-20}
# Budget ladder + max_steps + MODE embedded in DEST_BASE so a future
# resubmission with a different BUDGETS/MAX_STEPS/MODE never silently
# overwrites this run's copied-back JSON -- same don't-silently-collide
# principle already applied to every other launcher in this project (see
# e.g. mappo.sh's own ${MODE} tag).
BUDGET_TAG=$(echo "${BUDGETS}" | tr ' ' '_')
DEST_BASE=${DEST_BASE_OVERRIDE:-"/home/toulabin/projects/aip-jrwright/toulabin/logs/ma_mappo_checkpoint_curve_${MODE}_step${MAX_STEPS}_budgets${BUDGET_TAG}_30seed"}

module purge
module load python/3.11.4 scipy-stack
# Ray/RLlib depends on pyarrow. On Alliance, `pip install pyarrow` hits a dummy wheel that
# always fails; the real package comes from the Arrow module. Load it BEFORE creating /
# activating the venv (see https://docs.alliancecan.ca/wiki/Arrow, and mappo.sh's identical
# fix -- this launcher was missing it entirely until caught by a real CC job failure).
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

echo "Creating virtual environment..."
cd ../virtualenvs
python3.11 -m venv .venv
source .venv/bin/activate
echo "Activated virtualenv."

echo "Installing dependencies..."
cd ../project/Fn_ActiveInference/
grep -v 'opencv-python' requirements.txt > requirements_cc.txt
export TMPDIR="${SLURM_TMPDIR}/pip_tmp"
export PIP_CACHE_DIR="${SLURM_TMPDIR}/pip_cache"
mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}"
pip install --no-input --upgrade pip setuptools wheel
pip install --no-input -r requirements_cc.txt
echo "Dependencies installed."

echo "Checking pyarrow from Arrow module (must work before ray install)..."
python -c "import pyarrow; print('pyarrow OK', getattr(pyarrow, '__version__', '?'))" || {
    echo "ERROR: pyarrow not importable. Load 'gcc arrow' before activating the venv."
    exit 1
}

echo "Installing ray + RLlib deps (not in requirements.txt per its own comment)..."
RAY_VER="2.55.1"
if ! pip install --no-input --prefer-binary "ray==${RAY_VER}"; then
    echo "ERROR: pip install ray==${RAY_VER} failed."
    exit 1
fi
if ! pip install --no-input --prefer-binary \
    "dm-tree" "lz4" "tensorboardX" "gymnasium" "pandas" "pydantic"; then
    echo "ERROR: pip install of RLlib dependencies failed."
    exit 1
fi
echo "ray + RLlib deps installed (pyarrow from Arrow module)."

python -c "import pyarrow; import ray; from ray.rllib.algorithms.ppo import PPOConfig; import torch; print('ray/rllib/torch/pyarrow import OK')" || {
    echo "ERROR: ray/rllib/torch/pyarrow import check failed after install."
    exit 1
}
echo "ray/rllib/torch import check OK."

echo "Preflight (mappo_checkpoint_curve): checking imports..."
python -c "
import numpy as np  # noqa: F401
import pyarrow  # noqa: F401
import ray  # noqa: F401
import torch  # noqa: F401
from ray.rllib.algorithms.ppo import PPOConfig  # noqa: F401
from run_scripts_red_blue_doors.multi_agent.run_two_ppo_agents import (
    CentralizedCriticPPOTorchRLModule, RedBlueButtonPPOWrapper, run_seed_experiment,
)  # noqa: F401
import run_scripts_red_blue_doors.multi_agent.run_mappo_checkpoint_curve as _m  # noqa: F401
print('Preflight OK: mappo_checkpoint_curve (redbluebutton)')
" || {
    echo "ERROR: preflight imports failed (see traceback above)."
    exit 1
}

export PYTHONHASHSEED=0

SEED_IDX=${SLURM_ARRAY_TASK_ID}
echo "---- mappo_checkpoint_curve seed_idx=${SEED_IDX} mode=${MODE} max_steps=${MAX_STEPS} budgets=[${BUDGETS}] eval_episodes=${EVAL_EPISODES}/${EVAL_EPISODES_PER_CONFIG} ----"

mkdir -p "$DEST_BASE"
OUT_DIR="$SLURM_TMPDIR/mappo_checkpoint_curve_out"
mkdir -p "$OUT_DIR"
LOG_FILE="$SLURM_TMPDIR/ma_mappo_ckpt_curve_seed${SEED_IDX}_${MODE}.log"

# One seed per array task, --train-seeds ${SEED_IDX} matching every other
# MA Red-Blue-Button launcher's SEED_IDX-as-seed convention exactly.
python -u run_scripts_red_blue_doors/multi_agent/run_mappo_checkpoint_curve.py \
  --mode ${MODE} \
  --train-seeds ${SEED_IDX} \
  --budgets ${BUDGETS} \
  --eval-episodes ${EVAL_EPISODES} \
  --eval-episodes-per-config ${EVAL_EPISODES_PER_CONFIG} \
  --max-steps ${MAX_STEPS} \
  --num-workers 2 --envs-per-worker 4 \
  --out-dir "$OUT_DIR" > "$LOG_FILE" 2>&1
EXIT_CODE=$?

echo "Copying logs and per-seed curve JSON..."
cp "$LOG_FILE" "${DEST_BASE}/" 2>/dev/null || echo "Warning: log file not found"
cp "$OUT_DIR"/mappo_curve_trainseed*.json "${DEST_BASE}/" 2>/dev/null || echo "Warning: curve JSON not found"
cp "$OUT_DIR/mappo_curve_summary.json" "${DEST_BASE}/mappo_curve_summary_seed${SEED_IDX}_${MODE}.json" 2>/dev/null || echo "Warning: summary JSON not found"
echo "Copy done"

if [ $EXIT_CODE -ne 0 ]; then
    echo "mappo_checkpoint_curve failed (seed_idx=${SEED_IDX}) exit=${EXIT_CODE}"
    tail -80 "$LOG_FILE"
    exit $EXIT_CODE
fi
echo "---- mappo_checkpoint_curve seed_idx=${SEED_IDX} complete ----"
