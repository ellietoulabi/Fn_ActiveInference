#!/bin/bash
#SBATCH --account=def-jrwright
#SBATCH --job-name=ma_mappo
#SBATCH --array=0-29                  # 30 seeds (0..29), one per array task -- same seed pool as
                                       # two_aif_fully_collective.sh / _independent.sh /
                                       # _individually_collective.sh (identical generate_random_config,
                                       # so seed N gives the same map here as in those three).
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0-12:00
#SBATCH --output=ma_mappo_%A_%a.out

# Genuine MAPPO (centralized critic, decentralized actor -- see
# run_two_ppo_agents.py::CentralizedCriticPPOTorchRLModule) on TwoAgentRedBlueButton.
# Trains a fresh policy for this seed (--mode pretrained: domain-randomized maps,
# generous budget), then evaluates on the same 100-episode/20-per-config schedule
# used by the AIF paradigms (two_aif_agents_{independent,individually_collective,
# fully_collective}.sh) -- keep these three numbers in sync with those scripts;
# a mismatch here silently breaks the apples-to-apples comparison.
#
# Override episode length / training budget at submit time, e.g.:
#   MAX_STEPS=30 sbatch cc_scripts/redbluebutton/mappo.sh
#   TRAIN_STEPS_BUDGET=2000 sbatch cc_scripts/redbluebutton/mappo.sh   # quick smoke test;
#     empirically undertrained (~15-30% success, see ai/02-debug.md) -- do not use for real
#     numbers. Default (unset) uses --mode's own budget formula, ~100k steps at --episodes 100,
#     which empirically converges to 90-100% success on both seeds tested.
set -uo pipefail                      # no -e: we still copy logs on failure

MAX_STEPS=${MAX_STEPS:-50}
MODE=${MODE:-pretrained}
TRAIN_STEPS_BUDGET=${TRAIN_STEPS_BUDGET:-}

module purge
module load python/3.11.4 scipy-stack
# Ray/RLlib depends on pyarrow. On Alliance, `pip install pyarrow` hits a dummy wheel that
# always fails; the real package comes from the Arrow module. Load it BEFORE creating /
# activating the venv (see https://docs.alliancecan.ca/wiki/Arrow).
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
# Exclude opencv-python for Compute Canada; not needed for RedBlueButton + PPO.
grep -v 'opencv-python' requirements.txt > requirements_cc.txt
# Keep pip build/cache off $HOME (quota) and away from tiny /tmp.
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
# Do NOT use pip's ray[rllib] extra on Alliance: it pulls pyarrow, and Alliance's
# wheelhouse only has a dummy pyarrow that always fails. Real pyarrow comes from
# 'module load gcc arrow' above. Install ray + the other RLlib deps explicitly.
RAY_VER="2.55.1"  # matches the version this code was actually developed/tested against; must be one of Alliance's wheelhouse-available versions, not an arbitrary pip version (see error: "Could not find a version that satisfies the requirement ray==2.40.0")
if ! pip install --no-input --prefer-binary "ray==${RAY_VER}"; then
    echo "ERROR: pip install ray==${RAY_VER} failed."
    exit 1
fi
# pydantic: ray 2.55.1's ray.train.v2 imports it unconditionally but it isn't
# pulled in as a declared dependency on this wheelhouse -- confirmed via a
# real CC job failure (ModuleNotFoundError: No module named 'pydantic',
# surfacing through rllib -> tune -> train -> pydantic). See
# cc_scripts/overcooked/mappo_semantic_action_level.sh for the same fix.
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
echo "ray/rllib/torch/pyarrow import check OK."

SEED_IDX=${SLURM_ARRAY_TASK_ID}
echo "---- Starting seed index ${SEED_IDX} (mode=${MODE} max_steps=${MAX_STEPS}) ----"

export PYTHONHASHSEED=0

# Budget suffix keeps a smoke-test submission (TRAIN_STEPS_BUDGET set) from ever landing
# in the same folder as a real production run (TRAIN_STEPS_BUDGET unset) -- same
# don't-silently-collide principle as the step${MAX_STEPS} tag.
BUDGET_SUFFIX=""
if [ -n "${TRAIN_STEPS_BUDGET}" ]; then
    BUDGET_SUFFIX="_budget${TRAIN_STEPS_BUDGET}"
fi
# MODE included so `pretrained` and `online` submissions at the same MAX_STEPS
# can never collide/overwrite each other's copied-back logs -- same
# don't-silently-collide principle as BUDGET_SUFFIX above (found 2026-08-24:
# this was previously missing, and both modes would have written to the same
# destination folder).
DEST_BASE=${DEST_BASE_OVERRIDE:-"/home/toulabin/projects/def-jrwright/toulabin/logs/ma_mappo_redbluebutton_${MODE}_step${MAX_STEPS}${BUDGET_SUFFIX}_30seed"}
mkdir -p "${DEST_BASE}"
LOG_FILE="$SLURM_TMPDIR/ma_mappo_seed${SEED_IDX}.log"

EXTRA_ARGS=()
if [ -n "${TRAIN_STEPS_BUDGET}" ]; then
    EXTRA_ARGS+=(--train-steps-budget "${TRAIN_STEPS_BUDGET}")
fi

python -u run_scripts_red_blue_doors/multi_agent/run_two_ppo_agents.py \
  --seed ${SEED_IDX} \
  --episodes 100 \
  --episodes-per-config 20 \
  --max-steps ${MAX_STEPS} \
  --mode ${MODE} \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
  --stats-output "logs/ma_mappo_seed${SEED_IDX}_stats.json" > "$LOG_FILE" 2>&1

EXIT_CODE=$?

echo "Copying logs..."
cp "$LOG_FILE" "${DEST_BASE}/" 2>/dev/null || echo "Warning: log file not found"
cp logs/two_ppo_agents_*_seed${SEED_IDX}_ep*_*.csv "${DEST_BASE}/" 2>/dev/null || echo "Warning: CSV log file not found"
cp "logs/ma_mappo_seed${SEED_IDX}_stats.json" "${DEST_BASE}/" 2>/dev/null || echo "Warning: stats JSON file not found"
# Configs JSON (2026-08-24): exact EVAL button-position sequence for this seed, so an
# additional baseline can be run later against the identical maps -- would
# otherwise be lost with the rest of SLURM_TMPDIR once this job ends.
cp logs/two_ppo_agents_*_seed${SEED_IDX}_ep*_*_configs.json "${DEST_BASE}/" 2>/dev/null || echo "Warning: configs JSON file not found"
echo "Copy done"

if [ $EXIT_CODE -ne 0 ]; then
    echo "run_two_ppo_agents.py failed for seed index $SEED_IDX with exit code $EXIT_CODE"
    tail -50 "$LOG_FILE"
    exit $EXIT_CODE
fi

echo "---- MAPPO seed index ${SEED_IDX} complete ----"
