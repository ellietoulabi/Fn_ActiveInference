#!/bin/bash
#SBATCH --account=def-jrwright
#SBATCH --job-name=ma_redblue_opsrl
#SBATCH --array=0-29                   # 30 seeds (0..29), one per array task -- same seed pool as
                                        # two_aif_independent.sh / _individually_collective.sh /
                                        # _fully_collective.sh / mappo.sh (identical generate_random_config,
                                        # so seed N gives the same map here as in those scripts).
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1-00:00
#SBATCH --output=ma_opsrl_%A_%a.out

# This script runs two independent OPSRL agents (agents/OPSRL/ma_agent.py::MAOPSRLAgent)
# on TwoAgentRedBlueButton -- the Bayesian model-based RL baseline extended from Stage 1
# (compare_nine_agents.py) to the two-agent setting, given the same ego-relative
# observation (own pos, partner pos, own on-red/on-blue, both button states) Independent
# AIF and MAPPO use. It mirrors the pattern used in two_aif_independent.sh / mappo.sh.
#
# Same 100-episode/20-per-config schedule used by the AIF paradigms and MAPPO -- keep
# this in sync with two_aif_independent.sh / two_aif_individually_collective.sh /
# two_aif_fully_collective.sh / mappo.sh; a mismatch here silently breaks the
# apples-to-apples comparison.

set -uo pipefail                      # no -e: we still copy logs on failure

# Override episode length at submit time, e.g.:
#   MAX_STEPS=30 sbatch cc_scripts/redbluebutton/two_opsrl.sh
MAX_STEPS=${MAX_STEPS:-50}

module purge
module load python/3.11.4 scipy-stack

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
# Exclude opencv-python: Compute Canada provides OpenCV via a module (pip package is a dummy that fails).
# OPSRL RedBlueButton runs need only numpy/gymnasium/tqdm -- no OpenCV, no ray/RLlib.
grep -v 'opencv-python' requirements.txt > requirements_cc.txt
pip install --no-input -r requirements_cc.txt
echo "Dependencies installed."

SEED_IDX=${SLURM_ARRAY_TASK_ID}
echo "---- Starting OPSRL seed index ${SEED_IDX} (max_steps=${MAX_STEPS}) ----"

# Reproducible runs: seed is passed via --seed; Python script uses it directly.
export PYTHONHASHSEED=0

# "opsrl" is baked into both the destination folder and every copied filename so
# results from this agent are never ambiguous with the AIF/MAPPO paradigms' logs
# sitting alongside them in the same parent logs directory.
DEST_BASE=${DEST_BASE_OVERRIDE:-"/home/toulabin/projects/def-jrwright/toulabin/logs/ma_opsrl_redbluebutton_step${MAX_STEPS}_30seed"}
mkdir -p "${DEST_BASE}"
LOG_FILE="$SLURM_TMPDIR/ma_opsrl_seed${SEED_IDX}.log"

python -u run_scripts_red_blue_doors/multi_agent/run_two_opsrl_agents.py \
  --seed ${SEED_IDX} \
  --episodes 100 \
  --episodes-per-config 20 \
  --max-steps ${MAX_STEPS} \
  --print-steps > "$LOG_FILE" 2>&1

EXIT_CODE=$?

echo "Copying logs..."
cp "$LOG_FILE" "${DEST_BASE}/" 2>/dev/null || echo "Warning: verbose log file not found"
cp logs/two_opsrl_agents_seed${SEED_IDX}_ep*_*.csv "${DEST_BASE}/" 2>/dev/null || echo "Warning: CSV log file not found"
cp logs/two_opsrl_agents_seed${SEED_IDX}_ep*_*_stats.json "${DEST_BASE}/" 2>/dev/null || echo "Warning: stats JSON file not found"
# Configs JSON (2026-08-24): exact button-position sequence for this seed, so an
# additional baseline can be run later against the identical maps -- would
# otherwise be lost with the rest of SLURM_TMPDIR once this job ends.
cp logs/two_opsrl_agents_seed${SEED_IDX}_ep*_*_configs.json "${DEST_BASE}/" 2>/dev/null || echo "Warning: configs JSON file not found"
echo "Copy done"

if [ $EXIT_CODE -ne 0 ]; then
    echo "run_two_opsrl_agents.py failed for seed index $SEED_IDX with exit code $EXIT_CODE"
    tail -50 "$LOG_FILE"
    exit $EXIT_CODE
fi

echo "---- OPSRL seed index ${SEED_IDX} complete ----"
