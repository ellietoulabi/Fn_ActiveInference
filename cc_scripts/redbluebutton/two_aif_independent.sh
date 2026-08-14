#!/bin/bash
#SBATCH --account=aip-jrwright
#SBATCH --job-name=ma_redblue_aif_ind
#SBATCH --array=0-29                   # 30 seeds (0..29), one per array task
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=0-5:00
#SBATCH --output=ma_ind_%A_%a.out

# This script runs the Independent paradigm (SA model) for two AIF agents.
# It mirrors the pattern used in cc_scripts/redbluebutton/eight.sh.

set -uo pipefail                      # no -e: we still copy logs on failure

# Override episode length at submit time, e.g.:
#   MAX_STEPS=30 sbatch cc_scripts/redbluebutton/two_aif_independent.sh
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
# RedBlueButton two-agent AIF runs do not need OpenCV.
grep -v 'opencv-python' requirements.txt > requirements_cc.txt
pip install --no-input -r requirements_cc.txt
echo "Dependencies installed."

SEED_IDX=${SLURM_ARRAY_TASK_ID}
echo "---- Starting seed index ${SEED_IDX} ----"

# Reproducible runs: seed is passed via --seed; Python script uses it directly.
export PYTHONHASHSEED=0

DEST_BASE=${DEST_BASE_OVERRIDE:-"/home/toulabin/projects/aip-jrwright/toulabin/logs/ma_ind_30seed"}
mkdir -p "${DEST_BASE}"
LOG_FILE="$SLURM_TMPDIR/ma_ind_seed${SEED_IDX}.log"

# You can tweak episodes/configs below if desired
python -u run_scripts_red_blue_doors/multi_agent/run_two_aif_agents_independent.py \
  --seed ${SEED_IDX} \
  --episodes 200 \
  --episodes-per-config 25 \
  --max-steps ${MAX_STEPS} \
  --print-steps \
  --log-policy-beliefs \
  --policy-top-k 5 \
  --log-full-q-pi \
  --log-state-beliefs > "$LOG_FILE" 2>&1

EXIT_CODE=$?

echo "Copying logs..."
cp "$LOG_FILE" "${DEST_BASE}/" 2>/dev/null || echo "Warning: verbose log file not found"
cp logs/two_aif_agents_independent_seeds*_ep*_*.csv "${DEST_BASE}/" 2>/dev/null || echo "Warning: CSV log file not found"
cp logs/two_aif_agents_independent_seeds*_ep*_*_stats.json "${DEST_BASE}/" 2>/dev/null || echo "Warning: stats JSON file not found"
cp logs/two_aif_agents_independent_seeds*_ep*_*_policy_log.jsonl "${DEST_BASE}/" 2>/dev/null || echo "Warning: policy belief JSONL file not found"
echo "Copy done"

if [ $EXIT_CODE -ne 0 ]; then
    echo "run_two_aif_agents_independent.py failed for seed index $SEED_IDX with exit code $EXIT_CODE"
    tail -50 "$LOG_FILE"
    exit $EXIT_CODE
fi

echo "---- Independent paradigm seed index ${SEED_IDX} complete ----"

