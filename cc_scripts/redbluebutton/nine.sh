#!/bin/bash
#SBATCH --account=aip-jrwright
#SBATCH --job-name=sa_redblue_nine_agents
#SBATCH --array=0-29
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0-6:00

# Override the protocol at submit time, e.g.:
#   EPISODES=400 EPISODES_PER_CONFIG=50 MAX_STEPS=30 sbatch cc_scripts/redbluebutton/nine.sh
# DEST_BASE is auto-named from these three values so different protocols
# submitted this way can never collide/overwrite each other's copied-back
# logs; override DEST_BASE_OVERRIDE directly only if you want a custom name.
set -euo pipefail

EPISODES=${EPISODES:-200}
EPISODES_PER_CONFIG=${EPISODES_PER_CONFIG:-25}
MAX_STEPS=${MAX_STEPS:-50}

module purge
module load python/3.11.4  scipy-stack

if [ "${SLURM_TMPDIR:-}" == "" ]; then
    echo "Error: SLURM_TMPDIR not defined"
    exit 1
fi

cd $SLURM_TMPDIR
echo "Working in SLURM_TMPDIR: $SLURM_TMPDIR"

mkdir -p project virtualenvs
echo "Created project and virtualenvs directories"

echo "Cloning repository..."
cd project
git clone --quiet https://github.com/ellietoulabi/Fn_ActiveInference.git
echo "Repository cloned."

echo "Creating virtual environment..."
cd ../virtualenvs
python3.11 -m venv .venv
echo "Virtual environment created"
source .venv/bin/activate
echo "Activated virtualenv."

echo "Installing dependencies..."
cd ../project/Fn_ActiveInference/
# Exclude opencv-python: Compute Canada provides OpenCV via a module (pip package is a dummy that fails).
# compare_nine_agents does not need OpenCV (RedBlueButton env only).
grep -v 'opencv-python' requirements.txt > requirements_cc.txt
pip install --no-input -r requirements_cc.txt
echo "Dependencies installed"

SEED_IDX=$SLURM_ARRAY_TASK_ID
echo "---- Starting seed index ${SEED_IDX} (episodes=${EPISODES} episodes_per_config=${EPISODES_PER_CONFIG} max_steps=${MAX_STEPS}) ----"

# Reproducible runs: seed is passed via --seed_idx; Python script uses BASE_SEED + seed_idx.
# PYTHONHASHSEED=0 makes dict/set iteration order deterministic across runs.
export PYTHONHASHSEED=0
if ! python -u run_scripts_red_blue_doors/compare_agents/compare_nine_agents.py --seed_idx ${SEED_IDX} --log-aif-beliefs \
    --episodes ${EPISODES} --episodes-per-config ${EPISODES_PER_CONFIG} --max-steps ${MAX_STEPS}; then
    EXIT_CODE=$?
    echo "compare_nine_agents.py for seed index $SEED_IDX failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

DEST_BASE=${DEST_BASE_OVERRIDE:-"/home/toulabin/projects/aip-jrwright/toulabin/logs/sa_redbluebutton_ep${EPISODES}_cfg${EPISODES_PER_CONFIG}_step${MAX_STEPS}_30seed"}
mkdir -p "${DEST_BASE}"

echo "Copying logs and Q-tables to home directory..."

cp logs/nine_agents_comparison_ep*_step*_seed${SEED_IDX}_*.csv "${DEST_BASE}/" 2>/dev/null || echo "Warning: CSV log file not found"
cp logs/nine_agents_qtable_ep*_step*_seed${SEED_IDX}_*.json "${DEST_BASE}/" 2>/dev/null || echo "Warning: Some Q-table files not found"
cp logs/nine_agents_aif_beliefs_ep*_step*_seed${SEED_IDX}_*.jsonl "${DEST_BASE}/" 2>/dev/null || echo "Warning: AIF belief JSONL file not found"
# Configs JSON (2026-08-24): the exact button-position sequence used for this seed,
# needed to run an additional agent later against the identical map sequence -- would
# otherwise be lost with the rest of SLURM_TMPDIR once this job ends.
cp logs/nine_agents_configs_ep*_step*_seed${SEED_IDX}_*.json "${DEST_BASE}/" 2>/dev/null || echo "Warning: configs JSON file not found"

echo "Copy done"
echo "---- Nine Agents Seed Index ${SEED_IDX} complete ----"
