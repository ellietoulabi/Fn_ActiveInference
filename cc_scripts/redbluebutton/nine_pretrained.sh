#!/bin/bash
#SBATCH --account=aip-jrwright
#SBATCH --job-name=sa_redblue_nine_agents_pretrained
#SBATCH --array=0-29
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0-12:00

# Pretrained-fair variant of nine.sh: the 8 non-AIF agents are pretrained to
# convergence against domain-randomized configs before the scored relocation
# protocol runs (compare_nine_agents_pretrained.py), rather than starting cold
# (compare_nine_agents.py, launched via nine.sh). Kept as a fully separate
# script/job-name/DEST_BASE from nine.sh, on purpose, so the two protocols can
# be submitted and run in parallel without any risk of their logs mixing.
#
# Override the protocol at submit time, e.g.:
#   EPISODES=400 EPISODES_PER_CONFIG=50 MAX_STEPS=30 sbatch cc_scripts/redbluebutton/nine_pretrained.sh
# DEST_BASE is auto-named from these three values (plus a literal "pretrained"
# segment, distinguishing it from nine.sh's DEST_BASE even at identical
# ep/cfg/step values) so different protocols and different scripts can never
# collide/overwrite each other's copied-back logs; override DEST_BASE_OVERRIDE
# directly only if you want a custom name.
set -euo pipefail

EPISODES=${EPISODES:-100}
EPISODES_PER_CONFIG=${EPISODES_PER_CONFIG:-20}
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
# compare_nine_agents_pretrained does not need OpenCV (RedBlueButton env only).
grep -v 'opencv-python' requirements.txt > requirements_cc.txt
pip install --no-input -r requirements_cc.txt
echo "Dependencies installed"

SEED_IDX=$SLURM_ARRAY_TASK_ID
echo "---- Starting seed index ${SEED_IDX} (PRETRAINED protocol, episodes=${EPISODES} episodes_per_config=${EPISODES_PER_CONFIG} max_steps=${MAX_STEPS}) ----"

# Reproducible runs: --seeds 1 --seed-idx-offset ${SEED_IDX} runs exactly one
# seed per array task (seed_idx = SEED_IDX, actual seed = BASE_SEED + SEED_IDX)
# -- this is the documented, intended parallelization pattern for this script
# (see its own --seed-idx-offset help text), matching nine.sh's one-seed-per-
# array-task convention. Writes into a local logs_pretrained/ subfolder
# (distinct from nine.sh's logs/) purely to keep the two protocols visually
# separate even before the copy-back step, though the filenames themselves
# already differ (this script's CSVs are named "..._pretrained_...").
# PYTHONHASHSEED=0 makes dict/set iteration order deterministic across runs.
export PYTHONHASHSEED=0
mkdir -p logs_pretrained
if ! python -u run_scripts_red_blue_doors/compare_agents/compare_nine_agents_pretrained.py \
    --seeds 1 --seed-idx-offset ${SEED_IDX} \
    --episodes ${EPISODES} --episodes-per-config ${EPISODES_PER_CONFIG} --max-steps ${MAX_STEPS} \
    --log-dir logs_pretrained; then
    EXIT_CODE=$?
    echo "compare_nine_agents_pretrained.py for seed index $SEED_IDX failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

DEST_BASE=${DEST_BASE_OVERRIDE:-"/home/toulabin/projects/aip-jrwright/toulabin/logs/sa_redbluebutton_pretrained_ep${EPISODES}_cfg${EPISODES_PER_CONFIG}_step${MAX_STEPS}_30seed"}
mkdir -p "${DEST_BASE}"

echo "Copying logs to home directory..."

cp logs_pretrained/nine_agents_comparison_pretrained_ep*_step*_seed${SEED_IDX}_*.csv "${DEST_BASE}/" 2>/dev/null || echo "Warning: CSV log file not found"
cp logs_pretrained/pretrain_stats_seed${SEED_IDX}_*.json "${DEST_BASE}/" 2>/dev/null || echo "Warning: pretrain_stats JSON file not found"
# Configs JSON (2026-08-24): the exact button-position sequence used for this seed's
# scored protocol, needed to run an additional agent later against the identical map
# sequence -- would otherwise be lost with the rest of SLURM_TMPDIR once this job ends.
cp logs_pretrained/nine_agents_configs_pretrained_ep*_step*_seed${SEED_IDX}_*.json "${DEST_BASE}/" 2>/dev/null || echo "Warning: configs JSON file not found"

echo "Copy done"
echo "---- Nine Agents (Pretrained) Seed Index ${SEED_IDX} complete ----"
