#!/bin/bash
#SBATCH --account=def-jrwright
#SBATCH --job-name=aif_fullcoll
#SBATCH --array=0-4                   # seeds 0..4
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0-2:00

# Runs the FullyCollective paradigm (centralized joint model).

set -euo pipefail

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
pip install --no-input -r requirements.txt
echo "Dependencies installed."

SEED_IDX=${SLURM_ARRAY_TASK_ID}
echo "---- Starting seed index ${SEED_IDX} ----"

python -u run_scripts/multi_agent/run_two_aif_agents_fully_collective.py \
  --seed ${SEED_IDX} \
  --episodes 1000 \
  --episodes-per-config 100 \
  --max-steps 50 \
  --verbose \
  --episode-progress \
  --show-beliefs \
  --show-policies

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "run_two_aif_agents_fully_collective.py failed for seed index $SEED_IDX with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

DEST_BASE="/home/toulabin/projects/def-jrwright/toulabin/logs"
mkdir -p "${DEST_BASE}"

echo "Copying logs..."
cp logs/two_aif_agents_fully_collective_seeds*_ep*_*.csv "${DEST_BASE}/" 2>/dev/null || echo "Warning: CSV log file not found"

echo "Copy done"
echo "---- FullyCollective paradigm seed index ${SEED_IDX} complete ----"

