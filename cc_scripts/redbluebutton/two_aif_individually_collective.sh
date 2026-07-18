#!/bin/bash
#SBATCH --account=def-jrwright
#SBATCH --job-name=ma_redblue_aif_ic
#SBATCH --array=0-14                  # seeds 0..14
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=0-10:00

# Runs the IndividuallyCollective paradigm (two agents, joint model each).

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
# Exclude opencv-python: Compute Canada provides OpenCV via a module (pip package is a dummy that fails).
# RedBlueButton two-agent AIF runs do not need OpenCV.
grep -v 'opencv-python' requirements.txt > requirements_cc.txt
pip install --no-input -r requirements_cc.txt
echo "Dependencies installed."

SEED_IDX=${SLURM_ARRAY_TASK_ID}
echo "---- Starting seed index ${SEED_IDX} ----"

# Reproducible runs: seed is passed via --seed; Python script uses it directly.
export PYTHONHASHSEED=0

python -u run_scripts_red_blue_doors/multi_agent/run_two_aif_agents_individually_collective.py \
  --seed ${SEED_IDX} \
  --episodes 200 \
  --episodes-per-config 25 \
  --max-steps 30 \
  --verbose \
  --episode-progress \
  --show-beliefs \
  --show-policies

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "run_two_aif_agents_individually_collective.py failed for seed index $SEED_IDX with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

DEST_BASE="${HOME}/projects/def-jrwright/toulabin/logs"
mkdir -p "${DEST_BASE}"

echo "Copying logs..."
cp logs/two_aif_agents_individually_collective_seeds*_ep*_*.csv "${DEST_BASE}/" 2>/dev/null || echo "Warning: CSV log file not found"
cp logs/two_aif_agents_individually_collective_seeds*_ep*_*_stats.json "${DEST_BASE}/" 2>/dev/null || echo "Warning: stats JSON file not found"

echo "Copy done"
echo "---- IndividuallyCollective paradigm seed index ${SEED_IDX} complete ----"

