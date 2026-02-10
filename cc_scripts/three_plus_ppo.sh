#!/bin/bash
#SBATCH --account=def-jrwright
#SBATCH --job-name=three_plus_ppo
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0-8:59

# Compare three AIF pairings + PPO on RedBlueButton
# Settings: 5 seeds, 200 episodes/seed, map change every 25 episodes, max 30 steps.

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
# Exclude opencv-python for Compute Canada; not needed for RedBlueButton + PPO.
grep -v 'opencv-python' requirements.txt > requirements_cc.txt
pip install --no-input -r requirements_cc.txt
echo "Dependencies installed."

echo "---- Starting compare_three_pairings_plus_ppo ----"

export PYTHONHASHSEED=0
python -u run_scripts_red_blue_doors/compare_agents/compare_three_pairings_plus_ppo.py \
  --seeds 5 \
  --episodes 200 \
  --episodes-per-config 25 \
  --max-steps 30

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "compare_three_pairings_plus_ppo.py failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

DEST_BASE="${HOME}/projects/def-jrwright/toulabin/logs"
mkdir -p "${DEST_BASE}"

echo "Copying comparison logs to home directory..."
cp logs/compare_three_pairings_plus_ppo_*/comparison.csv "${DEST_BASE}/" 2>/dev/null || echo "Warning: comparison CSV not found"
cp logs/compare_three_pairings_plus_ppo_*/comparison_summary.json "${DEST_BASE}/" 2>/dev/null || echo "Warning: comparison summary JSON not found"

echo "Copy done"
echo "---- three_plus_ppo job complete ----"

