#!/bin/bash
#SBATCH --account=def-jrwright
#SBATCH --job-name=three_plus_ppo
#SBATCH --array=0-14              # seeds 0..14 as separate jobs (15 seeds)
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=0-13:59

# Compare three AIF pairings + two PPO conditions (pretrained, online-budget; see
# run_two_ppo_agents.py --mode) on RedBlueButton.
# Settings: 15 seeds, 200 episodes/seed, map change every 25 episodes, max 30 steps.

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

echo "Installing ray[rllib] (needed for PPO baseline; not in requirements.txt per its own comment)..."
if ! pip install --no-input "ray[rllib]>=2.0.0"; then
    echo "ERROR: pip install ray[rllib] failed."
    exit 1
fi
echo "ray[rllib] installed."

python -c "import ray; from ray.rllib.algorithms.ppo import PPOConfig; import torch; print('ray/rllib/torch import OK')" || {
    echo "ERROR: ray/rllib/torch import check failed after install."
    exit 1
}

echo "ray/rllib/torch import check OK."

SEED_IDX=${SLURM_ARRAY_TASK_ID}
OUT_DIR="logs/compare_three_pairings_plus_ppo_seed${SEED_IDX}"
echo "---- Starting compare_three_pairings_plus_ppo for seed index ${SEED_IDX} ----"

export PYTHONHASHSEED=0
python -u run_scripts_red_blue_doors/compare_agents/compare_three_pairings_plus_ppo.py \
  --seeds 1 \
  --seed "${SEED_IDX}" \
  --episodes 200 \
  --episodes-per-config 25 \
  --max-steps 30 \
  --output-dir "${OUT_DIR}"

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "compare_three_pairings_plus_ppo.py failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

DEST_BASE="${HOME}/projects/def-jrwright/toulabin/logs"
DEST="${DEST_BASE}/three_plus_ppo_seed${SEED_IDX}"
mkdir -p "${DEST}"

echo "Copying all logs for seed ${SEED_IDX} to ${DEST}..."

# Compare table + per-paradigm stats JSONs (seed-specific dir; no overwrite across array tasks)
cp -r "${OUT_DIR}" "${DEST}/" 2>/dev/null || echo "Warning: compare output dir not found"

# AIF step CSVs + stats (Ind / IC / FC)
cp logs/two_aif_agents_*_seeds*_ep*_*.csv "${DEST}/" 2>/dev/null || echo "Warning: AIF CSV logs not found"
cp logs/two_aif_agents_*_seeds*_ep*_*_stats.json "${DEST}/" 2>/dev/null || echo "Warning: AIF stats JSONs not found"

# PPO step CSVs + stats (pretrained + online)
cp logs/two_ppo_agents_*_seeds*_ep*_*.csv "${DEST}/" 2>/dev/null || echo "Warning: PPO CSV logs not found"
cp logs/two_ppo_agents_*_seeds*_ep*_*_stats.json "${DEST}/" 2>/dev/null || echo "Warning: PPO stats JSONs not found"

echo "Copy done -> ${DEST}"
echo "---- three_plus_ppo seed index ${SEED_IDX} complete ----"

