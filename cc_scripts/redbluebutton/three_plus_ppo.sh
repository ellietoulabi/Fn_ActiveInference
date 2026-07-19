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

echo "Installing ray + RLlib deps (needed for PPO; not in requirements.txt per its own comment)..."
# Do NOT use pip's ray[rllib] extra on Alliance: it pulls pyarrow, and Alliance's
# wheelhouse only has a dummy pyarrow that always fails. Real pyarrow comes from
# 'module load gcc arrow' above. Install ray + the other RLlib deps explicitly.
RAY_VER="2.40.0"
if ! pip install --no-input --prefer-binary "ray==${RAY_VER}"; then
    echo "ERROR: pip install ray==${RAY_VER} failed."
    exit 1
fi
if ! pip install --no-input --prefer-binary \
    "dm-tree" "lz4" "tensorboardX" "gymnasium" "pandas"; then
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

