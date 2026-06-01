#!/bin/bash
#SBATCH --account=def-jrwright
#SBATCH --job-name=aif_fc_sal
#SBATCH --array=0-4                   # one seed per array task
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=2-00:00
#SBATCH --output=fc_sal_%A_%a.out

# FullyCollective paradigm, semantic-action level, one seed per array task.
# One IC brain controls both agents; the brain plans over 400 joint primitive
# policies. Full per-step logs go to a per-seed .log file.
#
# Override episode length / precision at submit time, e.g.:
#   MAX_STEPS=500 sbatch cc_scripts/fc_semantic_action_level.sh
set -uo pipefail                      # no -e: we still copy logs on failure

MAX_STEPS=${MAX_STEPS:-2000}
GAMMA=${GAMMA:-4.0}
ALPHA=${ALPHA:-8.0}
DEST_BASE="/home/toulabin/projects/def-jrwright/toulabin/logs/sal_fc"

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

echo "Creating virtual environment (system-site-packages for scipy-stack)..."
cd ../virtualenvs
python3.11 -m venv --system-site-packages .venv
source .venv/bin/activate
echo "Activated virtualenv."

echo "Installing dependencies (cc_scripts/requirements-cc-sal.txt; not full requirements.txt)..."
cd ../project/Fn_ActiveInference/
if ! pip install --no-input -r cc_scripts/requirements-cc-sal.txt; then
    echo "ERROR: pip install failed. Do not use requirements.txt on Alliance (opencv-python dummy wheel)."
    exit 1
fi
echo "Dependencies installed."

python -c "import numpy; import gymnasium; import dill" || {
    echo "ERROR: post-install import check failed (numpy/gymnasium/dill)."
    exit 1
}

SEED_IDX=${SLURM_ARRAY_TASK_ID}
EP_SEED=$((76 + SEED_IDX))
AGENT_SEED=$((1000 + SEED_IDX))
echo "---- fc seed_idx=${SEED_IDX} ep=${EP_SEED} brain=${AGENT_SEED} max_steps=${MAX_STEPS} ----"

mkdir -p "$DEST_BASE"
LOG_FILE="$SLURM_TMPDIR/fc_sal_ep${EP_SEED}_brain${AGENT_SEED}.log"

export PYTHONPATH="$PWD:$PWD/environments/overcooked_ai/src"
python -u run_scripts_overcooked/run_fully_collective_semantic_action_level.py \
  --n-runs 1 \
  --episode-seeds ${EP_SEED} \
  --agent-seeds ${AGENT_SEED} \
  --gamma ${GAMMA} --alpha ${ALPHA} \
  --max-steps ${MAX_STEPS} \
  --log-steps > "$LOG_FILE" 2>&1
EXIT_CODE=$?

echo "Copying logs..."
cp "$LOG_FILE" "$DEST_BASE/" 2>/dev/null || echo "Warning: log file not found"
echo "Copy done"

if [ $EXIT_CODE -ne 0 ]; then
    echo "fc run failed (seed_idx=${SEED_IDX}) exit=${EXIT_CODE}"
    exit $EXIT_CODE
fi
echo "---- fc seed_idx=${SEED_IDX} complete ----"
