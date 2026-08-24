#!/bin/bash
#SBATCH --account=aip-jrwright
#SBATCH --job-name=ma_redblue_opsrl_pretrained
#SBATCH --array=0-29                   # 30 seeds (0..29), one per array task -- same seed pool as
                                        # two_aif_independent.sh / _individually_collective.sh /
                                        # _fully_collective.sh / mappo.sh / two_opsrl.sh (identical
                                        # generate_random_config, so seed N gives the same map here
                                        # as in those scripts' scored-evaluation phase).
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1-00:00
#SBATCH --output=ma_opsrl_pretrained_%A_%a.out

# Pretrain-to-convergence variant of the OPSRL baseline
# (agents/OPSRL/ma_agent_pretrained_convergence.py::MAOPSRLAgentPretrainedConvergence,
# a trivial subclass of the same MAOPSRLAgent two_opsrl.sh already runs cold-start --
# see run_two_opsrl_agents_pretrained.py). Each seed's pair of agents is pretrained
# on domain-randomized maps until a windowed win-rate plateaus, THEN scored on the
# exact same 100-episode/20-per-config schedule every other MA Red-Blue-Button
# method uses -- keep this in sync with two_opsrl.sh / two_aif_*.sh / mappo.sh; a
# mismatch here silently breaks the apples-to-apples comparison.
#
# Pretraining's own map draws use a separate RNG stream (offset from the eval
# seed inside the script itself), so they can never leak into the scored-eval
# map sequence -- confirmed in ai/02-debug.md's OPSRL pretrained-variant entries.
#
# Override at submit time, e.g.:
#   MAX_STEPS=30 sbatch cc_scripts/redbluebutton/two_opsrl_pretrained.sh
#   PRETRAIN_MAX_EPISODES=500 sbatch cc_scripts/redbluebutton/two_opsrl_pretrained.sh

set -uo pipefail                      # no -e: we still copy logs on failure

MAX_STEPS=${MAX_STEPS:-50}
PRETRAIN_MIN_EPISODES=${PRETRAIN_MIN_EPISODES:-200}
PRETRAIN_MAX_EPISODES=${PRETRAIN_MAX_EPISODES:-3000}

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
echo "---- Starting OPSRL-pretrained seed index ${SEED_IDX} (max_steps=${MAX_STEPS}, pretrain ${PRETRAIN_MIN_EPISODES}-${PRETRAIN_MAX_EPISODES} episodes) ----"

export PYTHONHASHSEED=0

# "opsrl_pretrained" (distinct from two_opsrl.sh's plain "opsrl") baked into both
# the destination folder and every copied filename so this never collides with
# or is mistaken for the cold-start OPSRL baseline's own logs.
DEST_BASE=${DEST_BASE_OVERRIDE:-"/home/toulabin/projects/aip-jrwright/toulabin/logs/ma_opsrl_pretrained_redbluebutton_step${MAX_STEPS}_30seed"}
mkdir -p "${DEST_BASE}"
LOG_FILE="$SLURM_TMPDIR/ma_opsrl_pretrained_seed${SEED_IDX}.log"

python -u run_scripts_red_blue_doors/multi_agent/run_two_opsrl_agents_pretrained.py \
  --seed ${SEED_IDX} \
  --episodes 100 \
  --episodes-per-config 20 \
  --max-steps ${MAX_STEPS} \
  --pretrain-min-episodes ${PRETRAIN_MIN_EPISODES} \
  --pretrain-max-episodes ${PRETRAIN_MAX_EPISODES} \
  --print-steps > "$LOG_FILE" 2>&1

EXIT_CODE=$?

echo "Copying logs..."
cp "$LOG_FILE" "${DEST_BASE}/" 2>/dev/null || echo "Warning: verbose log file not found"
cp logs/two_opsrl_agents_pretrained_convergence_seed${SEED_IDX}_ep*_*.csv "${DEST_BASE}/" 2>/dev/null || echo "Warning: CSV log file not found"
cp logs/two_opsrl_agents_pretrained_convergence_seed${SEED_IDX}_ep*_*_stats.json "${DEST_BASE}/" 2>/dev/null || echo "Warning: stats JSON file not found"
cp logs/two_opsrl_agents_pretrained_convergence_seed${SEED_IDX}_ep*_*_configs.json "${DEST_BASE}/" 2>/dev/null || echo "Warning: configs JSON file not found"
echo "Copy done"

if [ $EXIT_CODE -ne 0 ]; then
    echo "run_two_opsrl_agents_pretrained.py failed for seed index $SEED_IDX with exit code $EXIT_CODE"
    tail -50 "$LOG_FILE"
    exit $EXIT_CODE
fi

echo "---- OPSRL-pretrained seed index ${SEED_IDX} complete ----"
