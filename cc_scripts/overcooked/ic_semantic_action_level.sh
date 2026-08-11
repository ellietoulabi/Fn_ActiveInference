#!/bin/bash
#SBATCH --account=aip-jrwright
#SBATCH --job-name=aif_ic_sal_30seed_collisionfix
#SBATCH --array=0-29                  # 30 seeds (one episode per task) -- fixed pool: ep=76..105, a0=1000..1029, a1=2000..2029
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=4-00:00
#SBATCH --output=ic_sal_30seed_%A_%a.out

# IndividuallyCollective paradigm, semantic-action level, one seed per array
# task. Uses the CollisionFix stack (ai/02-debug.md sections J.8-J.9) -- the
# only IC variant with the collision-blindness fix and the PROGRESS_SUCCESS_PROB
# fix; the plain/base stack this launcher used to point at produces zero
# deliveries. IC is the most expensive paradigm (400 joint policies x 2 agents
# per step), hence the larger --time budget.
#
# Full logging: per-step CSV, per-step JSONL (map + full state beliefs incl.
# entropy + top-5 of the 400 joint policy probabilities), and verbose
# human-readable stdout (--log-steps --policy-log-top-k 5) captured in the
# copied-back .log file. Logs top-5 policies, not the full 400-entry q_pi
# array, to keep per-step JSONL size manageable -- --log-jsonl alone used to
# force the full array in regardless (fixed 2026-08-11, see
# run_individually_collective_policy_semantic_action_level_seed_sweep_optimized_collision_fix.py).
#
# Seed pool is shared, fixed, and identical in derivation across the ind/ic/fc
# launchers so the same 30 (episode, agent0, agent1) seed-triples are used for
# all three paradigms -- a fair, matched comparison.
#
# Override episode length / precision at submit time, e.g.:
#   MAX_STEPS=500 sbatch cc_scripts/overcooked/ic_semantic_action_level.sh
set -uo pipefail                      # no -e: we still copy logs on failure

MAX_STEPS=${MAX_STEPS:-1500}
GAMMA=${GAMMA:-4.0}
ALPHA=${ALPHA:-1.0}
DEST_BASE=${DEST_BASE_OVERRIDE:-"/home/toulabin/projects/aip-jrwright/toulabin/logs/sal_ic_30seed_collisionfix"}

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

echo "Installing dependencies (cc_scripts/overcooked/requirements-cc-sal.txt; not full requirements.txt)..."
cd ../project/Fn_ActiveInference/
if ! pip install --no-input -r cc_scripts/overcooked/requirements-cc-sal.txt; then
    echo "ERROR: pip install failed. Do not use requirements.txt on Alliance (opencv-python dummy wheel)."
    exit 1
fi
echo "Dependencies installed."

# shellcheck source=overcooked/_sal_common.sh
source cc_scripts/overcooked/_sal_common.sh
sal_ensure_venv_runtime_deps || exit 1
sal_verify_imports || exit 1
sal_preflight ic || exit 1

SEED_IDX=${SLURM_ARRAY_TASK_ID}
EP_SEED=$((76 + SEED_IDX))
A0_SEED=$((1000 + SEED_IDX))
A1_SEED=$((2000 + SEED_IDX))
echo "---- ic seed_idx=${SEED_IDX} ep=${EP_SEED} a0=${A0_SEED} a1=${A1_SEED} max_steps=${MAX_STEPS} ----"

mkdir -p "$DEST_BASE"
CSV_DIR="$SLURM_TMPDIR/logs_sal"
mkdir -p "$CSV_DIR"
LOG_FILE="$SLURM_TMPDIR/ic_sal_30seed_collisionfix_ep${EP_SEED}_a0_${A0_SEED}_a1_${A1_SEED}.log"

python -u run_scripts_overcooked/run_individually_collective_policy_semantic_action_level_seed_sweep_optimized_collision_fix.py \
  --n-runs 1 \
  --episode-seeds ${EP_SEED} \
  --agent0-seeds ${A0_SEED} \
  --agent1-seeds ${A1_SEED} \
  --gamma ${GAMMA} --alpha ${ALPHA} \
  --max-steps ${MAX_STEPS} \
  --log-csv --log-jsonl --log-steps --policy-log-top-k 5 --log-dir "$CSV_DIR" > "$LOG_FILE" 2>&1
EXIT_CODE=$?

sal_copy_artifacts "$DEST_BASE" "$LOG_FILE" "$CSV_DIR"

if [ $EXIT_CODE -ne 0 ]; then
    echo "ic sweep failed (seed_idx=${SEED_IDX}) exit=${EXIT_CODE}"
    sal_report_failure "$LOG_FILE"
    exit $EXIT_CODE
fi
echo "---- ic seed_idx=${SEED_IDX} complete ----"
