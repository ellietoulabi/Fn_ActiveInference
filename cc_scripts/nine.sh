#!/bin/bash
#SBATCH --account=def-jrwright
#SBATCH --job-name=nine_agents
#SBATCH --array=0-4
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0-8:59

module purge
module load python/3.11.4  scipy-stack

if [ "$SLURM_TMPDIR" == "" ]; then
    echo "Error: SLURM_TMPDIR not defined"
    exit 1
fi

cd $SLURM_TMPDIR
echo "Working in SLURM_TMPDIR: $SLURM_TMPDIR"

mkdir project
mkdir virtualenvs
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
pip install -r requirements.txt
echo "Dependencies installed"

SEED_IDX=$SLURM_ARRAY_TASK_ID
echo "---- Starting seed index ${SEED_IDX} ----"

# Reproducible runs: seed is passed via --seed_idx; Python script uses BASE_SEED + seed_idx.
# PYTHONHASHSEED=0 makes dict/set iteration order deterministic across runs.
export PYTHONHASHSEED=0
python run_scripts_red_blue_doors/compare_agents/compare_nine_agents.py --seed_idx ${SEED_IDX} --num_seeds 5
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "compare_nine_agents.py for seed index $SEED_IDX failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

DEST_BASE="${HOME}/projects/def-jrwright/toulabin/logs"
mkdir -p "${DEST_BASE}"

echo "Copying logs and Q-tables to home directory..."

cp logs/nine_agents_comparison_ep*_step*_seed${SEED_IDX}_*.csv "${DEST_BASE}/" 2>/dev/null || echo "Warning: CSV log file not found"
cp logs/nine_agents_qtable_ep*_step*_seed${SEED_IDX}_*.json "${DEST_BASE}/" 2>/dev/null || echo "Warning: Some Q-table files not found"

echo "Copy done"
echo "---- Nine Agents Seed Index ${SEED_IDX} complete ----"
