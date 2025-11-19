#!/bin/bash
#SBATCH --account=def-jrwright            
#SBATCH --job-name=eight_agents           # job name
#SBATCH --array=0-29                      # seeds 0-29 (30 seeds total)
#SBATCH --cpus-per-task=4
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

# Run the comparison script for this seed
python run_scripts/compare_agents/compare_eight_agents.py --seed_idx ${SEED_IDX} --num_seeds 30
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "compare_eight_agents.py for seed index $SEED_IDX failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

# Copy output files to home directory
DEST_BASE="/home/toulabin/projects/def-jrwright/toulabin/logs"
mkdir -p "${DEST_BASE}"

echo "Copying logs and Q-tables to home directory..."

# Copy CSV log file (pattern: eight_agents_comparison_ep*_step*_seed${SEED_IDX}_*.csv)
cp logs/eight_agents_comparison_ep*_step*_seed${SEED_IDX}_*.csv "${DEST_BASE}/" 2>/dev/null || echo "Warning: CSV log file not found"

# Copy Q-table files (pattern: eight_agents_qtable_ep*_step*_seed${SEED_IDX}_*_seed*.json)
cp logs/eight_agents_qtable_ep*_step*_seed${SEED_IDX}_*.json "${DEST_BASE}/" 2>/dev/null || echo "Warning: Some Q-table files not found"

echo "Copy done"
echo "---- Eight Agents Seed Index ${SEED_IDX} complete ----"