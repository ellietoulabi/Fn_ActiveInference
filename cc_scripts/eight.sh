#!/bin/bash
#SBATCH --account=def-jrwright            
#SBATCH --job-name=redblueseed            # job name
#SBATCH --array=0-4                       # seeds 0,1,2,3,4
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
git clone --quiet https://github.com/ellietoulabi/AIF_RedBlueDoors.git 
echo "Repository cloned."


echo "Creating virtual environment..."
cd ../virtualenvs
python3.11 -m venv .venv
echo "Virtual environment created"
source .venv/bin/activate
echo "Activated virtualenv."


echo "Installing dependencies..."
cd ../project/AIF_RedBlueDoors/
pip install -r requirements.txt
echo "Dependencies installed"

SEED=$SLURM_ARRAY_TASK_ID
echo "---- Starting seed ${SEED} ----"


cd runs



python run_redbluedoors_aif_aif.py --seed ${SEED} --episodes 1000 --max_steps 150 --change_every 50
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "run_redbluedoors_aif_aif.py for seed $SEED failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

SEED_LOG_DIR="aif_aif_log_seed_${SEED}.csv"
DEST_BASE="/home/toulabin/projects/def-jrwright/toulabin/logs"

mkdir -p "${DEST_BASE}"

echo "Copying logs to home directory..."
cp "${SEED_LOG_DIR}" "${DEST_BASE}/"

echo "Copy done"
echo "---- AIF AIF Seed ${SEED} complete ----"