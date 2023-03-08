#!/bin/bash -x

#SBATCH -n 8                # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -p tambe
#SBATCH -t 1-18:00:00         # Runtime in D-HH:MM:SS, minimum of 10 minutes
#SBATCH --mem=32000          # Memory pool for all cores (see also --mem-per-cpu) MBs
#SBATCH -o joblogs/%A_%a.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e joblogs/%A_%a.out # File to which STDERR will be written, %j inserts jobid

set -x

date
cdir=$(pwd)

module load python/3.7.7-fasrc01
module load Gurobi/9.1.2
source /n/home12/susobhan/.bashrc
conda activate /n/home12/susobhan/.conda/envs/rmab


which python
python runExps.py ${SLURM_ARRAY_TASK_ID}

