#!/bin/bash
#SBATCH --job-name=regression
#SBATCH --time=00:30:00
#SBATCH --mail-type=end
#SBATCH --account=jiankang1
#SBATCH --partition=standard
#SBATCH --mem=15g
#SBATCH --cpus-per-task=1
#SBATCH --array=1
##SBATCH --output=/home/%u/slurm_output/slurm-%A_%a.out
module load python3.9-anaconda/2021.11
source /home/${USER}/.bashrc
source activate torchgpu
srun python run_linear_voxel.py $SLURM_ARRAY_TASK_ID