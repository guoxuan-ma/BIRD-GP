#!/bin/bash
#SBATCH --job-name=fit_all
#SBATCH --time=15:00:00
#SBATCH --mail-type=end
#SBATCH --account=jiankang1
#SBATCH --partition=spgpu
#SBATCH --mem=10g
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --array=1
#SBATCH --exclude=gl1507
##SBATCH --output=/home/%u/slurm_output/slurm-%A_%a.out
module load python3.9-anaconda/2021.11
source /home/${USER}/.bashrc
source activate torchgpu
srun python birdgp_fit_image_all_for_importance.py $SLURM_ARRAY_TASK_ID