#!/bin/bash
#SBATCH --job-name=0socirand
#SBATCH --time=5:00:00
#SBATCH --mail-type=end
#SBATCH --account=jiankang1
#SBATCH --partition=standard
#SBATCH --mem=8g
#SBATCH --cpus-per-task=1
#SBATCH --array=21-100
##SBATCH --output=/home/%u/slurm_output/slurm-%A_%a.out
module load python3.9-anaconda/2021.11
source /home/${USER}/.bashrc
source activate torchgpu
srun python run_birdgp_lr+svgd.py $SLURM_ARRAY_TASK_ID