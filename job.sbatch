#!/bin/bash
#SBATCH --job-name=example_train
#SBATCH --account=me_jiechen
#SBATCH --partition=a30_normal_q             # or a30_normal_q / v100_normal_q
#SBATCH --qos=fal_a30_normal_base             # pick *_short or *_long as needed
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/andrew03/example_arc_repo/logs/%x_%j.out
#SBATCH --error=/home/andrew03/example_arc_repo/logs/%x_%j.err
#SBATCH --chdir=/home/andrew03/example_arc_repo

set -euo pipefail
module reset
module load Python/3.11.3-GCCcore-12.3.0       # load the same/compatible module on Falcon
source /home/andrew03/example_arc_repo/venvs/ex_311_a30_normal_q/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

python -u train.py
