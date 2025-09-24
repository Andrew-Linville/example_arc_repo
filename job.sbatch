#!/bin/bash
#SBATCH --job-name=fake-train
#SBATCH --account=personal
#SBATCH --partition=a30_normal_q
#SBATCH --qos=fal_a30_normal_base
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --chdir=/home/$USER/example-train-repo

set -euo pipefail

module reset
module load Python/3.11.3-GCCcore-12.3.0x


source ./ex_311_a30_normal_q/bin/activate

python train.py

