#!/bin/bash

#SBATCH --job-name=sigreg-weight-smoke
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/sp7007/byol_sigreg/logs/sigreg_weight_smoke_%j.log

source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/sp7007/conda-envs/byol_sigreg

cd /scratch/sp7007/byol_sigreg

for CFG in \
    B_byol_sigreg_smoke \
    C_noema_sigreg_smoke \
    D_nostop_sigreg_smoke \
    E_symmetric_sigreg_smoke
do

    echo ""
    echo "===================================="
    echo "RUNNING: $CFG"
    echo "===================================="

    python train_ssl.py \
        --config configs/${CFG}.yaml

done
