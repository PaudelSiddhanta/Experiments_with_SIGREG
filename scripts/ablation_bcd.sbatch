#!/bin/bash

#SBATCH --job-name=ssl-bcd
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/sp7007/byol_sigreg/logs/ablation_bcd_%j.log

source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/sp7007/conda-envs/byol_sigreg

cd /scratch/sp7007/byol_sigreg

echo "================ GPU CHECK ================"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

for CFG in \
    B_byol_sigreg \
    C_noema_sigreg \
    D_nostop_sigreg
do

    echo ""
    echo "================================================"
    echo "PRETRAINING: $CFG"
    echo "================================================"

    python train_ssl.py \
        --config configs/${CFG}.yaml

    CKPT=/scratch/sp7007/byol_sigreg/outputs/${CFG}/seed_0/checkpoints/epoch_100.pt

    echo ""
    echo "================================================"
    echo "LINEAR PROBE: $CFG"
    echo "================================================"

    python linear_probe.py \
        --config configs/${CFG}.yaml \
        --checkpoint "$CKPT"

done

echo "================ B/C/D DONE ================"
