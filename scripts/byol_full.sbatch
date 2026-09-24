#!/bin/bash

#SBATCH --job-name=byol-cifar10
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/sp7007/byol_sigreg/logs/byol_full_%j.log

source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/sp7007/conda-envs/byol_sigreg

cd /scratch/sp7007/byol_sigreg

echo "================ GPU CHECK ================"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

echo "==========================================="

echo ""
echo "========= BYOL PRETRAINING ========="

python train_ssl.py \
    --config configs/byol_cifar10.yaml

CHECKPOINT=/scratch/sp7007/byol_sigreg/outputs/byol_cifar10_r18/seed_0/checkpoints/epoch_100.pt

echo ""
echo "========= LINEAR PROBE ========="

python linear_probe.py \
    --config configs/byol_cifar10.yaml \
    --checkpoint "$CHECKPOINT"

echo ""
echo "========= EXPERIMENT DONE ========="

