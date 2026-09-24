#!/bin/bash

#SBATCH --job-name=byol-smoke
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/sp7007/byol_sigreg/logs/byol_smoke_%j.log

source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/sp7007/conda-envs/byol_sigreg

mkdir -p /scratch/sp7007/byol_sigreg/logs
mkdir -p /scratch/sp7007/byol_sigreg/checkpoints
mkdir -p /scratch/sp7007/byol_sigreg/data

cd /scratch/sp7007/byol_sigreg


echo "================ GPU CHECK ================"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("PyTorch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU count:", torch.cuda.device_count())

    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    print("CUDA tensor test:", y.device)
PY

echo "==========================================="


python train_byol.py \
    --data-dir /scratch/sp7007/byol_sigreg/data \
    --save-dir /scratch/sp7007/byol_sigreg/checkpoints \
    --epochs 1 \
    --batch-size 256 \
    --num-workers 8 \
    --lr 3e-4


echo "================ DONE ====================="
