#!/bin/bash

#SBATCH --job-name=ssl-ablation-smoke
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/sp7007/byol_sigreg/logs/ablation_smoke_%j.log

source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/sp7007/conda-envs/byol_sigreg

cd /scratch/sp7007/byol_sigreg

echo "================ GPU CHECK ================"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

echo "==========================================="

for CFG in \
    A_byol_smoke \
    B_byol_sigreg_smoke \
    C_noema_sigreg_smoke \
    D_nostop_sigreg_smoke \
    E_symmetric_sigreg_smoke \
    F_symmetric_noreg_smoke

do

    echo ""
    echo "================================================"
    echo "RUNNING: $CFG"
    echo "================================================"

    python train_ssl.py \
        --config configs/${CFG}.yaml

done

echo ""
echo "================ ALL SMOKES DONE ================"
