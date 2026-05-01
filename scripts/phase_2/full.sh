#!/bin/bash -l
#SBATCH --job-name=muonhc_phase2_full
#SBATCH --output=/home/thahoa/muonHC/logs/%x-%j.out
#SBATCH --error=/home/thahoa/muonHC/logs/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus=H100:1

set -euo pipefail

log() {
  echo "[$(date)] $*"
}

module load miniforge3 cuda h100 dev2025a cmake

export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

eval "$(mamba shell hook --shell bash)"
mamba activate runai

ENV_FILE="/home/thahoa/muonHC/.env"
CONFIG_FILE="/home/thahoa/muonHC/configs/phase_2/cfgs_full.yaml"

if [ -f "$ENV_FILE" ]; then
    echo "Loading environment variables from $ENV_FILE"
    # set -a: automatically export all variables defined in the following source command
    set -a
    source "$ENV_FILE"
    set +a
else
    echo "Warning: .env file not found at $ENV_FILE"
fi

# Verify WANDB_API_KEY is loaded (optional debug)
if [ -z "$WANDB_API_KEY" ]; then
    echo "Warning: WANDB_API_KEY is not set!"
else
    echo "WANDB_API_KEY loaded successfully (length: ${#WANDB_API_KEY})"
fi
export OMP_NUM_THREADS=1

# Verify GPU allocation
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Auto-detect number of GPUs from Slurm
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr "," "\n" | wc -l)

echo "Using $NUM_GPUS GPU(s): $CUDA_VISIBLE_DEVICES"

# Launch training
torchrun --nproc_per_node=$NUM_GPUS /home/thahoa/muonHC/training.py --config $CONFIG_FILE