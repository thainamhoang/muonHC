#!/bin/bash -l
#SBATCH --job-name=muonhc_generic_train
#SBATCH --output=/home/thahoa/muonHC/logs/%x-%j.out
#SBATCH --error=/home/thahoa/muonHC/logs/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus=H200:1

set -euo pipefail

module load miniforge3 cuda h100 dev2025a cmake

export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

eval "$(mamba shell hook --shell bash)"
mamba activate runai

ENV_FILE="/home/thahoa/muonHC/.env"
CONFIG_FILE=$1

if [ -f "$ENV_FILE" ]; then
    echo "Loading environment variables from $ENV_FILE"
    set -a
    source "$ENV_FILE"
    set +a
else
    echo "Warning: .env file not found at $ENV_FILE"
fi

if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "Warning: WANDB_API_KEY is not set!"
else
    echo "WANDB_API_KEY loaded successfully (length: ${#WANDB_API_KEY})"
fi

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo

echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr "," "\n" | wc -l)
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((10000 + ${SLURM_JOB_ID:-0} % 50000))

echo "Using $NUM_GPUS GPU(s): $CUDA_VISIBLE_DEVICES"
echo "Torch rendezvous: ${MASTER_ADDR}:${MASTER_PORT}"

# Temporarily disable exit-on-error to allow sleep after torchrun
set +e
torchrun \
  --nnodes=1 \
  --nproc_per_node=$NUM_GPUS \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  /home/thahoa/muonHC/training.py --config $CONFIG_FILE

TORCHRUN_EXIT=$?
echo "torchrun finished with exit code $TORCHRUN_EXIT"

# Keep node alive for 2 hours (7200 seconds)
echo "Sleeping for 2 hours to keep node alive..."
sleep 7200

# Exit with the original torchrun exit code
exit $TORCHRUN_EXIT