#!/bin/bash -l
#SBATCH --job-name=muonhc_dual_train
#SBATCH --output=/home/thahoa/muonHC/logs/%x-%j.out
#SBATCH --error=/home/thahoa/muonHC/logs/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus=H200:1

set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: sbatch scripts/dual_train.sh <config_1.yaml> <config_2.yaml>"
    echo "Example:"
    echo "  sbatch scripts/dual_train.sh \\"
    echo "    /home/thahoa/muonHC/configs/phase_3/cfgs_full_muon.yaml \\"
    echo "    /home/thahoa/muonHC/configs/phase_3/cfgs_full_spatial_muon.yaml"
    exit 2
fi

CONFIG_FILE_1="$1"
CONFIG_FILE_2="$2"

module load miniforge3 cuda h100 dev2025a cmake

export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

eval "$(mamba shell hook --shell bash)"
mamba activate runai

ENV_FILE="/home/thahoa/muonHC/.env"

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

export OMP_NUM_THREADS="${OMP_NUM_THREADS_PER_RUN:-1}"
export PYTHONUNBUFFERED=1
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export MASTER_ADDR=127.0.0.1

echo "Allocated GPUs: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

BASE_PORT=$((10000 + ${SLURM_JOB_ID:-0} % 40000))
MASTER_PORT_1=$BASE_PORT
MASTER_PORT_2=$((BASE_PORT + 1))

echo "Dual training on one node/GPU allocation"
echo "  config 1: $CONFIG_FILE_1"
echo "  config 2: $CONFIG_FILE_2"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "  OMP_NUM_THREADS per run: $OMP_NUM_THREADS"
echo "  rendezvous 1: ${MASTER_ADDR}:${MASTER_PORT_1}"
echo "  rendezvous 2: ${MASTER_ADDR}:${MASTER_PORT_2}"

cleanup() {
    if [ -n "${PID_1:-}" ] && kill -0 "$PID_1" 2>/dev/null; then
        kill "$PID_1" 2>/dev/null || true
    fi
    if [ -n "${PID_2:-}" ] && kill -0 "$PID_2" 2>/dev/null; then
        kill "$PID_2" 2>/dev/null || true
    fi
}
trap cleanup INT TERM

run_one() {
    local run_id="$1"
    local config_file="$2"
    local master_port="$3"

    echo "[run ${run_id}] starting: ${config_file}"
    torchrun \
      --nnodes=1 \
      --nproc_per_node=1 \
      --master_addr="$MASTER_ADDR" \
      --master_port="$master_port" \
      /home/thahoa/muonHC/training.py --config "$config_file"
}

run_one 1 "$CONFIG_FILE_1" "$MASTER_PORT_1" &
PID_1=$!

sleep "${DUAL_LAUNCH_STAGGER_SECONDS:-30}"

run_one 2 "$CONFIG_FILE_2" "$MASTER_PORT_2" &
PID_2=$!

STATUS_1=0
STATUS_2=0

wait "$PID_1" || STATUS_1=$?
wait "$PID_2" || STATUS_2=$?

echo "Run 1 exit status: $STATUS_1"
echo "Run 2 exit status: $STATUS_2"

if [ "$STATUS_1" -ne 0 ] || [ "$STATUS_2" -ne 0 ]; then
    exit 1
fi
