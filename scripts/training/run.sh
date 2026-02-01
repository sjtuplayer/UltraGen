#!/bin/bash
# ===================================================================
# UltraGen Multi-Machine Distributed Training Script
# ===================================================================
# This script sets up and launches distributed training across multiple
# machines using PyTorch Distributed Data Parallel (DDP).
#
# Prerequisites:
#   1. Create a hosts file (e.g., hosts.txt) with one IP per line
#   2. Ensure SSH access to all machines
#   3. All machines should have the same conda environment
#   4. All machines should have access to shared storage for dataset
#
# Usage:
#   bash scripts/training/run.sh
# ===================================================================

# -------------------------------------------------------------------
# Network Configuration for NCCL (adjust for your cluster)
# -------------------------------------------------------------------
export NCCL_IB_TIMEOUT=24
export NCCL_NVLS_ENABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECK_DISABLE=1
export NCCL_P2P_DISABLE=0          # Enable P2P for faster communication
export NCCL_IB_DISABLE=0           # Enable InfiniBand if available
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME=bond1    # Change to your network interface (e.g., eth0, ib0)
export UCX_NET_DEVICES=bond1
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
export NCCL_COLLNET_ENABLE=0
export SHARP_COLL_ENABLE_SAT=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=0
export NCCL_DEBUG=WARN             # Change to INFO for debugging

# -------------------------------------------------------------------
# Project Configuration
# -------------------------------------------------------------------
# Change to your project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Conda environment setup (modify path if needed)
# Option 1: If conda is in PATH
conda activate ultragen

# Option 2: If you need to specify conda path
# eval "$(/path/to/conda/bin/conda shell.bash hook)"
# conda activate ultragen

# -------------------------------------------------------------------
# Distributed Training Configuration
# -------------------------------------------------------------------
# Hosts file: one IP address per line, first line is master node
HOSTS_FILE="hosts.txt"             # Change to your hosts file path

# Number of machines
N_NODES=8                          # Change to your number of machines

# Number of GPUs per machine
GPUS_PER_NODE=8                    # Change to your GPUs per machine

# Master port for distributed communication
MASTER_PORT=12346                  # Change if port is occupied

# -------------------------------------------------------------------
# Training Paths Configuration
# -------------------------------------------------------------------
# Dataset path (should be accessible from all nodes)
DATASET_PATH="./data/UltraVideo"   # UltraVideo dataset or your custom dataset

# Base model checkpoint
DIT_PATH="./checkpoints/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"

# Resume from checkpoint (optional, comment out if training from scratch)
# DIT_LOAD_PATH="./checkpoints/epoch=22-step=1817.ckpt"

# Output directory for trained models and logs
OUTPUT_PATH="./experiments/ultragen_distributed"

# -------------------------------------------------------------------
# Training Hyperparameters
# -------------------------------------------------------------------
TRAIN_ARCHITECTURE="full"
LEARNING_RATE=1e-4
STEPS_PER_EPOCH=5000
MAX_EPOCHS=1000
ACCUMULATE_GRAD_BATCHES=1

# Video resolution settings
HEIGHT=1088                        # 1080P height
WIDTH=1920                         # 1080P width
NUM_FRAMES=81                      # Number of frames

# Enable gradient checkpointing to save memory
USE_GRADIENT_CHECKPOINTING="--use_gradient_checkpointing"

# For 4K training, uncomment the following:
# USE_4K_VIDEO="--use_4k_video"
# HEIGHT=2160
# WIDTH=3840
# NUM_FRAMES=33

# -------------------------------------------------------------------
# Automatic Node Configuration
# -------------------------------------------------------------------
# Get current machine's IP/hostname
HOSTNAME=${LOCAL_IP:-$(hostname -I | awk '{print $1}')}

# Find node rank from hosts file (0-indexed)
NODE_RANK=$(awk -v hostname="$HOSTNAME" '{
    if ($0 == hostname) {
        print NR-1
        exit
    }
}' "$HOSTS_FILE")

# Validate node rank
if [ -z "$NODE_RANK" ]; then
    echo "Error: Cannot find current host ($HOSTNAME) in $HOSTS_FILE"
    echo "Please ensure your host is listed in the hosts file."
    exit 1
fi

echo "====================================================================="
echo "Node Rank: $NODE_RANK"
echo "====================================================================="

# Get master node address (first line in hosts file)
MASTER_ADDR=$(head -n 1 "$HOSTS_FILE")
echo "Master Address: $MASTER_ADDR"
echo "====================================================================="

# -------------------------------------------------------------------
# Launch Training
# -------------------------------------------------------------------
echo "Starting distributed training..."
echo "Configuration:"
echo "  - Nodes: $N_NODES"
echo "  - GPUs per node: $GPUS_PER_NODE"
echo "  - Dataset: $DATASET_PATH"
echo "  - Output: $OUTPUT_PATH"
echo "  - Architecture: $TRAIN_ARCHITECTURE"
echo "  - Resolution: ${WIDTH}x${HEIGHT}, ${NUM_FRAMES} frames"
echo "====================================================================="

# Build the training command
TRAIN_CMD="torchrun \
  --nproc_per_node=$GPUS_PER_NODE \
  --nnodes=$N_NODES \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  tools/training/train.py \
  --task train \
  --train_architecture $TRAIN_ARCHITECTURE \
  --dataset_path $DATASET_PATH \
  --dit_path $DIT_PATH \
  --output_path $OUTPUT_PATH \
  --height $HEIGHT \
  --width $WIDTH \
  --num_frames $NUM_FRAMES \
  --steps_per_epoch $STEPS_PER_EPOCH \
  --max_epochs $MAX_EPOCHS \
  --learning_rate $LEARNING_RATE \
  --accumulate_grad_batches $ACCUMULATE_GRAD_BATCHES \
  $USE_GRADIENT_CHECKPOINTING \
  --n_nodes $N_NODES"

# Add resume checkpoint if specified
if [ -n "$DIT_LOAD_PATH" ]; then
    TRAIN_CMD="$TRAIN_CMD --dit_load_path $DIT_LOAD_PATH"
fi

# Add 4K flag if specified
if [ -n "$USE_4K_VIDEO" ]; then
    TRAIN_CMD="$TRAIN_CMD $USE_4K_VIDEO"
fi

# Execute training command
eval $TRAIN_CMD &

# -------------------------------------------------------------------
# Launch on All Nodes (using pssh)
# -------------------------------------------------------------------
# If you're running this on the master node and want to automatically
# launch on all other nodes, uncomment the following line:
# pssh -i -t 0 -h "$HOSTS_FILE" "cd $PROJECT_ROOT && bash scripts/training/run.sh"

echo "====================================================================="
echo "Training launched! Monitor progress in: $OUTPUT_PATH"
echo "====================================================================="
