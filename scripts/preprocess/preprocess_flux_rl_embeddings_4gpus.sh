#!/bin/bash
#SBATCH --job-name=danceGRPO-preprocess-embeddings
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --gres=gpu:a6000ada:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/atlas2/u/vjayam/experiments/cfgrl/logs/slurm/danceGRPO_preprocess_embeddings_%j.out
#SBATCH --error=/atlas2/u/vjayam/experiments/cfgrl/logs/slurm/danceGRPO_preprocess_embeddings_%j.err

# =============================================================================
# Preprocess FLUX text embeddings — 4 GPUs (A6000 Ada)
# =============================================================================
# Encodes prompts from assets/prompts.txt into text embeddings for training.
# Output lands in atlas2 storage (prompt_embed/ + videos2caption.json).
#
# This is the same as the 8-GPU version but with GPU_NUM=4 and atlas2 paths.
# Text encoding is lightweight so 4 GPUs is fine.
# =============================================================================

set -euo pipefail

ATLAS_BASE="/atlas2/u/vjayam/experiments/cfgrl-expo/DanceGRPO"
GPU_NUM=4
MODEL_PATH="${ATLAS_BASE}/data/flux"
OUTPUT_DIR="${ATLAS_BASE}/data/rl_embeddings"

mkdir -p "${OUTPUT_DIR}"

torchrun --nproc_per_node=$GPU_NUM --master_port 19002 \
  fastvideo/data_preprocess/preprocess_flux_embedding.py \
  --model_path $MODEL_PATH \
  --output_dir $OUTPUT_DIR \
  --prompt_dir "./assets/prompts.txt"
