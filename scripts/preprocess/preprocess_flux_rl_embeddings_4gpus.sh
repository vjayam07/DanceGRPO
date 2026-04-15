#!/bin/bash
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
