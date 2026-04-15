#!/bin/bash
# =============================================================================
# Preprocess FLUX text embeddings — 4 GPUs (A6000 Ada)
# =============================================================================
# Encodes prompts from assets/prompts.txt into text embeddings for training.
# Output lands in data/rl_embeddings/ (prompt_embed/ + videos2caption.json).
#
# This is the same as the 8-GPU version but with GPU_NUM=4.
# Text encoding is lightweight so 4 GPUs is fine.
# =============================================================================

set -euo pipefail

GPU_NUM=4
MODEL_PATH="data/flux"
OUTPUT_DIR="data/rl_embeddings"

torchrun --nproc_per_node=$GPU_NUM --master_port 19002 \
    fastvideo/data_preprocess/preprocess_flux_embedding.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --prompt_dir "./assets/prompts.txt"
