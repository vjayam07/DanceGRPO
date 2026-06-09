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
# Encodes CFG-RL Expo's exact 75k train / 500 eval HPDv2 prompt split.
# Output lands in atlas2 storage with split labels in videos2caption.json.
#
# This is the same as the 8-GPU version but with GPU_NUM=4 and atlas2 paths.
# Text encoding is lightweight so 4 GPUs is fine.
# =============================================================================

set -euo pipefail

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dancegrpo

# Install flash-attn on GPU node if not already present
python -c "import flash_attn" 2>/dev/null || \
  pip install packaging ninja && pip install flash-attn==2.7.0.post2 --no-build-isolation

ATLAS_BASE="/atlas2/u/vjayam/experiments/cfgrl-expo/DanceGRPO"
CFGRL_EXPO_DIR="${CFGRL_EXPO_DIR:-/atlas2/u/vjayam/experiments/cfgrl-expo}"
GPU_NUM=4
MODEL_PATH="${ATLAS_BASE}/data/flux"
OUTPUT_DIR="${ATLAS_BASE}/data/rl_embeddings"
PROMPT_PATH="${CFGRL_EXPO_DIR}/expo/envs/flux_prompts_hpdv2.json"

mkdir -p "${OUTPUT_DIR}"

torchrun --nproc_per_node=$GPU_NUM --master_port 19002 \
  fastvideo/data_preprocess/preprocess_flux_embedding.py \
  --model_path $MODEL_PATH \
  --output_dir $OUTPUT_DIR \
  --prompt_dir "$PROMPT_PATH" \
  --num_train_prompts 75000 \
  --num_eval_prompts 500 \
  --prompt_seed 42 \
  --train_test_split 0.8
