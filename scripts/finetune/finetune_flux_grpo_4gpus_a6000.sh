#!/bin/bash
#SBATCH --job-name=danceGRPO-finetune-full
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --gres=gpu:a6000ada:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=72:00:00
#SBATCH --output=/atlas2/u/vjayam/experiments/cfgrl/logs/slurm/danceGRPO_finetune_full_%j.out
#SBATCH --error=/atlas2/u/vjayam/experiments/cfgrl/logs/slurm/danceGRPO_finetune_full_%j.err

# =============================================================================
# DanceGRPO FLUX full-model GRPO - 4x A6000 Ada (48GB)
# =============================================================================
#
# Matches finetune_flux_grpo_4gpus_lora_a6000.sh where applicable:
#   - 4 GPUs, 512x512 images, 4 sampling steps, and seed 42
#   - batch size 1, 12 gradient accumulation steps, and 12 generations
#   - HPSv2 reward and the same dataset/storage layout
#
# Full-model training uses FSDP full sharding and BF16 master weights to reduce
# per-GPU memory use. EMA is intentionally disabled because it keeps another
# full copy of the model.
# =============================================================================

set -euo pipefail

export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
# Uncomment to disable wandb:
# export WANDB_DISABLED=true

ATLAS_BASE="/atlas2/u/vjayam/experiments/cfgrl-expo/DanceGRPO"
OUTPUT_DIR="${ATLAS_BASE}/data/outputs/grpo_full_4gpu_a6000"
DATA_DIR="${ATLAS_BASE}/data"
CACHE_DIR="${ATLAS_BASE}/data/.cache"
HPS_CKPT_DIR="${ATLAS_BASE}/hps_ckpt"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${DATA_DIR}/rl_embeddings"
mkdir -p "${CACHE_DIR}"
mkdir -p "${HPS_CKPT_DIR}"

# HPSv2 reward model (needed on first run)
if [ ! -d "HPSv2" ]; then
  echo ">>> Installing HPSv2 reward model..."
  git clone https://github.com/tgxs002/HPSv2.git
  cd HPSv2 && pip install -e . && cd ..
fi

for checkpoint in open_clip_pytorch_model.bin HPS_v2.1_compressed.pt; do
  if [ ! -f "${HPS_CKPT_DIR}/${checkpoint}" ]; then
    echo "Missing HPSv2 checkpoint: ${HPS_CKPT_DIR}/${checkpoint}" >&2
    echo "Run scripts/setup_baseline_run.sh or download the checkpoint before training." >&2
    exit 1
  fi
done

torchrun --nproc_per_node=4 --master_port 19002 \
  fastvideo/train_grpo_flux.py \
  --seed 42 \
  --pretrained_model_name_or_path "${DATA_DIR}/flux" \
  --vae_model_path "${DATA_DIR}/flux" \
  --cache_dir "${CACHE_DIR}" \
  --data_json_path "${DATA_DIR}/rl_embeddings/videos2caption.json" \
  --gradient_checkpointing \
  --train_batch_size 1 \
  --num_latent_t 1 \
  --sp_size 1 \
  --train_sp_batch_size 1 \
  --dataloader_num_workers 4 \
  --gradient_accumulation_steps 12 \
  --max_train_steps 1000 \
  --learning_rate 1e-5 \
  --mixed_precision bf16 \
  --master_weight_type bf16 \
  --fsdp_sharding_startegy full \
  --checkpointing_steps 50 \
  --allow_tf32 \
  --cfg 0.0 \
  --output_dir "${OUTPUT_DIR}" \
  --h 512 \
  --w 512 \
  --t 1 \
  --sampling_steps 4 \
  --eta 0.3 \
  --lr_warmup_steps 0 \
  --sampler_seed 1223627 \
  --max_grad_norm 0.01 \
  --weight_decay 0.0001 \
  --use_hpsv2 \
  --num_generations 12 \
  --shift 3 \
  --use_group \
  --ignore_last \
  --timestep_fraction 0.6 \
  --init_same_noise \
  --clip_range 1e-4 \
  --adv_clip_max 5.0 \
  --hps_ckpt_dir "${HPS_CKPT_DIR}"
