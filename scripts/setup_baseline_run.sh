#!/bin/bash
# =============================================================================
# DanceGRPO Baseline Run — Full Setup for 4x A6000 Ada (48GB)
# =============================================================================
# End-to-end guide for running a DanceGRPO FLUX LoRA baseline to compare
# against cfgrl-expo. Run each step sequentially.
#
# Comparison setup:
#   - Model:      FLUX.1-dev (same as cfgrl-expo)
#   - Reward:     HPSv2      (same as cfgrl-expo)
#   - Prompts:    HPDv2      (same source as cfgrl-expo)
#   - Resolution: 512x512    (matches cfgrl-expo)
#   - Method:     DanceGRPO with LoRA (rank 128)
#   - Hardware:   4x A6000 Ada (48GB each), use 4 of 5 available
#
# All data, checkpoints, images, and logs are saved to:
#   /atlas2/u/vjayam/experiments/cfgrl-expo/DanceGRPO/
#
# Expected training time: ~1-2 days for 300 iterations
# Expected outcome: HPSv2 reward increase from ~0.27 to ~0.31+
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."  # cd to DanceGRPO root

ATLAS_BASE="/atlas2/u/vjayam/experiments/cfgrl-expo/DanceGRPO"

echo "============================================"
echo " DanceGRPO Baseline Setup"
echo " Hardware: 4x NVIDIA A6000 Ada (48GB)"
echo " Storage:  ${ATLAS_BASE}"
echo "============================================"

# =========================================================
# STEP 1: Environment Setup
# =========================================================
step1_environment() {
    echo ""
    echo ">>> STEP 1: Environment Setup"
    echo "    Install PyTorch 2.5, Flash Attention 2, and fastvideo"
    echo ""

    # Core dependencies
    pip install torch==2.5.0 torchvision --index-url https://download.pytorch.org/whl/cu121

    # Flash Attention 2 (will compile for Ada Lovelace arch - sm_89)
    pip install packaging ninja
    pip install flash-attn==2.7.0.post2 --no-build-isolation

    # Install fastvideo package
    pip install -e .

    # Additional dependencies
    pip install ml-collections absl-py inflect==6.0.4 pydantic==1.10.9 \
        huggingface_hub==0.24.0 protobuf==3.20.0

    # HPSv2 reward model
    if [ ! -d "HPSv2" ]; then
        git clone https://github.com/tgxs002/HPSv2.git
        cd HPSv2 && pip install -e . && cd ..
    fi

    echo ">>> STEP 1 complete."
}

# =========================================================
# STEP 2: Download Checkpoints
# =========================================================
step2_checkpoints() {
    echo ""
    echo ">>> STEP 2: Download Checkpoints"
    echo "    Download FLUX.1-dev and HPSv2 reward model weights"
    echo "    All saved to: ${ATLAS_BASE}/data/"
    echo ""

    # Create directories on atlas2
    mkdir -p "${ATLAS_BASE}/data/flux"
    mkdir -p "${ATLAS_BASE}/data/.cache"
    mkdir -p "${ATLAS_BASE}/hps_ckpt"

    echo "--- Download FLUX.1-dev ---"
    echo "Run: huggingface-cli download black-forest-labs/FLUX.1-dev --local-dir ${ATLAS_BASE}/data/flux"
    echo ""
    echo "--- Download HPS-v2.1 checkpoint ---"
    echo "Run: huggingface-cli download xswu/HPSv2 HPS_v2.1_compressed.pt --local-dir ${ATLAS_BASE}/hps_ckpt"
    echo ""
    echo "--- Download CLIP ViT-H-14 checkpoint ---"
    echo "Run: huggingface-cli download laion/CLIP-ViT-H-14-laion2B-s32B-b79K open_clip_pytorch_model.bin --local-dir ${ATLAS_BASE}/hps_ckpt"
    echo ""

    # Uncomment these to run automatically:
    # huggingface-cli download black-forest-labs/FLUX.1-dev --local-dir "${ATLAS_BASE}/data/flux"
    # huggingface-cli download xswu/HPSv2 HPS_v2.1_compressed.pt --local-dir "${ATLAS_BASE}/hps_ckpt"
    # huggingface-cli download laion/CLIP-ViT-H-14-laion2B-s32B-b79K open_clip_pytorch_model.bin --local-dir "${ATLAS_BASE}/hps_ckpt"

    echo ">>> STEP 2: Verify these files exist before proceeding:"
    echo "    ${ATLAS_BASE}/data/flux/transformer/    (FLUX transformer weights)"
    echo "    ${ATLAS_BASE}/data/flux/vae/             (FLUX VAE)"
    echo "    ${ATLAS_BASE}/data/flux/text_encoder/    (CLIP-L)"
    echo "    ${ATLAS_BASE}/data/flux/text_encoder_2/  (T5-XXL)"
    echo "    ${ATLAS_BASE}/hps_ckpt/HPS_v2.1_compressed.pt"
    echo "    ${ATLAS_BASE}/hps_ckpt/open_clip_pytorch_model.bin"
}

# =========================================================
# STEP 3: Preprocess Embeddings
# =========================================================
step3_preprocess() {
    echo ""
    echo ">>> STEP 3: Preprocess Text Embeddings (4 GPUs)"
    echo "    Encodes HPDv2 prompts with FLUX text encoders."
    echo "    Output: ${ATLAS_BASE}/data/rl_embeddings/"
    echo ""

    bash scripts/preprocess/preprocess_flux_rl_embeddings_4gpus.sh

    echo ">>> STEP 3 complete. Verify ${ATLAS_BASE}/data/rl_embeddings/videos2caption.json exists."
}

# =========================================================
# STEP 4: Run Training
# =========================================================
step4_train() {
    echo ""
    echo ">>> STEP 4: Launch DanceGRPO Training"
    echo "    4x A6000 Ada, FLUX LoRA, 512x512, HPSv2 reward"
    echo "    Output: ${ATLAS_BASE}/data/outputs/grpo_baseline_4gpu_a6000/"
    echo ""

    bash scripts/finetune/finetune_flux_grpo_4gpus_lora_a6000.sh

    echo ">>> STEP 4 complete."
}

# =========================================================
# STEP 5: Monitor & Evaluate
# =========================================================
step5_evaluate() {
    local OUT="${ATLAS_BASE}/data/outputs/grpo_baseline_4gpu_a6000"
    echo ""
    echo ">>> STEP 5: Evaluate Results"
    echo ""
    echo "--- Reward curve ---"
    echo "The moving average reward is logged to:"
    echo "  ${OUT}/reward.txt"
    echo "Plot it to compare against cfgrl-expo's HPSv2 scores."
    echo ""
    echo "--- WandB ---"
    echo "If WANDB_MODE=online, check your wandb dashboard (project: flux)"
    echo ""
    echo "--- Checkpoints ---"
    echo "LoRA checkpoints saved at:"
    echo "  ${OUT}/lora-checkpoint-{step}-{epoch}/"
    echo ""
    echo "--- Generated images ---"
    echo "Training sample images saved to:"
    echo "  ${OUT}/images/"
    echo ""
    echo "--- Visualization ---"
    echo "For full checkpoint visualization:"
    echo "  1. rm -rf ${ATLAS_BASE}/data/flux/transformer/*"
    echo "  2. Copy trained LoRA checkpoint into ${ATLAS_BASE}/data/flux/transformer/"
    echo "  3. Run scripts/visualization/vis_flux.py"
    echo ""
    echo "--- Comparison with cfgrl-expo ---"
    echo "Key metrics to compare:"
    echo "  - HPSv2 reward: DanceGRPO reward.txt vs cfgrl-expo eval logs"
    echo "  - Training compute: wall-clock time and GPU-hours"
    echo "  - Reward curve shape: convergence speed"
}

# =========================================================
# Main
# =========================================================
echo ""
echo "Available steps:"
echo "  1) step1_environment   — Install dependencies"
echo "  2) step2_checkpoints   — Download model weights"
echo "  3) step3_preprocess    — Preprocess text embeddings"
echo "  4) step4_train         — Launch GRPO training"
echo "  5) step5_evaluate      — Evaluate and compare"
echo ""
echo "Run individual steps by sourcing this file and calling the function:"
echo "  source scripts/setup_baseline_run.sh"
echo "  step1_environment"
echo ""
echo "Or run everything sequentially (after downloading checkpoints):"
echo "  source scripts/setup_baseline_run.sh"
echo "  step1_environment && step2_checkpoints && step3_preprocess && step4_train"
