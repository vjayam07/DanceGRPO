GPU_NUM=8 # 2,4,8
MODEL_PATH="data/flux"
OUTPUT_DIR="data/rl_embeddings"
PROMPT_PATH="${CFGRL_EXPO_DIR:-../cfgrl-expo}/expo/envs/flux_prompts_hpdv2.json"

torchrun --nproc_per_node=$GPU_NUM --master_port 19002 \
    fastvideo/data_preprocess/preprocess_flux_embedding.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --prompt_dir "$PROMPT_PATH" \
    --num_train_prompts 75000 \
    --num_eval_prompts 500 \
    --prompt_seed 42 \
    --train_test_split 0.8
