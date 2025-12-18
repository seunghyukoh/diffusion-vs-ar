#!/bin/bash

set -euo pipefail

MODEL_SIZE="llada_tiny"
NUM_GPUS=8
BATCH_SIZE_PER_GPU=128
GRAD_ACCUM_STEPS=1
TOTAL_BATCH_SIZE=$((BATCH_SIZE_PER_GPU * GRAD_ACCUM_STEPS))

CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}

echo "WANDB_DISABLED: ${WANDB_DISABLED}"
export WANDB_DISABLED=true


echo "=== Sudoku Experiment ==="
echo "Project root: $PROJECT_ROOT"
echo "Working directory: $REPO_DIR"
echo "Model: model_config_${MODEL_SIZE}"
echo "GPUs: ${NUM_GPUS}"
echo "Batch size per GPU: ${BATCH_SIZE_PER_GPU}"
echo "Gradient accumulation steps: ${GRAD_ACCUM_STEPS}"
echo "Total batch size: ${TOTAL_BATCH_SIZE}"
echo ""

uv sync

echo "=== Training MDM ==="

exp=output/sudoku/mdm-${MODEL_SIZE}-alpha0.25-gamma1-bs${TOTAL_BATCH_SIZE}-lr1e-3-ep300-T20-`date "+%Y%m%d-%H%M%S"`
mkdir -p $exp

uv run \
    accelerate launch --multi_gpu --num_machines 1 --mixed_precision fp16 --num_processes 8 --main_process_port 20099 \
    src/train_bash.py \
        --stage mdm --overwrite_output_dir \
        --cache_dir ./cache \
        --model_name_or_path model_config_${MODEL_SIZE} \
        --do_train \
        --dataset sudoku_train \
        --finetuning_type full \
        --cutoff_len 164 \
        --output_dir $exp \
        --overwrite_cache \
        --per_device_train_batch_size ${BATCH_SIZE} \
        --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --val_size 448 \
        --per_device_eval_batch_size 32 \
        --evaluation_strategy steps \
        --eval_steps 100 \
        --save_steps 500 \
        --learning_rate 1e-3 \
        --num_train_epochs 300.0 \
        --plot_loss \
        --run_name ${dataset}_prefix \
        --preprocessing_num_workers 8 \
        --fp16 \
        --save_total_limit 1 \
        --remove_unused_columns False \
        --diffusion_steps 20 \
        --save_safetensors False \
        --token_reweighting True \
        --time_reweighting linear \
        --topk_decoding True \
        --alpha 0.25 \
        --gamma 1 \
        > $exp/train.log


echo ""
echo "=== Evaluating MDM Model ==="

for dataset in sudoku_test
do
topk_decoding=True
mkdir $exp/$dataset
CUDA_VISIBLE_DEVICES=1  \
python3 -u src/train_bash.py \
    --stage mdm --overwrite_output_dir \
    --cache_dir ./cache \
    --model_name_or_path model_config_${MODEL_SIZE} \
    --do_predict \
    --cutoff_len 164 \
    --dataset $dataset \
    --finetuning_type full \
    --diffusion_steps 20 \
    --output_dir $exp/${dataset} \
    --checkpoint_dir $exp  \
    --remove_unused_columns False \
    --decoding_strategy stochastic0.5-linear \
    --topk_decoding $topk_decoding \
    > $exp/${dataset}/eval-TopK$topk_decoding.log
done
