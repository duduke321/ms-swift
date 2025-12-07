#!/bin/bash


# ===== 环境变量配置 =====
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ===== 分布式训练配置 =====
MAX_PIXELS=1003520 \
NPROC_PER_NODE=8 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift rlhf \
    --rlhf_type gkd \
    --model /home/work_nfs19/asr_data/ckpt/Qwen3-Omni-30B-A3B-Instruct \
    --teacher_model /home/work_nfs19/asr_data/ckpt/Qwen3-Omni-30B-A3B-Instruct \
    --train_type lora \
    --torch_dtype bfloat16 \
    --lora_rank 4 \
    --lora_alpha 8 \
    --lora_dropout 0.05 \
    --target_modules all-linear \
    --dataset /home/work_nfs19/sywang/code/ms-swift/data/output.jsonl \
    --split_dataset_ratio 0.01 \
    --max_length 1024 \
    --max_completion_length 256 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 4 \
    --seq_kd false \
    --lmbda 0 \
    --use_liger_kernel false \
    --sft_alpha 1.0 \
    --temperature 2.0 \
    --beta 0.5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --freeze_vit true \
    --freeze_aligner true \
    --padding_free true \
    --warmup_ratio 0.05 \
    --deepspeed zero3 \
    --attn_impl flash_attention_2 \
    --output_dir output \
    --eval_steps 1000 \
    --save_steps 1000 \
    --save_total_limit 0 \
    --logging_steps 5 \
    --save_only_model true

# ============================================================================

