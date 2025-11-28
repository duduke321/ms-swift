#!/bin/bash
# ============================================================================
# GKD LoRA 自蒸馏训练脚本
# 模型: Qwen3-Omni-30B-A3B-Instruct (30B MoE 多模态)
# 训练模式: 共享基础模型 + LoRA 自蒸馏
#   - 教师: 基础模型 (LoRA 禁用)
#   - 学生: 基础模型 + LoRA (可训练)
# ============================================================================
#
# 显存占用估算 (8x GPU):
#   - 原版 GKD (两个独立模型): ~60-70 GiB per GPU
#   - LoRA 自蒸馏 (当前): ~35-40 GiB per GPU
#   - 节省: ~40-50%
#
# ============================================================================

# ===== 环境变量配置 =====
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ===== 分布式训练配置 =====
NPROC_PER_NODE=8 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift rlhf \
    --rlhf_type gkd \
    --model /home/work_nfs19/asr_data/ckpt/Qwen3-Omni-30B-A3B-Instruct \
    --teacher_model /home/work_nfs19/asr_data/ckpt/Qwen3-Omni-30B-A3B-Instruct \
    --train_type lora \
    --torch_dtype bfloat16 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --target_modules all-linear \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#2000' \
    --split_dataset_ratio 0.01 \
    --max_length 2048 \
    --max_completion_length 512 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --seq_kd false \
    --lmbda 0.5 \
    --temperature 2.0 \
    --beta 0.5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --warmup_ratio 0.05 \
    --deepspeed zero3 \
    --attn_impl flash_attention_2 \
    --output_dir output \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --save_only_model true

# ============================================================================
# 参数说明:
#
# GKD 参数:
#   --lmbda 0.5         : 50% 概率使用学生模型生成的数据训练 (on-policy)
#   --temperature 2.0   : 蒸馏温度，软化分布 (1.0-4.0)
#   --beta 0.5          : JSD 损失中的权重参数
#   --seq_kd false      : Token-level 蒸馏 (不是序列级)
#
# LoRA 参数:
#   --lora_rank 8       : LoRA 秩，越大模型容量越大但参数也越多
#   --lora_alpha 16     : LoRA alpha，通常是 rank 的 2倍
#   --lora_dropout 0.05 : LoRA dropout 率
#   --lora_target_modules ALL : 应用 LoRA 到所有线性层
#
# DeepSpeed:
#   --deepspeed zero3   : 学生模型使用 ZeRO-3 优化
#   (teacher_deepspeed 参数会被自动忽略，因为不需要独立教师模型)
#
# vLLM (不支持):
#   Qwen3-Omni 多模态模型暂不支持 vLLM 加速
#   使用传统生成方式
#
# ============================================================================
# 运行建议:
#
# 1. 首次运行建议减小 batch size 观察显存占用:
#    --per_device_train_batch_size 2
#
# 2. 如果显存充足，可以增大 batch size:
#    --per_device_train_batch_size 6  # 或 8
#
# 3. 使用梯度累积保持有效 batch size:
#    --per_device_train_batch_size 2
#    --gradient_accumulation_steps 2
#
# 4. 调整 LoRA rank 权衡性能和参数量:
#    较小 rank (4-8): 更快，参数少，可能性能略低
#    较大 rank (16-64): 更慢，参数多，可能性能更好
#
# ============================================================================
