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
#   - LoRA 自蒸馏 (优化后): ~30-35 GiB per GPU
#   - 标准 SFT (对比参考): ~20-25 GiB per GPU
#   
#   ⚠️  GKD 固有开销: 即使共享基础模型，仍需同时保存:
#       - 学生前向传播的激活值 (需要梯度)
#       - 教师前向传播的激活值 (no_grad但需要用于loss计算)
#       这导致显存占用约为标准SFT的 1.4-1.5x
#
# ============================================================================

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
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --seq_kd false \
    --lmbda 0.5 \
    --temperature 2.0 \
    --beta 0.5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing true \
    --freeze_vit true \
    --freeze_aligner true \
    --padding_free true \
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
# 显存优化参数:
#   --gradient_checkpointing true : 梯度检查点，节省 30-50% 显存（牺牲约 20% 速度）
#   --freeze_vit true            : 冻结视觉编码器，减少显存和计算
#   --freeze_aligner true        : 冻结多模态对齐器，减少显存和计算
#   --padding_free true          : 减少 padding 带来的显存浪费
#
# vLLM (不支持):
#   Qwen3-Omni 多模态模型暂不支持 vLLM 加速
#   使用传统生成方式
#
# ============================================================================
# 运行建议:
#
# ⚠️ 如果显存仍然不足，尝试以下优化（按优先级排序）:
#
# 1. **减小序列长度** (最有效):
#    --max_length 768          # 从 1024 降到 768
#    --max_completion_length 192
#
# 2. **进一步减小 LoRA rank**:
#    --lora_rank 2             # 从 4 降到 2 (最小推荐值)
#    --lora_alpha 4
#
# 3. **减少 LoRA 应用的层数**:
#    --target_modules q_proj k_proj v_proj o_proj  # 仅 attention 层
#
# 4. **如果使用 ZeRO-2 而非 ZeRO-3**:
#    --deepspeed zero2         # ZeRO-2 有时更稳定
#
# 5. **减小数据加载并行度**:
#    --dataloader_num_workers 1
#    --dataset_num_proc 1
#
# 6. **考虑使用 CPU offload** (显著降低速度):
#    需要自定义 DeepSpeed config 启用 offload_optimizer/offload_param
#
# 如果以上都无效，GKD 算法本身的双前向传播特性可能与您的硬件不匹配，
# 建议考虑:
#   - 使用标准 SFT (swift sft)
#   - 使用更少的 GPU 但增加梯度累积
#   - 等待更大显存的硬件
#
# ============================================================================

