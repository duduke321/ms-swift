# GKD LoRA 自蒸馏 - 快速参考

## 🎯 核心概念

```
同一个基础模型 (Qwen3-Omni-30B)
         │
    ┌────┴────┐
    │         │
教师模式    学生模式
(LoRA禁用) (LoRA启用)
    │         │
冻结预测   可训练预测
    └────┬────┘
      JSD损失
```

## ✅ 当前配置确认

您的 `gkd.sh` **已经正确配置**了 LoRA 自蒸馏！

关键配置：
```bash
--model /path/to/Qwen3-Omni-30B-A3B-Instruct      # 基础模型
--teacher_model /path/to/Qwen3-Omni-30B-A3B-Instruct  # 同一个！
--train_type lora                                  # 启用 LoRA
```

## 🚀 快速开始

```bash
# 直接运行
cd /Users/duduke/code/ms-swift/osum_v2
bash gkd.sh

# 或使用优化版
bash gkd_optimized.sh
```

## 📊 参数速查表

| 参数 | 默认值 | 建议范围 | 说明 |
|------|--------|----------|------|
| **GKD 参数** |
| `lmbda` | 0.5 | 0.3-0.7 | On-policy 采样概率 |
| `temperature` | 2.0 | 1.0-4.0 | 蒸馏温度（软化分布） |
| `beta` | 0.5 | 0.3-0.7 | JSD 损失权重 |
| `seq_kd` | false | true/false | 序列级 vs Token级 KD |
| **LoRA 参数** |
| `lora_rank` | 8 | 4-64 | LoRA 秩（容量） |
| `lora_alpha` | 16 | 8-128 | LoRA alpha (通常=2×rank) |
| `lora_dropout` | 0.05 | 0.0-0.1 | Dropout 率 |
| **训练参数** |
| `batch_size` | 4 | 1-8 | 每卡 batch size |
| `learning_rate` | 1e-5 | 5e-6 ~ 5e-5 | 学习率 |
| `grad_accum` | 1 | 1-8 | 梯度累积步数 |

## 🎛️ 常用调优场景

### 场景 1: 显存不足
```bash
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 2 \  # 保持有效 batch=4
--lora_rank 4 \                     # 减小 LoRA rank
```

### 场景 2: 追求性能
```bash
--per_device_train_batch_size 4 \
--lora_rank 16 \                    # 增大 LoRA rank
--lora_alpha 32 \
--learning_rate 5e-6 \              # 降低学习率
```

### 场景 3: 快速实验
```bash
--per_device_train_batch_size 8 \
--lora_rank 4 \
--dataset 'your_dataset#500' \      # 减少数据量
--num_train_epochs 0.5 \
```

## 🔍 运行检查

训练开始时，日志应该显示：

```
✓ Using shared base model architecture:
  - Student model: base_model + LoRA (trainable)
  - Teacher model: base_model with LoRA disabled (frozen)
```

如果看到错误：
```
✗ ValueError: GKDTrainer requires the model to have LoRA adapters.
```
检查 `--train_type lora` 是否存在。

## 💾 显存占用估算

### Qwen3-Omni-30B-A3B (8x GPU, ZeRO-3)

| 配置 | 每卡显存 | 总显存 | 说明 |
|------|---------|--------|------|
| **原版 GKD** (两个独立模型) |
| batch=4, rank=8 | ~60 GB | ~480 GB | 需要模型并行 |
| **LoRA 自蒸馏** (共享模型) |
| batch=4, rank=8 | ~36 GB | ~288 GB | ✅ 推荐 |
| batch=6, rank=8 | ~45 GB | ~360 GB | 显存充足时 |
| batch=2, rank=8 | ~28 GB | ~224 GB | 显存紧张时 |
| batch=4, rank=16 | ~38 GB | ~304 GB | 更大容量 |

**节省**: ~40-50% 显存

## 🔧 故障排查

### 问题 1: OOM (显存不足)

**症状**: `RuntimeError: CUDA out of memory`

**解决方案**:
```bash
# 方案 A: 减小 batch size
--per_device_train_batch_size 2

# 方案 B: 减小 LoRA rank
--lora_rank 4

# 方案 C: 减小序列长度
--max_length 1024
```

### 问题 2: 训练不稳定/Loss NaN

**症状**: Loss 突然变成 NaN 或剧烈震荡

**解决方案**:
```bash
# 降低学习率
--learning_rate 5e-6  # 或 1e-6

# 降低 lambda (减少 on-policy)
--lmbda 0.3

# 增加 warmup
--warmup_ratio 0.1
```

### 问题 3: 训练速度慢

**症状**: it/s 很低

**解决方案**:
```bash
# 增加 dataloader workers
--dataloader_num_workers 8

# 减小 batch size，增加梯度累积
--per_device_train_batch_size 2
--gradient_accumulation_steps 2
```

## 📈 监控命令

```bash
# 监控 GPU 使用
watch -n 1 nvidia-smi

# 监控训练日志
tail -f output/runs/xxx/logs/xxx.log

# 查看 TensorBoard
tensorboard --logdir output/runs
```

## 🎨 高级技巧

### 技巧 1: 动态调整 Lambda

```python
# 可以在训练过程中逐步增加 on-policy 比例
# 初期: lmbda=0.3 (更稳定)
# 后期: lmbda=0.7 (更多探索)
```

### 技巧 2: 多阶段训练

```bash
# 阶段 1: 小 rank 快速收敛
bash gkd_stage1.sh  # rank=4, epochs=1

# 阶段 2: 大 rank 精细调优
# 从阶段 1 checkpoint 继续
bash gkd_stage2.sh  # rank=16, epochs=2
```

### 技巧 3: 混合精度优化

```bash
# BF16 (推荐，Qwen3 原生支持)
--torch_dtype bfloat16

# FP16 (如果 GPU 不支持 BF16)
--torch_dtype float16
--fp16 true
```

## 📝 实验记录模板

```markdown
## 实验 #1
- 日期: 2025-11-28
- 配置: lmbda=0.5, rank=8, batch=4
- 数据: alpaca-gpt4-zh (2000 samples)
- 显存: 36 GB/GPU
- 速度: 11s/it
- 结果: Loss下降平稳，无 OOM
- 备注: 基线配置，运行正常
```

## 🔗 相关文件

- `gkd.sh` - 您的原始脚本
- `gkd_optimized.sh` - 优化版脚本（添加了详细注释）
- `README.md` - 完整文档
- `swift/trainers/rlhf_trainer/gkd_trainer.py` - 修改后的训练器

## 💡 核心要点

1. ✅ **配置已正确**: 您的 gkd.sh 已经是 LoRA 自蒸馏模式
2. ✅ **无需修改**: `--teacher_model` 指向同一模型即可
3. ✅ **自动检测**: trainer 会自动使用共享基础模型
4. ✅ **显存友好**: 节省 40-50% 显存
5. ✅ **训练加速**: LoRA 梯度计算更快

## ⚡ 一键启动

```bash
cd /Users/duduke/code/ms-swift/osum_v2 && bash gkd.sh
```

训练愉快！🚀
