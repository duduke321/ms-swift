# 4 * 73GiB, 11s/it

export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=4,5,6,7

MASTER_PORT=29600 \
NPROC_PER_NODE=4 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift rlhf \
    --rlhf_type gkd \
    --model /home/work_nfs11/asr_data/ckpt/Qwen2.5-Omni-3B \
    --teacher_model /home/work_nfs11/asr_data/ckpt/Qwen2.5-Omni-3B \
    --train_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#2000' \
    --split_dataset_ratio 0.01 \
    --seq_kd false \
    --lmbda 0.5 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --max_completion_length 512 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --save_only_model true \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --deepspeed zero3 \
    --attn_impl flash_attention_2 \
    --teacher_deepspeed zero3_offload
#    --use_vllm true \
#    --vllm_mode colocate \
#    --vllm_gpu_memory_utilization 0.3 \
#    --vllm_tensor_parallel_size 2 \
#    --sleep_level 1
