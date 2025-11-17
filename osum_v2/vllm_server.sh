# CUDA_VISIBLE_DEVICES=5 \
# swift rollout \
#     --model Qwen/Qwen2.5-7B \
#     --vllm_max_model_len 2560

# vllm serve /home/work_nfs19/sywang/ckpt/Qwen3-Omni-30B-A3B-Instruct --port 8000 --host 127.0.0.1 --dtype bfloat16 --max-model-len 65536 --allowed-local-media-path / -tp 8

# 4 * 54GiB
# 5s/it (with vLLM)
# 14s/it (without vLLM)
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=4,5,6,7
NPROC_PER_NODE=4 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift rlhf \
    --rlhf_type gkd \
    --model /home/work_nfs19/sywang/ckpt/Qwen3-Omni-30B-A3B-Instruct \
    --teacher_model /home/work_nfs19/sywang/ckpt/Qwen3-Omni-30B-A3B-Instruct \
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
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000
