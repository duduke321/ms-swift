# 2 * 60GiB
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MAX_PIXELS=1003520 \
NPROC_PER_NODE=8 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
swift sft \
    --model /home/work_nfs19/sywang/ckpt/Qwen3-Omni-30B-A3B-Instruct \
    --dataset /home/work_nfs19/sywang/code/ms-swift/data/output.jsonl \
    --split_dataset_ratio 0.01 \
    --load_from_cache_file true \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --attn_impl flash_attention_2 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --padding_free true \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing true \
    --eval_steps 1000 \
    --save_steps 1000 \
    --save_total_limit 100 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataset_num_proc 8 \
    --deepspeed zero3 \
    --dataloader_num_workers 8
