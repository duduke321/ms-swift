#! /bin/bash

test_sets=(test_meeting test_net)
for test_set in ${test_sets[@]}; do
    wav_scp=/home/work_nfs15/asr_data/data/asr_test_sets/${test_set}/wav.scp
    text=/home/work_nfs15/asr_data/data/asr_test_sets/${test_set}/text
    output_text=./result/${test_set}/infer.txt
    wer_result=./result/${test_set}/wer.txt

    python infer.py \
        --wav_scp $wav_scp \
        --output $output_text \
        --model_path /home/work_nfs19/asr_data/ckpt/Qwen3-Omni-30B-A3B-Instruct/

    python /home/work_nfs11/code/sywang/workspace/toolkit/duduke/wenet_tools/compute-wer.py --char=1 \
        $text \
        $output_text > $wer_result
done

