#! /bin/bash

test_sets=(test_meeting test_net)
for test_set in ${test_sets[@]}; do
    wav_scp=/home/work_nfs15/asr_data/data/asr_test_sets/${test_set}/wav.scp
    text=/home/work_nfs15/asr_data/data/asr_test_sets/${test_set}/text
    output_text=./result/${test_set}/infer_2.txt
    wer_result=./result/${test_set}/wer.txt

    python /home/work_nfs11/code/sywang/workspace/toolkit/duduke/wenet_tools/compute-wer.py --char=1 \
        $text \
        $output_text > $wer_result
done

