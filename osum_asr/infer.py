import os
import torch
import argparse
from tqdm import tqdm

from vllm import LLM, SamplingParams
from transformers import Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

def read_wav_scp(wav_scp_path):
    """读取 wav.scp 文件，返回 utt_id 和 audio_path 的列表"""
    data = []
    with open(wav_scp_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            utt_id, audio_path = parts
            
            # 处理可能的管道命令（如 "utt_id | sox ... |"）
            if '|' in audio_path:
                # 提取管道命令后的最后一个路径
                audio_path = audio_path.split('|')[-1].strip()
            
            # 转为绝对路径
            if not os.path.isabs(audio_path):
                audio_path = os.path.abspath(audio_path)
            
            # 检查文件是否存在
            if os.path.exists(audio_path):
                data.append((utt_id, audio_path))
            else:
                print(f"警告: 音频文件不存在: {audio_path}")
    
    return data

def asr_inference(llm, processor, audio_path, sampling_params):
    """对单个音频文件进行 ASR 识别"""
    messages = [
        {
            "role": "system",
             "content": [
                {"type": "text", "text": "You are a speech recognition assistant. First, confirm the output language by starting your response with the exact language name (e.g., Chinese, English, Japanese). Then, listen to the provided audio content carefully, transcribe every word accurately into text in the confirmed language. Ensure no words are missing or added. Output only the language name followed by the transcribed text."},
             ]
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                # {"type": "text", "text": "请将这段中文语音转换为纯文本。"},
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)

    inputs = {
        'prompt': text,
        'multi_modal_data': {},
        "mm_processor_kwargs": {
            "use_audio_in_video": False,
        },
    }

    if audios is not None:
        inputs['multi_modal_data']['audio'] = audios

    outputs = llm.generate([inputs], sampling_params=sampling_params)
    
    return outputs[0].outputs[0].text.strip()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASR inference using Qwen3-Omni model')
    parser.add_argument('--wav_scp', type=str, required=True, help='输入 wav.scp 文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出识别结果文件路径')
    parser.add_argument('--model_path', type=str, 
                       default="/home/work_nfs19/sywang/ckpt/Qwen3-Omni-30B-A3B-Instruct",
                       help='模型路径')
    parser.add_argument('--batch_size', type=int, default=1, 
                       help='批处理大小（当前版本建议设为1）')
    
    args = parser.parse_args()
    
    # vLLM engine v1 not supported yet
    os.environ['VLLM_USE_V1'] = '0'
    MODEL_PATH = args.model_path

    print("正在加载模型...")
    llm = LLM(
            model=MODEL_PATH, trust_remote_code=True, gpu_memory_utilization=0.95,
            tensor_parallel_size=torch.cuda.device_count(),
            limit_mm_per_prompt={'image': 3, 'video': 3, 'audio': 3},
            max_num_seqs=8,
            max_model_len=32768,
            seed=1234,
    )

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=16384,
    )

    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)

    # 读取 wav.scp 文件
    print(f"正在读取 wav.scp 文件: {args.wav_scp}")
    audio_list = read_wav_scp(args.wav_scp)
    print(f"共找到 {len(audio_list)} 个音频文件")

    # 进行 ASR 识别并写入结果
    print("开始进行 ASR 识别...")
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    
    with open(args.output, 'w', encoding='utf-8') as f_out:
        for utt_id, audio_path in tqdm(audio_list, desc="ASR 识别"):
            try:
                result_text = asr_inference(llm, processor, audio_path, sampling_params)
                # 写入结果文件，格式：utt_id 识别文本
                f_out.write(f"{utt_id} {result_text}\n")
                f_out.flush()  # 实时写入，避免丢失
            except Exception as e:
                print(f"错误: 处理 {utt_id} ({audio_path}) 时出错: {e}")
                # 即使出错也写入空结果，保持文件格式一致
                f_out.write(f"{utt_id} \n")
                f_out.flush()
    
    print(f"识别完成！结果已保存到: {args.output}")