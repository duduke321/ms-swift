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


def build_input(processor, messages, use_audio_in_video):
    """根据官方代码构建输入"""
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)

    inputs = {
        'prompt': text,
        'multi_modal_data': {},
        "mm_processor_kwargs": {
            "use_audio_in_video": use_audio_in_video,
        },
    }

    if images is not None:
        inputs['multi_modal_data']['image'] = images
    if videos is not None:
        inputs['multi_modal_data']['video'] = videos
    if audios is not None:
        inputs['multi_modal_data']['audio'] = audios
    
    return inputs


def load_model(model_path):
    """加载 LLM 模型"""
    # vLLM engine v1 not supported yet
    os.environ['VLLM_USE_V1'] = '0'
    
    print("正在加载模型...")
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        tensor_parallel_size=torch.cuda.device_count(),
        limit_mm_per_prompt={'image': 3, 'video': 3, 'audio': 3},
        max_num_seqs=8,
        max_model_len=32768,
        seed=1234,
    )
    return llm


def create_sampling_params():
    """创建采样参数"""
    return SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=16384,
    )


def load_processor(model_path):
    """加载处理器"""
    return Qwen3OmniMoeProcessor.from_pretrained(model_path)


def create_asr_messages(audio_path):
    """创建 ASR 识别的消息格式"""
    return [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": "请将这段中文语音转换为纯文本。"},
            ]
        }
    ]


def batch_inference(llm, processor, audio_list, sampling_params, batch_size, use_audio_in_video=False):
    """批量进行 ASR 识别"""
    results = []
    
    for batch_start in tqdm(range(0, len(audio_list), batch_size), desc="ASR 识别批次"):
        batch_end = min(batch_start + batch_size, len(audio_list))
        batch_data = audio_list[batch_start:batch_end]
        
        # 构建批量输入
        conversations = []
        for utt_id, audio_path in batch_data:
            messages = create_asr_messages(audio_path)
            conversations.append(messages)
        
        # 批量构建输入
        inputs = [build_input(processor, messages, use_audio_in_video) for messages in conversations]
        
        try:
            # 批量生成
            outputs = llm.generate(inputs, sampling_params=sampling_params)
            
            # 处理批量结果
            for i, (utt_id, audio_path) in enumerate(batch_data):
                try:
                    result_text = outputs[i].outputs[0].text.strip()
                    results.append((utt_id, result_text))
                except Exception as e:
                    print(f"错误: 处理结果 {utt_id} 时出错: {e}")
                    results.append((utt_id, ""))
        except Exception as e:
            print(f"错误: 批量处理批次 {batch_start}-{batch_end} 时出错: {e}")
            # 即使出错也写入空结果，保持文件格式一致
            for utt_id, audio_path in batch_data:
                results.append((utt_id, ""))
    
    return results


def write_results(results, output_path):
    """将结果写入文件"""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for utt_id, result_text in results:
            f_out.write(f"{utt_id} {result_text}\n")
            f_out.flush()  # 实时写入，避免丢失


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='ASR inference using Qwen3-Omni model')
    parser.add_argument('--wav_scp', type=str, required=True, help='输入 wav.scp 文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出识别结果文件路径')
    parser.add_argument('--model_path', type=str, 
                       default="/home/work_nfs19/sywang/ckpt/Qwen3-Omni-30B-A3B-Instruct",
                       help='模型路径')
    parser.add_argument('--batch_size', type=int, default=8, 
                       help='批处理大小')
    
    args = parser.parse_args()
    
    # 加载模型和处理器
    llm = load_model(args.model_path)
    sampling_params = create_sampling_params()
    processor = load_processor(args.model_path)
    
    # 读取 wav.scp 文件
    print(f"正在读取 wav.scp 文件: {args.wav_scp}")
    audio_list = read_wav_scp(args.wav_scp)
    print(f"共找到 {len(audio_list)} 个音频文件")
    
    # 进行批量 ASR 识别
    print("开始进行 ASR 识别...")
    results = batch_inference(
        llm=llm,
        processor=processor,
        audio_list=audio_list,
        sampling_params=sampling_params,
        batch_size=args.batch_size,
        use_audio_in_video=False
    )
    
    # 写入结果
    write_results(results, args.output)
    print(f"识别完成！结果已保存到: {args.output}")


if __name__ == '__main__':
    main()