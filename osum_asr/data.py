import json
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
import threading


def validate_audio_fast(audio_path):
    """快速验证音频文件（只检查文件是否存在和大小）"""
    try:
        # 只检查文件是否存在且大小大于0
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            # 检查文件扩展名（快速过滤）
            ext = os.path.splitext(audio_path)[1].lower()
            if ext in ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac']:
                return True
        return False
    except Exception:
        return False


def process_line(args):
    """处理单行数据（用于并行处理）"""
    line, text_dict = args
    line = line.strip()
    if not line:
        return None
    
    parts = line.split(None, 1)
    if len(parts) != 2:
        return None
    
    utt_id, audio_path = parts
    
    if utt_id not in text_dict:
        return None
    
    # 处理可能的管道命令
    if '|' in audio_path:
        audio_path = audio_path.split('|')[-1].strip()
    
    # 转为绝对路径
    if not os.path.isabs(audio_path):
        audio_path = os.path.abspath(audio_path)
    
    # 快速验证（只检查文件存在性，不加载音频）
    if not validate_audio_fast(audio_path):
        return None
    
    transcription = text_dict[utt_id]
    return (utt_id, audio_path, transcription)


def convert_to_jsonl(wav_scp_path, text_path, output_jsonl_path, user_prompt="<audio>", max_lines=100000, num_workers=None):
    """
    将 wav.scp 和 text 文件转换为 JSONL 格式
    高效读取，只保留最多 max_lines 条匹配的数据
    
    Args:
        num_workers: 并行处理的线程数，None 表示使用 CPU 核心数
    """
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)  # 最多8个线程，避免过多I/O竞争
    
    # Step 1: 先读取 text 文件
    print(f"正在读取 text 文件: {text_path}")
    text_dict = {}
    with open(text_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="读取 text"):
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                utt_id, transcription = parts
                text_dict[utt_id] = transcription
    print(f"从 text 读取了 {len(text_dict)} 条数据")

    # Step 2: 流式并行处理 wav.scp（边读边处理，达到目标数量就停止）
    print(f"正在并行读取 wav.scp 并匹配（使用 {num_workers} 个线程）...")
    matched_data = []
    stop_flag = threading.Event()
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        pending_count = 0
        
        # 流式读取和处理
        with open(wav_scp_path, 'r', encoding='utf-8') as f:
            pbar = tqdm(desc="处理 wav.scp", unit="行")
            
            for line in f:
                # 如果已经达到目标数量，停止提交新任务
                if stop_flag.is_set() or len(matched_data) >= max_lines:
                    break
                
                # 提交任务
                future = executor.submit(process_line, (line, text_dict))
                futures.append(future)
                pending_count += 1
                
                # 当待处理任务达到一定数量时，开始收集结果
                if pending_count >= num_workers * 2:
                    # 收集已完成的任务
                    completed = []
                    for future in list(futures):
                        if future.done():
                            completed.append(future)
                            futures.remove(future)
                            pending_count -= 1
                    
                    # 处理完成的任务
                    for future in completed:
                        try:
                            result = future.result()
                            if result is not None:
                                matched_data.append(result)
                                if len(matched_data) >= max_lines:
                                    stop_flag.set()
                                    break
                        except Exception as e:
                            pass  # 忽略错误
                    
                    pbar.update(len(completed))
            
            # 处理剩余的任务
            if not stop_flag.is_set():
                for future in tqdm(as_completed(futures), total=len(futures), desc="完成剩余任务"):
                    if stop_flag.is_set() or len(matched_data) >= max_lines:
                        break
                    try:
                        result = future.result()
                        if result is not None:
                            matched_data.append(result)
                            if len(matched_data) >= max_lines:
                                stop_flag.set()
                                break
                    except Exception:
                        pass
            
            pbar.close()
            
            # 取消未完成的任务
            for future in futures:
                future.cancel()

    print(f"成功匹配 {len(matched_data)} 条数据")

    # Step 3: 批量写入 JSONL
    os.makedirs(os.path.dirname(output_jsonl_path) if os.path.dirname(output_jsonl_path) else '.', exist_ok=True)
    print(f"正在写入 JSONL 文件...")
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for _, audio_path, transcription in tqdm(matched_data, desc="写入 JSONL"):
            data = {
                "messages": [
                    {"role": "system", "content": "You are a speech recognition assistant. First, confirm the output language by starting your response with the exact language name (e.g., Chinese, English, Japanese). Then, listen to the provided audio content carefully, transcribe every word accurately into text in the confirmed language. Ensure no words are missing or added. Output only the language name followed by the transcribed text."},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": f"Chinese. {transcription}"}
                ],
                "audios": [audio_path]
            }
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"成功生成 {len(matched_data)} 条数据到 {output_jsonl_path}")


if __name__ == "__main__":
    wav_scp_path = "/home/work_nfs15/asr_data/data/wenetspeech/train/wav.scp"
    text_path = "/home/work_nfs15/asr_data/data/wenetspeech/train/text"
    output_jsonl_path = "../data/output.jsonl"

    convert_to_jsonl(wav_scp_path, text_path, output_jsonl_path, max_lines=100000)