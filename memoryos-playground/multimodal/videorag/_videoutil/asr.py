import os
import re
import torch
import logging
from tqdm import tqdm
from faster_whisper import WhisperModel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

def speech_to_text(video_name, working_dir, segment_index2name, audio_output_format):
    model = WhisperModel("/root/models/faster-distil-whisper-large-v3")
    model.logger.setLevel(logging.WARNING)
    
    cache_path = os.path.join(working_dir, '_cache', video_name)
    
    transcripts = {}
    languages = {}
    
    # 第一步：先检测第一个有效片段的语言，以便后续明确指定
    detected_video_language = None
    for index in segment_index2name:
        segment_name = segment_index2name[index]
        audio_file = os.path.join(cache_path, f"{segment_name}.{audio_output_format}")
        if os.path.exists(audio_file):
            # 快速检测语言（只检测，不完整转录）
            _, info = model.transcribe(audio_file, task="transcribe", language=None, beam_size=1, vad_filter=False)
            detected_video_language = getattr(info, 'language', None) if info else None
            if detected_video_language:
                break
    
    # 第二步：使用检测到的语言明确指定参数进行转录
    for index in tqdm(segment_index2name, desc=f"Speech Recognition {video_name}"):
        segment_name = segment_index2name[index]
        audio_file = os.path.join(cache_path, f"{segment_name}.{audio_output_format}")

        # if the audio file does not exist, skip it
        if not os.path.exists(audio_file):
            transcripts[index] = ""
            languages[index] = "unknown"
            continue
        
        # 明确设置 task="transcribe" 确保只转录不翻译，language=None 让模型自动检测语言
        segments, info = model.transcribe(
            audio_file,
            task="transcribe",  # 只转录，不翻译
            language=None,  # 自动检测语言
            condition_on_previous_text=False,  # 避免上下文影响
        )
        result = ""
        for segment in segments:
            result += "[%.2fs -> %.2fs] %s\n" % (segment.start, segment.end, segment.text)
        transcripts[index] = result
        
        # Extract language from Whisper info object
        # faster-whisper returns info as a named tuple with language attribute
        detected_lang = getattr(info, 'language', None) if info else None
        
        # 双重验证：检查 transcript 文本内容中的中文字符
        has_chinese = False
        if result.strip():
            # 移除时间戳标记后检查
            clean_text = re.sub(r'\[\d+\.\d+s\s*->\s*\d+\.\d+s\]', '', result)
            has_chinese = bool(re.search(r'[\u4e00-\u9fff]', clean_text))
        
        if detected_lang:
            # Map Whisper language codes to our format
            # Whisper uses ISO 639-1 codes (e.g., 'en', 'zh', 'ja', etc.)
            if detected_lang == 'zh' or detected_lang.startswith('zh'):
                languages[index] = "zh"
            elif detected_lang == 'en':
                # 如果 Whisper 识别为英文，但文本中包含中文字符，则改为中文
                languages[index] = "zh" if has_chinese else "en"
            else:
                # 对于其他语言，如果包含中文，也认为是中文
                languages[index] = "zh" if has_chinese else detected_lang
        else:
            # If no language detected, check text content
            if not result.strip():
                languages[index] = "unknown"
            elif has_chinese:
                languages[index] = "zh"
            else:
                languages[index] = "en"
    
    return transcripts, languages