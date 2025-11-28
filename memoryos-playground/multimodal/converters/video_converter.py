"""Video converter backed by the VideoRAG pipeline."""

from __future__ import annotations

import json
import multiprocessing
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..converter import ConversionChunk, ConversionOutput, MultimodalConverter
from ..factory import ConverterFactory
from ..videorag import QueryParam, VideoRAG
from ..videorag._llm import deepseek_bge_config
from ..videorag._utils import always_get_an_event_loop, logger
from ..videorag._videoutil import (
    merge_segment_information,
    saving_video_segments,
    segment_caption,
    speech_to_text,
    split_video,
)


class VideoConverter(MultimodalConverter):
    """
    利用 VideoRAG 完成真实视频理解。

    convert() 将执行以下步骤：
      1. 调用 VideoRAG.insert_video() 对单个视频做切片、语音识别与多模态描述。
      2. 将每个视频片段的 Caption + Transcript 作为一个 ConversionChunk。
      3. （可选）若提供 question，将直接调用 VideoRAG.query() 生成总结 Chunk。
    """

    SUPPORTED_EXTENSIONS: List[str] = ["mp4", "mov", "avi", "mkv", "webm", "flv"]
    DEFAULT_QUERY: str = "请结合视频所有片段，用中文总结主要人物、事件与场景变化。"

    def convert(
        self,
        source,
        *,
        source_type: str = "file_path",
        **kwargs: Any,
    ) -> ConversionOutput:
        video_path = self._normalize_source(source, source_type)
        self._ensure_spawn_start_method()

        working_dir = Path(
            kwargs.get("working_dir")
            or self.config.get("working_dir")
            or "./videorag-workdir"
        ).expanduser()
        working_dir.mkdir(parents=True, exist_ok=True)

        llm_config = kwargs.get("llm_config") or self.config.get("llm_config") or deepseek_bge_config
        videorag_params = {
            "working_dir": str(working_dir),
            **self.config.get("videorag_params", {}),
            **kwargs.get("videorag_params", {}),
        }
        videorag = VideoRAG(llm=llm_config, **videorag_params)

        self._report_progress(0.05, "准备视频处理流程")
        self._ingest_video_with_progress(videorag, str(video_path))

        self._report_progress(0.78, "汇总 VideoRAG 片段结果")
        video_name = video_path.stem
        segments = videorag.video_segments._data.get(video_name)
        if not segments:
            return ConversionOutput(
                status="failed",
                error=f"未找到视频 {video_name} 的片段数据，可能处理失败。",
                metadata={"video_path": str(video_path)},
            )

        chunks = self._build_segment_chunks(video_name, str(video_path), segments)

        summary_chunk = self._maybe_query_summary(videorag, video_name, len(chunks), kwargs)
        if summary_chunk:
            chunks.append(summary_chunk)

        overall_metadata = {
            "converter_provider": "VideoRAG",
            "converter_version": "1.0.0",
            "video_path": str(video_path),
            "working_dir": str(working_dir),
            "segment_count": len(segments),
            "question": summary_chunk.metadata["question"] if summary_chunk else None,
        }
        text = "\n\n".join(chunk.text for chunk in chunks)
        self._report_progress(1.0, "VideoRAG 处理完成")
        return ConversionOutput(status="success", text=text, chunks=chunks, metadata=overall_metadata)

    def supports(self, *, file_type: str, mime_type: str = None) -> bool:
        return file_type.lower() in self.SUPPORTED_EXTENSIONS

    # --- Internal helpers -------------------------------------------------

    @staticmethod
    def _ensure_spawn_start_method() -> None:
        try:
            current = multiprocessing.get_start_method(allow_none=True)
            if current != "spawn":
                multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            # start method 已设置，忽略
            pass

    def _normalize_source(self, source, source_type: str) -> Path:
        if source_type != "file_path":
            raise ValueError(f"VideoConverter 目前仅支持 file_path，收到 {source_type}")
        video_path = Path(source)
        if not video_path.exists():
            raise FileNotFoundError(f"视频文件不存在：{source}")
        if not self.supports(file_type=video_path.suffix.lstrip(".")):
            raise ValueError(f"不支持的视频格式：{video_path.suffix}")
        return video_path

    def _build_segment_chunks(
        self,
        video_name: str,
        video_path: str,
        segments: Dict[str, Dict[str, Any]],
    ) -> List[ConversionChunk]:
        ordered_segments = sorted(segments.items(), key=lambda item: int(item[0]))
        segment_total = len(ordered_segments)
        chunks: List[ConversionChunk] = []
        for chunk_idx, (segment_id, payload) in enumerate(ordered_segments):
            summary_text = (
                payload.get("metadata", {}).get("chunk_summary")
                if isinstance(payload.get("metadata"), dict)
                else None
            )
            text = (summary_text or payload.get("content", "")).strip()
            if not text:
                continue
            segment_meta = payload.get("metadata", {}).copy()
            metadata = {
                "source_type": "video",
                "video_name": video_name,
                "video_path": video_path,
                "segment_index": int(segment_id),
                "time_range": payload.get("time"),
                "frame_times": payload.get("frame_times"),
                "transcript": payload.get("transcript"),
                "duration_seconds": payload.get("duration_seconds"),
                "chunk_index": chunk_idx,
                "chunk_count_estimate": segment_total,
            }
            metadata.update({k: v for k, v in segment_meta.items() if v is not None})
            chunks.append(
                ConversionChunk(
                    text=text,
                    chunk_index=chunk_idx,
                    metadata=metadata,
                )
            )
        return chunks

    def _maybe_query_summary(
        self,
        videorag: VideoRAG,
        video_name: str,
        next_chunk_index: int,
        kwargs: Dict[str, Any],
    ) -> Optional[ConversionChunk]:
        auto_summary = kwargs.get("auto_summary", self.config.get("auto_summary", False))
        question: Optional[str] = kwargs.get("question") or self.config.get("question")
        if not question and auto_summary:
            question = self.DEFAULT_QUERY
        if not question:
            return None

        debug_caption = kwargs.get("debug_caption", self.config.get("debug_caption", False))
        videorag.load_caption_model(debug=debug_caption)
        query_param = kwargs.get("query_param") or self.config.get("query_param") or QueryParam(mode="videorag")
        response = videorag.query(query=question, param=query_param)
        if isinstance(response, str):
            summary_text = response
        else:
            summary_text = json.dumps(response, ensure_ascii=False, indent=2)

        return ConversionChunk(
            text=summary_text,
            chunk_index=next_chunk_index,
            metadata={
                "chunk_type": "videorag_summary",
                "question": question,
                "video_name": video_name,
            },
        )

    def _ingest_video_with_progress(self, videorag: VideoRAG, video_path: str) -> None:
        loop = always_get_an_event_loop()
        video_name = Path(video_path).stem
        if video_name in videorag.video_segments._data:
            self._report_progress(0.7, f"视频 {video_name} 已存在，跳过重建")
            return

        self._report_progress(0.08, "注册视频路径")
        loop.run_until_complete(videorag.video_path_db.upsert({video_name: video_path}))

        self._report_progress(0.12, "切分视频 & 采样帧")
        segment_index2name, segment_times_info = split_video(
            video_path,
            videorag.working_dir,
            videorag.video_segment_length,
            videorag.rough_num_frames_per_segment,
            videorag.audio_output_format,
        )

        self._report_progress(0.2, "执行语音识别")
        transcripts, languages = speech_to_text(
            video_name,
            videorag.working_dir,
            segment_index2name,
            videorag.audio_output_format,
        )

        manager = multiprocessing.Manager()
        captions = manager.dict()
        error_queue = manager.Queue()

        self._report_progress(0.28, "保存视频切片 & 生成多模态描述")
        process_saving_video_segments = multiprocessing.Process(
            target=saving_video_segments,
            args=(
                video_name,
                video_path,
                videorag.working_dir,
                segment_index2name,
                segment_times_info,
                error_queue,
                videorag.video_output_format,
            ),
        )

        process_segment_caption = multiprocessing.Process(
            target=segment_caption,
            args=(
                video_name,
                video_path,
                segment_index2name,
                transcripts,
                segment_times_info,
                captions,
                error_queue,
            ),
        )

        process_saving_video_segments.start()
        process_segment_caption.start()
        process_saving_video_segments.join()
        process_segment_caption.join()

        while not error_queue.empty():
            error_message = error_queue.get()
            logger.error(f"[VideoConverter] 视频处理失败：{error_message}")
            raise RuntimeError(error_message)

        self._report_progress(0.45, "汇总片段信息")
        captions = dict(captions)
        segments_information = merge_segment_information(
            segment_index2name,
            segment_times_info,
            transcripts,
            captions,
            languages,
        )
        manager.shutdown()

        loop.run_until_complete(videorag.video_segments.upsert({video_name: segments_information}))
        self._report_progress(0.55, "写入视频分段缓存")

        self._report_progress(0.62, "编码视频特征向量")
        loop.run_until_complete(
            videorag.video_segment_feature_vdb.upsert(
                video_name,
                segment_index2name,
                videorag.video_output_format,
            )
        )

        cache_path = Path(videorag.working_dir) / "_cache" / video_name
        if cache_path.exists():
            shutil.rmtree(cache_path)

        loop.run_until_complete(videorag._save_video_segments())
        self._report_progress(0.68, "持久化视频分段")

        self._report_progress(0.7, "更新多模态索引")
        loop.run_until_complete(videorag.ainsert(videorag.video_segments._data))


# 注册 VideoConverter
ConverterFactory.register("video", VideoConverter, priority=0)