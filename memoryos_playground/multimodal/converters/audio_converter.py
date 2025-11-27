"""Placeholder audio converter implementation."""

from __future__ import annotations

from typing import Any, List

from ..converter import ConversionChunk, ConversionOutput, MultimodalConverter
from ..factory import ConverterFactory


class AudioConverter(MultimodalConverter):
    SUPPORTED_EXTENSIONS: List[str] = ["mp3", "wav", "flac", "aac", "ogg"]

    def convert(self, source, *, source_type: str = "file_path", **kwargs: Any) -> ConversionOutput:
        self._report_progress(0.15, "Starting placeholder audio conversion")
        placeholder_text = (
            "Audio ingestion placeholder. Replace AudioConverter with Whisper or a speech-to-text SDK."
        )
        chunk = ConversionChunk(text=placeholder_text, chunk_index=0, metadata={"source_type": "audio"})
        self._report_progress(1.0, "Audio conversion placeholder finished")
        return ConversionOutput(
            status="partial",
            chunks=[chunk],
            metadata={"converter_provider": "placeholder_audio"},
            error="Audio conversion not implemented. Please provide a concrete converter.",
        )

    def supports(self, *, file_type: str, mime_type: str = None) -> bool:
        return file_type.lower() in self.SUPPORTED_EXTENSIONS


ConverterFactory.register("audio", AudioConverter, priority=0)

