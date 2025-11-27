"""Placeholder image converter implementation."""

from __future__ import annotations

from typing import Any, List

from ..converter import ConversionChunk, ConversionOutput, MultimodalConverter
from ..factory import ConverterFactory


class ImageConverter(MultimodalConverter):
    SUPPORTED_EXTENSIONS: List[str] = ["png", "jpg", "jpeg", "gif", "bmp", "webp"]

    def convert(self, source, *, source_type: str = "file_path", **kwargs: Any) -> ConversionOutput:
        """
        示例实现：不做实际 OCR，只返回固定内容，便于演示调用流程。
        """
        self._report_progress(0.5, "Simulating image-to-text conversion")
        chunk = ConversionChunk(
            text="【示例OCR结果】这是一段由 ImageConverter 固定输出的文本。",
            chunk_index=0,
            metadata={"source_type": "image"},
        )
        self._report_progress(1.0, "Image conversion simulation completed")
        return ConversionOutput(
            status="success",
            chunks=[chunk],
            metadata={"converter_provider": "demo_image_converter"},
        )

    def supports(self, *, file_type: str, mime_type: str = None) -> bool:
        return file_type.lower() in self.SUPPORTED_EXTENSIONS


ConverterFactory.register("image", ImageConverter, priority=0)

