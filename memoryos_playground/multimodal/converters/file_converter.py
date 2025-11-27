"""Placeholder document converter implementation."""

from __future__ import annotations

from typing import Any, List

from ..converter import ConversionChunk, ConversionOutput, MultimodalConverter
from ..factory import ConverterFactory


class DocumentConverter(MultimodalConverter):
    SUPPORTED_EXTENSIONS: List[str] = [
        "pdf",
        "docx",
        "doc",
        "md",
        "txt",
        "pptx",
    ]

    def convert(self, source, *, source_type: str = "file_path", **kwargs: Any) -> ConversionOutput:
        self._report_progress(0.1, "Starting placeholder document conversion")
        placeholder_text = (
            "Document ingestion placeholder. Replace DocumentConverter with an actual parsing pipeline."
        )
        chunk = ConversionChunk(text=placeholder_text, chunk_index=0, metadata={"source_type": "document"})
        self._report_progress(1.0, "Document conversion placeholder finished")
        return ConversionOutput(
            status="partial",
            chunks=[chunk],
            metadata={"converter_provider": "placeholder_document"},
            error="Document conversion not implemented. Please provide a concrete converter.",
        )

    def supports(self, *, file_type: str, mime_type: str = None) -> bool:
        return file_type.lower() in self.SUPPORTED_EXTENSIONS


ConverterFactory.register("document", DocumentConverter, priority=0)

