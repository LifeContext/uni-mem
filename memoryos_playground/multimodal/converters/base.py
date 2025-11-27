"""Placeholder converters used for wiring tests."""

from __future__ import annotations

from typing import Any

from ..converter import ConversionOutput, MultimodalConverter


class PlaceholderConverter(MultimodalConverter):
    """Converter that simply echoes a notice for unsupported types."""

    def convert(self, source, *, source_type: str = "file_path", **kwargs: Any) -> ConversionOutput:
        text = f"[Placeholder converter] Unable to process {source} (type={source_type})."
        return ConversionOutput(status="failed", text="", metadata={}, error=text)

    def supports(self, *, file_type: str, mime_type: str = None) -> bool:
        return False

