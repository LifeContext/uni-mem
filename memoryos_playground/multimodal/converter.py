"""Abstract definitions for multimodal converters."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

TextSource = Union[str, bytes, Path]
ProgressCallback = Callable[[float, str], None]


@dataclass
class ConversionChunk:
    """Represents a single chunk of converted text."""

    text: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversionOutput:
    """Standard output structure returned by a converter."""

    status: str  # success, partial, failed
    text: str = ""
    chunks: List[ConversionChunk] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def ensure_chunks(self) -> None:
        """Ensure that chunks list is populated even if only raw text was set."""
        if self.chunks or not self.text:
            return
        self.chunks = [
            ConversionChunk(text=self.text, chunk_index=0, metadata=self.metadata.copy())
        ]


class MultimodalConverter(ABC):
    """Base class for all multimodal converters."""

    def __init__(
        self,
        *,
        max_chunk_tokens: int = 4000,
        progress_callback: Optional[ProgressCallback] = None,
        retry_count: int = 0,
        retry_delay: float = 1.0,
        **config: Any,
    ) -> None:
        self.max_chunk_tokens = max_chunk_tokens
        self.progress_callback = progress_callback
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self.config = config

    @abstractmethod
    def convert(
        self,
        source: TextSource,
        *,
        source_type: str = "file_path",
        **kwargs: Any,
    ) -> ConversionOutput:
        """Convert the source into text segments."""

    @abstractmethod
    def supports(self, *, file_type: str, mime_type: Optional[str] = None) -> bool:
        """Return True if this converter can handle the given file type/mime."""

    # --- Helper methods available to subclasses ---
    def _chunk_text(
        self,
        text: str,
        *,
        chunk_size: int = 2000,
        overlap: int = 200,
        base_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[ConversionChunk]:
        """Split long text into overlapping chunks to fit memory limits."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        if overlap < 0:
            raise ValueError("overlap cannot be negative")

        tokens = text.split()
        chunks: List[ConversionChunk] = []
        step = max(1, chunk_size - overlap)
        total = math.ceil(len(tokens) / step)
        for idx, start in enumerate(range(0, len(tokens), step)):
            end = min(len(tokens), start + chunk_size)
            chunk_tokens = tokens[start:end]
            chunk_text = " ".join(chunk_tokens)
            chunk_meta = (base_metadata or {}).copy()
            chunk_meta["chunk_index"] = idx
            chunk_meta["chunk_count_estimate"] = total
            chunks.append(
                ConversionChunk(text=chunk_text, chunk_index=idx, metadata=chunk_meta)
            )
        return chunks

    def _report_progress(self, progress: float, message: str) -> None:
        if self.progress_callback:
            clamped = max(0.0, min(1.0, progress))
            self.progress_callback(clamped, message)

    def _iter_chunks(self, text_iterable: Iterable[str]) -> List[ConversionChunk]:
        """Utility for converters that yield strings progressively."""
        chunks: List[ConversionChunk] = []
        for idx, chunk in enumerate(text_iterable):
            chunk_meta = {"chunk_index": idx}
            chunks.append(ConversionChunk(text=chunk, chunk_index=idx, metadata=chunk_meta))
        return chunks

