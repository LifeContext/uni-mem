"""Factory utilities for creating and configuring multimodal converters."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Type

from .converter import MultimodalConverter


class ConverterFactory:
    """Registry + factory for converter implementations."""

    _registry: Dict[str, List[Tuple[int, Type[MultimodalConverter]]]] = defaultdict(list)
    _config: Dict[str, Dict[str, Any]] = defaultdict(dict)

    @classmethod
    def register(
        cls,
        converter_type: str,
        converter_class: Type[MultimodalConverter],
        *,
        priority: int = 0,
    ) -> None:
        """Register a converter for a logical converter type."""

        converter_type = converter_type.lower()
        cls._registry[converter_type].append((priority, converter_class))
        cls._registry[converter_type].sort(key=lambda item: item[0], reverse=True)

    @classmethod
    def configure(cls, converter_type: str, **config: Any) -> None:
        """Attach configuration that will be passed when instantiating."""

        cls._config[converter_type.lower()].update(config)

    @classmethod
    def create(
        cls,
        *,
        converter_type: Optional[str] = None,
        file_extension: Optional[str] = None,
        mime_type: Optional[str] = None,
        **overrides: Any,
    ) -> Optional[MultimodalConverter]:
        """Create a converter instance based on provided hints."""

        converter_type = (converter_type or cls._infer_type(file_extension, mime_type)).lower()
        if converter_type not in cls._registry:
            return None

        _, converter_cls = cls._registry[converter_type][0]
        config = {**cls._config.get(converter_type, {}), **overrides}
        return converter_cls(**config)

    @classmethod
    def list_supported_types(cls) -> Dict[str, List[str]]:
        """Return registered converter types and implementation names."""

        return {
            key: [converter.__name__ for _, converter in converters]
            for key, converters in cls._registry.items()
        }

    @staticmethod
    def _infer_type(file_extension: Optional[str], mime_type: Optional[str]) -> str:
        if mime_type:
            major = mime_type.split("/")[0]
            if major in {"video", "audio", "image"}:
                return major
        if file_extension:
            ext = file_extension.lower().lstrip(".")
            if ext in {"mp4", "mov", "avi", "mkv", "webm"}:
                return "video"
            if ext in {"mp3", "wav", "flac", "aac"}:
                return "audio"
            if ext in {"png", "jpg", "jpeg", "gif", "bmp"}:
                return "image"
            return "document"
        return "document"

