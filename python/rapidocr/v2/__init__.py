"""Asynchronous, bucketed OCR with cross-request batching."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .det_engine import generate_det_buckets
from .rec_engine import generate_rec_buckets

if TYPE_CHECKING:
    from .api import FlashOCR

__all__ = ["FlashOCR", "generate_det_buckets", "generate_rec_buckets"]


def __getattr__(name: str) -> type[FlashOCR]:
    if name == "FlashOCR":
        from .api import FlashOCR

        return FlashOCR
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted([*globals(), *__all__])
