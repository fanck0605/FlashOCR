"""Asynchronous, bucketed OCR with cross-request batching."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .det_pipeline import generate_det_buckets
from .rec_pipeline import generate_rec_buckets

if TYPE_CHECKING:
    from .api import RapidOCRv2

__all__ = ["RapidOCRv2", "generate_det_buckets", "generate_rec_buckets"]


def __getattr__(name: str) -> type[RapidOCRv2]:
    if name == "RapidOCRv2":
        from .api import RapidOCRv2

        return RapidOCRv2
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted([*globals(), *__all__])
