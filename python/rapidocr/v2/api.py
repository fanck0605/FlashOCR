from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..utils.load_image import InputType, LoadImage
from ..utils.log import logger
from ..utils.parse_parameters import ParseParams
from .det_pipeline import DetShape, generate_det_buckets
from .ocr_pipeline import OCRPipeline
from .rec_pipeline import generate_rec_buckets

if TYPE_CHECKING:
    from collections.abc import Iterable
    from types import TracebackType

    from omegaconf import DictConfig

    from ..utils.output import RapidOCROutput

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


class RapidOCRv2:
    """Load configuration and images for the asynchronous OCR pipeline."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        params: dict[str, Any] | None = None,
        *,
        det_buckets: Iterable[DetShape] = generate_det_buckets(160, 640, 16),
        rec_buckets: Iterable[int] = generate_rec_buckets(3840, 16),
        det_batch_size: int = 4,
        det_concurrency: int = 1,
        rec_batch_size: int = 16,
        rec_concurrency: int = 1,
        cls_batch_size: int = 16,
        cls_concurrency: int = 1,
        max_wait: float = 0.003,
    ) -> None:
        cfg = self._load_config(config_path, params)
        self._load_img = LoadImage()
        self._pipeline = OCRPipeline(
            cfg,
            det_buckets=det_buckets,
            rec_buckets=rec_buckets,
            det_batch_size=det_batch_size,
            det_concurrency=det_concurrency,
            rec_batch_size=rec_batch_size,
            rec_concurrency=rec_concurrency,
            cls_batch_size=cls_batch_size,
            cls_concurrency=cls_concurrency,
            max_wait=max_wait,
        )

    @staticmethod
    def _load_config(
        config_path: str | Path | None, params: dict[str, Any] | None
    ) -> DictConfig:
        root = Path(__file__).resolve().parent.parent
        cfg = ParseParams.load(config_path or root / "config.yaml")
        if params:
            cfg = ParseParams.update_batch(cfg, params)
        if cfg.Global.model_root_dir is None:
            cfg.Global.model_root_dir = root / "models"
        logger.setLevel(cfg.Global.log_level.upper())
        for name in ("Det", "Rec", "Cls"):
            model_cfg = cfg[name]
            model_cfg.engine_cfg = cfg.EngineConfig[model_cfg.engine_type.value]
            model_cfg.model_root_dir = cfg.Global.model_root_dir
        cfg.Rec.font_path = cfg.Global.font_path
        return cfg

    async def start(self) -> Self:
        await self._pipeline.start()
        return self

    async def __call__(self, image: InputType) -> RapidOCROutput:
        await self.start()
        return await self._pipeline(self._load_img(image))

    async def batch(self, images: Iterable[InputType]) -> list[RapidOCROutput]:
        return list(await asyncio.gather(*(self(image) for image in images)))

    async def close(self) -> None:
        await self._pipeline.close()

    async def __aenter__(self) -> Self:
        return await self.start()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()
