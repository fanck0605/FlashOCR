from __future__ import annotations

import asyncio
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np
from omegaconf import DictConfig

from ..ch_ppocr_cls import TextClsOutput
from ..ch_ppocr_det import TextDetOutput
from ..ch_ppocr_rec import TextRecOutput
from ..cal_rec_boxes import CalRecBoxes
from ..utils.load_image import InputType, LoadImage
from ..utils.log import logger
from ..utils.parse_parameters import ParseParams
from ..utils.output import RapidOCROutput
from ..utils.process_img import (
    apply_vertical_padding,
    resize_image_within_bounds,
    map_boxes_to_original,
    map_img_to_original,
)
from ..utils.vis_res import VisRes
from .det_pipeline import DetPipeline
from .cls_pipeline import ClsPipeline
from .rec_pipeline import RecPipeline


class RapidOCRv2:
    """Async coordinator; Det and Rec are independent pipelines."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        params: dict[str, Any] | None = None,
        *,
        det_buckets: Iterable[tuple[int, int]] = (
            (512, 512),
            (736, 736),
            (736, 1280),
            (1280, 736),
            (1280, 1280),
        ),
        rec_widths: Iterable[int] = (320, 640, 960, 1280, 1920),
        det_batch_size: int = 4,
        rec_batch_size: int = 16,
        cls_batch_size: int = 16,
        max_wait_ms: float = 3,
        warmup: bool = True,
    ) -> None:
        self._cfg = self._load_config(config_path, params)
        self._load_img = LoadImage()
        self._cal_rec_boxes = CalRecBoxes()
        if not self._cfg.Global.use_det or not self._cfg.Global.use_rec:
            raise ValueError("v2 requires Det and Rec enabled")
        self._det_pipeline = DetPipeline(
            self._cfg.Det, det_buckets, det_batch_size, max_wait_ms
        )
        self._rec_pipeline = RecPipeline(
            self._cfg.Rec,
            rec_widths,
            rec_batch_size,
            max_wait_ms,
            return_word_box=self._cfg.Global.return_word_box,
        )
        self._cpu_executor = ThreadPoolExecutor(max_workers=2)
        self._cls_pipeline: ClsPipeline | None = None
        if self._cfg.Global.use_cls:
            self._cls_pipeline = ClsPipeline(self._cfg.Cls, cls_batch_size, max_wait_ms)
        self._warmup: bool = warmup
        self._start_task: asyncio.Task[None] | None = None
        self._closed: bool = False
        self._requests: set[asyncio.Task[RapidOCROutput]] = set()

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

    async def start(self) -> RapidOCRv2:
        if self._closed:
            raise RuntimeError("OCR is closed")
        if self._start_task is None:
            self._start_task = asyncio.create_task(self._start())
        await asyncio.shield(self._start_task)
        return self

    async def _start(self) -> None:
        await self._det_pipeline.start(self._warmup)
        await self._rec_pipeline.start(self._warmup)
        if self._cls_pipeline is not None:
            await self._cls_pipeline.start(self._warmup)

    def _prepare(
        self, image: InputType
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        original = self._load_img(image)
        settings = self._cfg.Global
        if settings.use_preprocess_img:
            img, ratio_h, ratio_w = resize_image_within_bounds(
                original, settings.min_side_len, settings.max_side_len
            )
        else:
            img, ratio_h, ratio_w = original, 1.0, 1.0
        record = {"preprocess": {"ratio_h": ratio_h, "ratio_w": ratio_w}}
        if settings.use_vertical_padding:
            img, record = apply_vertical_padding(
                img, record, settings.width_height_ratio, settings.min_height
            )
        else:
            record["padding_1"] = {"top": 0, "left": 0}
        return original, img, record

    async def __call__(self, image: InputType) -> RapidOCROutput:
        await self.start()
        if self._closed:
            raise RuntimeError("OCR is closed")
        task = asyncio.create_task(self._request(image))
        self._requests.add(task)
        task.add_done_callback(self._requests.discard)
        return await task

    async def batch(self, images: Iterable[InputType]) -> list[RapidOCROutput]:
        return list(await asyncio.gather(*(self(image) for image in images)))

    async def _request(self, image: InputType) -> RapidOCROutput:
        loop = asyncio.get_running_loop()
        original, prepared, record = await loop.run_in_executor(
            self._cpu_executor, self._prepare, image
        )
        det, crops = await self._det_pipeline.submit(prepared)
        if not crops:
            return RapidOCROutput()
        cls = TextClsOutput()
        rec_images = crops
        if self._cls_pipeline is not None:
            cls = await self._cls_pipeline.classify(crops)
            rec_images = cls.img_list
            assert rec_images is not None
        rec = await self._rec_pipeline.recognize(rec_images)
        return await loop.run_in_executor(
            self._cpu_executor,
            partial(self._build_output, original, det, cls, rec, crops, record),
        )

    def _build_output(
        self,
        original: np.ndarray,
        det: TextDetOutput,
        cls: TextClsOutput,
        rec: TextRecOutput,
        crops: list[np.ndarray],
        record: dict[str, Any],
    ) -> RapidOCROutput:
        assert det.boxes is not None and rec.txts is not None
        boxes = map_boxes_to_original(
            det.boxes.astype(np.float32), record, *original.shape[:2]
        )
        words = rec.word_results
        if self._cfg.Global.return_word_box and all(words):
            metadata = record["preprocess"]
            original_crops = map_img_to_original(
                crops, metadata["ratio_h"], metadata["ratio_w"]
            )
            word_output = self._cal_rec_boxes(
                original_crops, boxes, rec, self._cfg.Global.return_single_char_box
            )
            words = tuple(
                tuple(item for item in line if item[2] is not None)
                for line in word_output.word_results
            )
        indices = [
            i
            for i, (text, score) in enumerate(zip(rec.txts, rec.scores))
            if text.strip() and score >= self._cfg.Global.text_score
        ]
        if not indices:
            return RapidOCROutput()
        return RapidOCROutput(
            img=original,
            boxes=boxes[indices],
            txts=tuple(rec.txts[i] for i in indices),
            scores=tuple(rec.scores[i] for i in indices),
            word_results=tuple(words[i] for i in indices),
            elapse_list=[det.elapse, cls.elapse, rec.elapse],
            viser=VisRes(
                text_score=self._cfg.Global.text_score,
                lang_type=self._cfg.Rec.lang_type,
                font_path=self._cfg.Global.font_path,
            ),
        )

    async def close(self) -> None:
        self._closed = True
        if self._start_task is not None:
            await self._start_task
        if self._requests:
            await asyncio.gather(*tuple(self._requests), return_exceptions=True)
        await self._det_pipeline.close()
        if self._cls_pipeline is not None:
            await self._cls_pipeline.close()
        await self._rec_pipeline.close()
        self._cpu_executor.shutdown(wait=True)

    async def __aenter__(self) -> RapidOCRv2:
        return await self.start()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()
