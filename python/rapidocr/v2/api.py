from __future__ import annotations

import asyncio
from collections.abc import Iterable
from pathlib import Path
from types import TracebackType
from typing import Any, cast

import numpy as np
from omegaconf import DictConfig

from ..cal_rec_boxes import CalRecBoxes
from ..ch_ppocr_cls import TextClsOutput
from ..ch_ppocr_det import TextDetOutput
from ..ch_ppocr_rec import TextRecOutput
from ..utils.load_image import InputType, LoadImage
from ..utils.log import logger
from ..utils.output import RapidOCROutput
from ..utils.parse_parameters import ParseParams
from ..utils.process_img import (
    apply_vertical_padding,
    map_boxes_to_original,
    map_img_to_original,
    resize_image_within_bounds,
)
from ..utils.vis_res import VisRes
from .cls_pipeline import ClsPipeline
from .det_pipeline import DetPipeline
from .rec_pipeline import RecPipeline
from .typings import HWCImage


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
        det_concurrency: int = 1,
        rec_batch_size: int = 16,
        cls_batch_size: int = 16,
        max_wait_ms: float = 3,
        warmup: bool = True,
    ) -> None:
        self._cfg = self._load_config(config_path, params)
        self._load_img = LoadImage()
        self._cal_rec_boxes = CalRecBoxes()
        self._det_pipeline: DetPipeline | None = None
        if self._cfg.Global.use_det:
            self._det_pipeline = DetPipeline(
                self._cfg.Det,
                det_buckets,
                det_batch_size,
                max_wait_ms,
                det_concurrency,
            )
        self._rec_pipeline: RecPipeline | None = None
        if self._cfg.Global.use_rec:
            self._rec_pipeline = RecPipeline(
                self._cfg.Rec,
                rec_widths,
                rec_batch_size,
                max_wait_ms,
                return_word_box=self._cfg.Global.return_word_box,
            )
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
        if self._det_pipeline is not None:
            await self._det_pipeline.start(self._warmup)
        if self._rec_pipeline is not None:
            await self._rec_pipeline.start(self._warmup)
        if self._cls_pipeline is not None:
            await self._cls_pipeline.start(self._warmup)

    def _prepare(self, image: InputType) -> tuple[HWCImage, HWCImage, dict[str, Any]]:
        original = self._load_img(image)
        settings = self._cfg.Global
        if settings.use_preprocess_img:
            img, ratio_h, ratio_w = resize_image_within_bounds(
                original, settings.min_side_len, settings.max_side_len
            )
        else:
            img, ratio_h, ratio_w = original, 1.0, 1.0
        record = {"preprocess": {"ratio_h": ratio_h, "ratio_w": ratio_w}}
        if self._det_pipeline is not None and settings.use_vertical_padding:
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
        original, prepared, record = self._prepare(image)
        det = TextDetOutput(img=prepared)
        crops = [prepared]
        if self._det_pipeline is not None:
            det, crops = await self._det_pipeline.detect(prepared)
        if not crops:
            return RapidOCROutput()
        cls = TextClsOutput()
        rec_images = crops
        if self._cls_pipeline is not None:
            cls = await self._cls_pipeline.classify(crops)
            rec_images = cls.img_list
            assert rec_images is not None
        if self._rec_pipeline is None:
            return self._build_detection_output(original, det, record)
        rec = await self._rec_pipeline.recognize(rec_images)
        return self._build_output(original, det, cls, rec, crops, record)

    def _build_output(
        self,
        original: HWCImage,
        det: TextDetOutput,
        cls: TextClsOutput,
        rec: TextRecOutput,
        crops: list[HWCImage],
        record: dict[str, Any],
    ) -> RapidOCROutput:
        assert rec.txts is not None
        boxes = None
        if det.boxes is not None:
            boxes = map_boxes_to_original(
                det.boxes.astype(np.float32), record, *original.shape[:2]
            )
        # Legacy components annotate variable-length results as fixed-size tuples.
        words: tuple[Any, ...] = rec.word_results
        if boxes is not None and self._cfg.Global.return_word_box and all(words):
            metadata = record.get("preprocess")
            assert metadata is not None
            original_crops = map_img_to_original(
                crops, metadata["ratio_h"], metadata["ratio_w"]
            )
            word_output = self._cal_rec_boxes(
                original_crops, boxes, rec, self._cfg.Global.return_single_char_box
            )
            words = tuple(
                tuple(
                    (item[0], item[1], item[2])
                    for item in line
                    if isinstance(item, tuple)
                    and len(item) == 3
                    and item[2] is not None
                )
                for line in cast(tuple[tuple[Any, ...], ...], word_output.word_results)
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
            boxes=boxes[indices] if boxes is not None else None,
            txts=cast(tuple[str], tuple(rec.txts[i] for i in indices)),
            scores=cast(tuple[float], tuple(rec.scores[i] for i in indices)),
            word_results=cast(
                tuple[tuple[str, float, list[list[int]] | None]],
                tuple(words[i] for i in indices),
            ),
            elapse_list=[det.elapse, cls.elapse, rec.elapse],
            viser=VisRes(
                text_score=self._cfg.Global.text_score,
                lang_type=self._cfg.Rec.lang_type,
                font_path=self._cfg.Global.font_path,
            ),
        )

    def _build_detection_output(
        self,
        original: HWCImage,
        det: TextDetOutput,
        record: dict[str, Any],
    ) -> RapidOCROutput:
        if det.boxes is None or det.scores is None:
            return RapidOCROutput()
        boxes = map_boxes_to_original(
            det.boxes.astype(np.float32), record, *original.shape[:2]
        )
        return RapidOCROutput(
            img=original,
            boxes=boxes,
            scores=cast(tuple[float], tuple(det.scores)),
            elapse_list=[det.elapse],
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
        if self._det_pipeline is not None:
            await self._det_pipeline.close()
        if self._cls_pipeline is not None:
            await self._cls_pipeline.close()
        if self._rec_pipeline is not None:
            await self._rec_pipeline.close()

    async def __aenter__(self) -> RapidOCRv2:
        return await self.start()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()
