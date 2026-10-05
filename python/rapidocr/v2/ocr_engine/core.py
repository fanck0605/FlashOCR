from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ...cal_rec_boxes import CalRecBoxes
from ...ch_ppocr_cls import TextClsOutput
from ...ch_ppocr_det import TextDetOutput
from ...utils.output import RapidOCROutput
from ...utils.process_img import (
    apply_vertical_padding,
    map_boxes_to_original,
    map_img_to_original,
    resize_image_within_bounds,
)
from ...utils.vis_res import VisRes
from ..cls_engine import ClsEngine
from ..det_engine import DetEngine, DetShape, generate_det_buckets
from ..rec_engine import RecEngine, generate_rec_buckets

if TYPE_CHECKING:
    from collections.abc import Iterable

    from omegaconf import DictConfig

    from ...ch_ppocr_rec import TextRecOutput
    from ..typing import HWCImage


class OCREngine:
    """Compose detection, classification and recognition for loaded images."""

    def __init__(
        self,
        cfg: DictConfig,
        *,
        det_buckets: Iterable[DetShape] = generate_det_buckets(160, 640, 16),
        rec_buckets: Iterable[int] = generate_rec_buckets(3840, 8),
        det_batch_size: int = 4,
        det_concurrency: int = 1,
        rec_batch_size: int = 32,
        rec_concurrency: int = 1,
        cls_batch_size: int = 16,
        cls_concurrency: int = 1,
        max_wait: float = 0.003,
    ) -> None:
        self._cfg = cfg
        self._cal_rec_boxes = CalRecBoxes()
        self._det_engine: DetEngine | None = None
        if self._cfg.Global.use_det:
            self._det_engine = DetEngine(
                self._cfg.Det,
                det_buckets,
                det_batch_size,
                max_wait,
                det_concurrency,
            )
        self._rec_engine: RecEngine | None = None
        if self._cfg.Global.use_rec:
            self._rec_engine = RecEngine(
                self._cfg.Rec,
                rec_buckets,
                rec_batch_size,
                max_wait,
                return_word_box=self._cfg.Global.return_word_box,
                concurrency=rec_concurrency,
            )
        self._cls_engine: ClsEngine | None = None
        if self._cfg.Global.use_cls:
            self._cls_engine = ClsEngine(
                self._cfg.Cls, cls_batch_size, max_wait, concurrency=cls_concurrency
            )
        self._start_task: asyncio.Task[None] | None = None
        self._closed: bool = False
        self._requests: set[asyncio.Task[RapidOCROutput]] = set()

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("OCR is closed")
        if self._start_task is None:
            self._start_task = asyncio.create_task(self._start())
        await asyncio.shield(self._start_task)

    async def _start(self) -> None:
        if self._det_engine is not None:
            await self._det_engine.start()
        if self._rec_engine is not None:
            await self._rec_engine.start()
        if self._cls_engine is not None:
            await self._cls_engine.start()

    def _prepare(self, image: HWCImage) -> tuple[HWCImage, HWCImage, dict[str, Any]]:
        original = image
        settings = self._cfg.Global
        if settings.use_preprocess_img:
            img, ratio_h, ratio_w = resize_image_within_bounds(
                original, settings.min_side_len, settings.max_side_len
            )
        else:
            img, ratio_h, ratio_w = original, 1.0, 1.0
        record = {
            "preprocess": {"ratio_h": ratio_h, "ratio_w": ratio_w},
        }
        if self._det_engine is not None and settings.use_vertical_padding:
            img, record = apply_vertical_padding(
                img, record, settings.width_height_ratio, settings.min_height
            )
        else:
            record["padding_1"] = {"top": 0, "left": 0}
        return original, img, record

    async def __call__(self, image: HWCImage) -> RapidOCROutput:
        await self.start()
        if self._closed:
            raise RuntimeError("OCR is closed")
        task = asyncio.create_task(self._request(image))
        self._requests.add(task)
        task.add_done_callback(self._requests.discard)
        return await task

    async def batch(self, images: Iterable[HWCImage]) -> list[RapidOCROutput]:
        return list(await asyncio.gather(*(self(image) for image in images)))

    async def _request(self, image: HWCImage) -> RapidOCROutput:
        original, prepared, record = self._prepare(image)
        det = TextDetOutput(img=prepared)
        crops = [prepared]
        if self._det_engine is not None:
            det, crops = await self._det_engine.detect(prepared)
        if not crops:
            return RapidOCROutput()
        cls = TextClsOutput()
        rec_images = crops
        if self._cls_engine is not None:
            cls = await self._cls_engine.classify(crops)
            rec_images = cls.img_list
            assert rec_images is not None
        if self._rec_engine is None:
            return self._build_detection_output(original, det, record)
        rec = await self._rec_engine.recognize(rec_images)
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
                for line in cast(
                    "tuple[tuple[Any, ...], ...]", word_output.word_results
                )
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
            txts=cast("tuple[str]", tuple(rec.txts[i] for i in indices)),
            scores=cast("tuple[float]", tuple(rec.scores[i] for i in indices)),
            word_results=cast(
                "tuple[tuple[str, float, list[list[int]] | None]]",
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
            scores=cast("tuple[float]", tuple(det.scores)),
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
        if self._det_engine is not None:
            await self._det_engine.close()
        if self._cls_engine is not None:
            await self._cls_engine.close()
        if self._rec_engine is not None:
            await self._rec_engine.close()
