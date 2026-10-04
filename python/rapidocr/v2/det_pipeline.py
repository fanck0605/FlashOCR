from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
from omegaconf import DictConfig

from ..ch_ppocr_det.utils import DBPostProcess, DetPreProcess, TextDetOutput
from ..inference_engine.base import InferSession, get_engine
from ..utils.process_img import get_rotate_crop_image

DetShape = tuple[int, int]


@dataclass(frozen=True)
class DetPlan:
    bucket: DetShape
    resized_shape: DetShape


DetInput = tuple[np.ndarray, DetPlan]
DetResult = tuple[TextDetOutput, list[np.ndarray]]
DetQueueItem = tuple[float, DetInput, "asyncio.Future[DetResult]"]


class DetPipeline:
    def __init__(
        self,
        cfg: DictConfig,
        buckets: Iterable[DetShape],
        batch_size: int = 4,
        max_wait_ms: float = 3,
    ) -> None:
        self._cfg = cfg
        self._preprocess = DetPreProcess(
            cfg.limit_side_len, cfg.limit_type, cfg.get("mean"), cfg.get("std")
        )
        self._postprocess = DBPostProcess(
            thresh=cfg.get("thresh", 0.3),
            box_thresh=cfg.get("box_thresh", 0.5),
            max_candidates=cfg.get("max_candidates", 1000),
            unclip_ratio=cfg.get("unclip_ratio", 1.6),
            use_dilation=cfg.get("use_dilation", True),
            score_mode=cfg.get("score_mode", "fast"),
        )
        self._buckets = tuple(
            sorted(set(tuple(b) for b in buckets), key=lambda b: b[0] * b[1])
        )
        self._batch_size = batch_size
        self._max_wait = max_wait_ms / 1000
        self._queues: dict[DetShape, deque[DetQueueItem]] = {
            bucket: deque() for bucket in self._buckets
        }
        self._wakeup = asyncio.Event()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._worker: asyncio.Task[None] | None = None
        self._closed = False
        self._session: InferSession | None = None

    async def start(self, warmup: bool = True) -> None:
        loop = asyncio.get_running_loop()
        self._session = self._load_session()
        if warmup:
            for h, w in self._buckets:
                await loop.run_in_executor(
                    self._executor,
                    self._session,
                    np.zeros((self._batch_size, 3, h, w), np.float32),
                )
        self._worker = loop.create_task(self._run_batches())

    def _load_session(self) -> InferSession:
        factory = cast(
            Callable[[DictConfig], InferSession], get_engine(self._cfg.engine_type)
        )
        return factory(self._cfg)

    async def _run_batches(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            self._wakeup.clear()
            active = [(shape, queue) for shape, queue in self._queues.items() if queue]
            if not active:
                if self._closed:
                    return
                await self._wakeup.wait()
                continue
            now = loop.time()
            ready = [
                (shape, queue)
                for shape, queue in active
                if self._closed
                or len(queue) >= self._batch_size
                or now - queue[0][0] >= self._max_wait
            ]
            if not ready:
                delay = min(queue[0][0] + self._max_wait - now for _, queue in active)
                try:
                    await asyncio.wait_for(self._wakeup.wait(), delay)
                except asyncio.TimeoutError:
                    pass
                continue
            shape, queue = min(ready, key=lambda entry: entry[1][0][0])
            batch = [queue.popleft() for _ in range(min(len(queue), self._batch_size))]
            pending = [
                (item, future) for _, item, future in batch if not future.cancelled()
            ]
            if not pending:
                continue
            try:
                results = await self._infer(shape, [item for item, _ in pending])
                for (_, future), result in zip(pending, results):
                    if not future.done():
                        future.set_result(result)
            except Exception as exc:
                for _, future in pending:
                    if not future.done():
                        future.set_exception(exc)

    def _plan(self, img: np.ndarray) -> DetPlan:
        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            raise ValueError("Image dimensions must be positive")
        limit = self._preprocess.limit_side_len
        if self._preprocess.limit_type == "max":
            ratio = min(1.0, limit / max(h, w))
        else:
            ratio = max(1.0, limit / min(h, w))
        h = int(round(int(h * ratio) / 32) * 32)
        w = int(round(int(w * ratio) / 32) * 32)
        if h <= 0 or w <= 0:
            raise ValueError("Cannot resize image")
        fits = [b for b in self._buckets if b[0] >= h and b[1] >= w]
        bucket = (
            fits[0]
            if fits
            else max(self._buckets, key=lambda b: min(b[0] / h, b[1] / w))
        )
        scale = min(1.0, bucket[0] / h, bucket[1] / w)
        if scale < 1:
            h, w = max(1, int(h * scale)), max(1, int(w * scale))
        return DetPlan(bucket=(bucket[0], bucket[1]), resized_shape=(h, w))

    async def detect(self, img: np.ndarray) -> DetResult:
        if self._closed:
            raise RuntimeError("Det pipeline is closed")
        plan = self._plan(img)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[DetResult] = loop.create_future()
        self._queues[plan.bucket].append((loop.time(), (img, plan), future))
        self._wakeup.set()
        return await future

    async def _infer(self, bucket: DetShape, items: list[DetInput]) -> list[DetResult]:
        assert self._session is not None
        bh, bw = bucket
        pre = self._preprocess
        tensor = np.zeros((self._batch_size, 3, bh, bw), np.float32)
        for i, (img, plan) in enumerate(items):
            h, w = plan.resized_shape
            resized = cv2.resize(img, (w, h))
            tensor[i, :, :h, :w] = pre.permute(pre.normalize(resized))
            del resized
            await asyncio.sleep(0)
        preds = await asyncio.get_running_loop().run_in_executor(
            self._executor, self._session, tensor
        )
        outputs: list[DetResult] = []
        for i, (img, plan) in enumerate(items):
            h, w = plan.resized_shape
            ph, pw = preds.shape[-2:]
            valid = preds[
                i : i + 1, :, : max(1, round(h * ph / bh)), : max(1, round(w * pw / bw))
            ]
            boxes, scores = self._postprocess(valid, plan.resized_shape)
            if len(boxes):
                boxes = boxes.astype(np.float32)
                boxes[:, :, 0] *= img.shape[1] / w
                boxes[:, :, 1] *= img.shape[0] / h
                order = np.lexsort((boxes[:, 0, 0], boxes[:, 0, 1]))
                boxes = boxes[order]
                scores = [scores[j] for j in order]
                result = TextDetOutput(img, boxes.astype(np.int32), scores)
                crops = []
                for box in boxes:
                    crops.append(get_rotate_crop_image(img, box.copy()))
                    await asyncio.sleep(0)
            else:
                result, crops = TextDetOutput(), []
            outputs.append((result, crops))
            await asyncio.sleep(0)
        return outputs

    async def close(self) -> None:
        self._closed = True
        self._wakeup.set()
        if self._worker is not None:
            await self._worker
        self._executor.shutdown(wait=True)
