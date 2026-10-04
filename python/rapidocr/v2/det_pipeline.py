from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from omegaconf import DictConfig

from ..ch_ppocr_det import TextDetector, TextDetOutput

from ..utils.process_img import get_rotate_crop_image

DetShape = tuple[int, int]
DetInput = tuple[np.ndarray, np.ndarray]
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
        self._buckets = tuple(
            sorted(set(tuple(b) for b in buckets), key=lambda b: b[0] * b[1])
        )
        self._batch_size = batch_size
        self._max_wait = max_wait_ms / 1000
        self._queues: dict[DetShape, deque[DetQueueItem]] = {}
        self._wakeup = asyncio.Event()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._worker: asyncio.Task[None] | None = None
        self._closed = False
        self._model: TextDetector | None = None

    async def start(self, warmup: bool = True) -> None:
        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(
            self._executor, TextDetector, self._cfg
        )
        if warmup:
            for h, w in self._buckets:
                await loop.run_in_executor(
                    self._executor,
                    self._model.session,
                    np.zeros((self._batch_size, 3, h, w), np.float32),
                )
        self._worker = loop.create_task(self._run_batches())

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
                results = await loop.run_in_executor(
                    self._executor, self._infer, shape, [item for item, _ in pending]
                )
                for (_, future), result in zip(pending, results):
                    if not future.done():
                        future.set_result(result)
            except Exception as exc:
                for _, future in pending:
                    if not future.done():
                        future.set_exception(exc)

    def _prepare(self, img: np.ndarray) -> tuple[DetShape, np.ndarray]:
        assert self._model is not None
        pre = self._model.get_preprocess()
        resized = pre.resize(img)
        if resized is None:
            raise ValueError("Cannot resize image")
        h, w = resized.shape[:2]
        fits = [b for b in self._buckets if b[0] >= h and b[1] >= w]
        bucket = (
            fits[0]
            if fits
            else max(self._buckets, key=lambda b: min(b[0] / h, b[1] / w))
        )
        scale = min(1.0, bucket[0] / h, bucket[1] / w)
        if scale < 1:
            resized = cv2.resize(
                resized, (max(1, int(w * scale)), max(1, int(h * scale)))
            )
        return bucket, resized

    async def submit(self, img: np.ndarray) -> DetResult:
        if self._closed:
            raise RuntimeError("Det pipeline is closed")
        bucket, resized = self._prepare(img)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[DetResult] = loop.create_future()
        self._queues.setdefault(bucket, deque()).append(
            (loop.time(), (img, resized), future)
        )
        self._wakeup.set()
        return await future

    def _infer(self, bucket: DetShape, items: list[DetInput]) -> list[DetResult]:
        assert self._model is not None
        bh, bw = bucket
        pre = self._model.get_preprocess()
        tensor = np.zeros((self._batch_size, 3, bh, bw), np.float32)
        for i, (_, resized) in enumerate(items):
            h, w = resized.shape[:2]
            tensor[i, :, :h, :w] = pre.permute(pre.normalize(resized))
        preds = self._model.session(tensor)
        outputs: list[DetResult] = []
        for i, (img, resized) in enumerate(items):
            h, w = resized.shape[:2]
            ph, pw = preds.shape[-2:]
            valid = preds[
                i : i + 1, :, : max(1, round(h * ph / bh)), : max(1, round(w * pw / bw))
            ]
            boxes, scores = self._model.postprocess_op(valid, resized.shape[:2])
            if len(boxes):
                boxes = boxes.astype(np.float32)
                boxes[:, :, 0] *= img.shape[1] / resized.shape[1]
                boxes[:, :, 1] *= img.shape[0] / resized.shape[0]
                order = np.lexsort((boxes[:, 0, 0], boxes[:, 0, 1]))
                boxes = boxes[order]
                scores = [scores[j] for j in order]
                result = TextDetOutput(img, boxes.astype(np.int32), scores)
                crops = [get_rotate_crop_image(img, box.copy()) for box in boxes]
            else:
                result, crops = TextDetOutput(), []
            outputs.append((result, crops))
        return outputs

    async def close(self) -> None:
        self._closed = True
        self._wakeup.set()
        if self._worker is not None:
            await self._worker
        self._executor.shutdown(wait=True)
