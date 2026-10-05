from __future__ import annotations

import asyncio
import math
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from typing import TYPE_CHECKING, cast

import cv2
import numpy as np

from ...ch_ppocr_cls.main import CLS_SHAPE_BY_OCR_VERSION
from ...ch_ppocr_cls.utils import ClsPostProcess, TextClsOutput
from ..session import create_session
from ..typing import HWCImage

if TYPE_CHECKING:
    import sys

    from omegaconf import DictConfig

    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias

ClsResult: TypeAlias = tuple[HWCImage, tuple[str, float]]
ClsQueueItem: TypeAlias = tuple[float, HWCImage, "asyncio.Future[ClsResult]"]


class ClsEngine:
    """Cross-request classification with one fixed input and batch shape."""

    def __init__(
        self,
        cfg: DictConfig,
        batch_size: int = 16,
        max_wait: float = 0.02,
        concurrency: int = 1,
    ) -> None:
        if (
            type(batch_size) is not int
            or batch_size < 4
            or batch_size & (batch_size - 1)
        ):
            raise ValueError("Classification batch size must be a power of two >= 4")
        if max_wait < 0:
            raise ValueError("Classification wait time must be nonnegative")
        if concurrency < 1:
            raise ValueError("Classification concurrency must be positive")
        self._session = create_session(cfg)
        self._shape = tuple(CLS_SHAPE_BY_OCR_VERSION[cfg.ocr_version])
        self._threshold: float = cfg.cls_thresh
        self._postprocess = ClsPostProcess(cfg.label_list)
        self._batch_size = batch_size
        self._max_wait = max_wait
        self._queue: deque[ClsQueueItem] = deque()
        self._wakeup = asyncio.Event()
        self._concurrency = concurrency
        self._inflight: set[asyncio.Task[None]] = set()
        self._executor = ThreadPoolExecutor(max_workers=concurrency)
        self._worker: asyncio.Task[None] | None = None
        self._closed = False
        self._batch_count = 0
        self._sample_count = 0

    def batch_stats(self) -> dict[str, float | int]:
        return {
            "batches": self._batch_count,
            "samples": self._sample_count,
            "average_actual_batch": self._sample_count / self._batch_count
            if self._batch_count
            else 0.0,
            "configured_batch_size": self._batch_size,
            "batch_utilization": self._sample_count
            / (self._batch_count * self._batch_size)
            if self._batch_count
            else 0.0,
        }

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        for exponent in range(2, self._batch_size.bit_length()):
            batch_size = 1 << exponent
            await loop.run_in_executor(
                self._executor,
                self._session,
                np.zeros((batch_size, *self._shape), np.float32),
            )
        self._worker = loop.create_task(self._run_batches())

    async def classify(self, images: list[HWCImage]) -> TextClsOutput:
        if self._closed:
            raise RuntimeError("CLS engine is closed")
        if not images:
            return TextClsOutput(img_list=[], cls_res=[], elapse=0.0)
        loop = asyncio.get_running_loop()
        futures: list[asyncio.Future[ClsResult]] = []
        for image in images:
            future: asyncio.Future[ClsResult] = loop.create_future()
            self._queue.append((loop.time(), image, future))
            futures.append(future)
        self._wakeup.set()
        results = await asyncio.gather(*futures)
        return TextClsOutput(
            img_list=[image for image, _ in results],
            cls_res=[label for _, label in results],
            elapse=0.0,
        )

    async def _run_batches(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            self._inflight = {task for task in self._inflight if not task.done()}
            if len(self._inflight) >= self._concurrency:
                done, _ = await asyncio.wait(
                    self._inflight, return_when=asyncio.FIRST_COMPLETED
                )
                self._inflight.difference_update(done)
                continue
            self._wakeup.clear()
            while self._queue and self._queue[0][2].cancelled():
                self._queue.popleft()
            if not self._queue:
                if self._closed:
                    return
                await self._wakeup.wait()
                continue
            delay = max(0.0, self._queue[0][0] + self._max_wait - loop.time())
            ready = self._closed or delay <= 0
            if ready or len(self._queue) >= self._batch_size:
                limit = min(len(self._queue), self._batch_size)
                if any(entry[2].cancelled() for entry in islice(self._queue, limit)):
                    active: list[ClsQueueItem] = []
                    while self._queue and len(active) < limit:
                        entry = self._queue.popleft()
                        if not entry[2].cancelled():
                            active.append(entry)
                    self._queue.extendleft(reversed(active))
                if not ready:
                    ready = len(self._queue) >= self._batch_size
            if not ready:
                try:
                    await asyncio.wait_for(self._wakeup.wait(), delay)
                except asyncio.TimeoutError:
                    pass
                continue
            available = min(len(self._queue), self._batch_size)
            batch_size = max(4, 1 << (available.bit_length() - 1))
            batch_size = min(batch_size, self._batch_size)
            pending = [self._queue.popleft() for _ in range(min(available, batch_size))]
            task = asyncio.create_task(self._infer_and_resolve(pending))
            self._inflight.add(task)

    async def _infer_and_resolve(self, pending: list[ClsQueueItem]) -> None:
        try:
            results = await self._infer([image for _, image, _ in pending])
            for (_, _, future), result in zip(pending, results):
                if not future.done():
                    future.set_result(result)
        except Exception as exc:  # noqa: BLE001
            for _, _, future in pending:
                if not future.done():
                    future.set_exception(exc)

    def _prepare(
        self, image: HWCImage
    ) -> np.ndarray[tuple[int, int, int], np.dtype[np.float32]]:
        channels, height, width = self._shape
        resized_width = min(width, math.ceil(height * image.shape[1] / image.shape[0]))
        resized = cv2.resize(image, (resized_width, height)).astype(np.float32)
        normalized = (resized.transpose(2, 0, 1) / 255 - 0.5) / 0.5
        padded = np.zeros((channels, height, width), np.float32)
        padded[:, :, :resized_width] = normalized
        return padded

    async def _infer(self, images: list[HWCImage]) -> list[ClsResult]:
        assert self._session is not None
        self._batch_count += 1
        self._sample_count += len(images)
        tensor = np.zeros((max(4, len(images)), *self._shape), np.float32)
        for i, image in enumerate(images):
            tensor[i] = self._prepare(image)
            await asyncio.sleep(0)
        preds = await asyncio.get_running_loop().run_in_executor(
            self._executor, self._session, tensor
        )
        labels = self._postprocess(preds[: len(images)])
        results: list[ClsResult] = []
        for image, (label, score) in zip(images, labels):
            rotated = (
                cast(
                    "HWCImage",
                    cv2.rotate(image, cv2.ROTATE_180),
                )
                if "180" in label and score > self._threshold
                else image
            )
            results.append((rotated, (label, float(score))))
            await asyncio.sleep(0)
        return results

    async def close(self) -> None:
        self._closed = True
        self._wakeup.set()
        if self._worker is not None:
            await self._worker
        if self._inflight:
            await asyncio.gather(*self._inflight, return_exceptions=True)
        self._executor.shutdown(wait=True)
        close_session = getattr(self._session, "close", None)
        if close_session is not None:
            close_session()
