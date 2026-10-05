from __future__ import annotations

import asyncio
import math
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, cast

import cv2
import numpy as np

from ..ch_ppocr_cls.main import CLS_SHAPE_BY_OCR_VERSION
from ..ch_ppocr_cls.utils import ClsPostProcess, TextClsOutput
from ..inference_engine.base import InferSession, get_engine
from .typing import HWCImage

if TYPE_CHECKING:
    import sys
    from collections.abc import Callable

    from omegaconf import DictConfig

    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias

ClsResult: TypeAlias = tuple[HWCImage, tuple[str, float]]
ClsQueueItem: TypeAlias = tuple[float, HWCImage, "asyncio.Future[ClsResult]"]


class ClsPipeline:
    """Cross-request classification with one fixed input and batch shape."""

    def __init__(
        self,
        cfg: DictConfig,
        batch_size: int = 16,
        max_wait: float = 0.02,
        concurrency: int = 1,
    ) -> None:
        if batch_size < 1 or max_wait < 0:
            raise ValueError("Invalid CLS batch size or wait time")
        if concurrency < 1:
            raise ValueError("Classification concurrency must be positive")
        factory = cast(
            "Callable[[DictConfig], InferSession]", get_engine(cfg.engine_type)
        )
        self._session = factory(cfg)
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

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._executor,
            self._session,
            np.zeros((self._batch_size, *self._shape), np.float32),
        )
        self._worker = loop.create_task(self._run_batches())

    async def classify(self, images: list[HWCImage]) -> TextClsOutput:
        if self._closed:
            raise RuntimeError("CLS pipeline is closed")
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
            if not self._queue:
                if self._closed:
                    return
                await self._wakeup.wait()
                continue
            delay = max(0.0, self._queue[0][0] + self._max_wait - loop.time())
            if not self._closed and len(self._queue) < self._batch_size and delay > 0:
                try:
                    await asyncio.wait_for(self._wakeup.wait(), delay)
                except asyncio.TimeoutError:
                    pass
                continue
            pending: list[tuple[HWCImage, asyncio.Future[ClsResult]]] = []
            for _ in range(min(len(self._queue), self._batch_size)):
                _, image, future = self._queue.popleft()
                if not future.cancelled():
                    pending.append((image, future))
            if not pending:
                continue
            task = asyncio.create_task(self._infer_and_resolve(pending))
            self._inflight.add(task)

    async def _infer_and_resolve(
        self, pending: list[tuple[HWCImage, asyncio.Future[ClsResult]]]
    ) -> None:
        try:
            results = await self._infer([image for image, _ in pending])
            for (_, future), result in zip(pending, results):
                if not future.done():
                    future.set_result(result)
        except Exception as exc:  # noqa: BLE001
            for _, future in pending:
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
        tensor = np.zeros((self._batch_size, *self._shape), np.float32)
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
