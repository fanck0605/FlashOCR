from __future__ import annotations

import asyncio
import math
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from omegaconf import DictConfig

from ..ch_ppocr_cls.main import CLS_SHAPE_BY_OCR_VERSION
from ..ch_ppocr_cls.utils import ClsPostProcess, TextClsOutput
from ..inference_engine.base import InferSession, get_engine

ClsResult = tuple[np.ndarray, tuple[str, float]]
ClsQueueItem = tuple[float, np.ndarray, "asyncio.Future[ClsResult]"]


class ClsPipeline:
    """Cross-request classification with one fixed input and batch shape."""

    def __init__(
        self, cfg: DictConfig, batch_size: int = 16, max_wait_ms: float = 3
    ) -> None:
        if batch_size < 1 or max_wait_ms < 0:
            raise ValueError("Invalid CLS batch size or wait time")
        self._cfg = cfg
        self._shape = tuple(CLS_SHAPE_BY_OCR_VERSION[cfg.ocr_version])
        self._threshold: float = cfg.cls_thresh
        self._postprocess = ClsPostProcess(cfg.label_list)
        self._batch_size = batch_size
        self._max_wait = max_wait_ms / 1000
        self._queue: deque[ClsQueueItem] = deque()
        self._wakeup = asyncio.Event()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._worker: asyncio.Task[None] | None = None
        self._session: InferSession | None = None
        self._closed = False

    async def start(self, warmup: bool = True) -> None:
        self._session = get_engine(self._cfg.engine_type)(self._cfg)
        loop = asyncio.get_running_loop()
        if warmup:
            await loop.run_in_executor(
                self._executor,
                self._session,
                np.zeros((self._batch_size, *self._shape), np.float32),
            )
        self._worker = loop.create_task(self._run_batches())

    async def classify(self, images: list[np.ndarray]) -> TextClsOutput:
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
            self._wakeup.clear()
            if not self._queue:
                if self._closed:
                    return
                await self._wakeup.wait()
                continue
            if self._closed:
                self._max_wait = 0
            delay = max(0.0, self._queue[0][0] + self._max_wait - loop.time())
            if not self._closed and len(self._queue) < self._batch_size and delay > 0:
                await asyncio.sleep(delay)
                continue
            batch = [
                self._queue.popleft()
                for _ in range(min(len(self._queue), self._batch_size))
            ]
            pending = [
                (image, future) for _, image, future in batch if not future.cancelled()
            ]
            if not pending:
                continue
            try:
                results = await self._infer([image for image, _ in pending])
            except Exception as exc:  # noqa: BLE001
                # Propagate backend failures to every request in this batch.
                for _, future in pending:
                    if not future.done():
                        future.set_exception(exc)
                continue
            for (_, future), result in zip(pending, results):
                if not future.done():
                    future.set_result(result)

    def _prepare(self, image: np.ndarray) -> np.ndarray:
        channels, height, width = self._shape
        resized_width = min(width, math.ceil(height * image.shape[1] / image.shape[0]))
        resized = cv2.resize(image, (resized_width, height)).astype(np.float32)
        normalized = (resized.transpose(2, 0, 1) / 255 - 0.5) / 0.5
        padded = np.zeros((channels, height, width), np.float32)
        padded[:, :, :resized_width] = normalized
        return padded

    async def _infer(self, images: list[np.ndarray]) -> list[ClsResult]:
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
                cv2.rotate(image, cv2.ROTATE_180)
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
        self._executor.shutdown(wait=True)
