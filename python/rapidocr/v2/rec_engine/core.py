from __future__ import annotations

import asyncio
import math
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, cast

import cv2
import numpy as np

from ...ch_ppocr_rec.typings import TextRecOutput, WordInfo
from ...ch_ppocr_rec.utils import CTCLabelDecode
from ...inference_engine.base import FileInfo, InferSession, get_engine
from ...utils.download_file import DownloadFile, DownloadFileInput
from ...utils.log import logger
from ...utils.model_resolver import normalize_lang
from ...utils.typings import LangRec
from ...utils.utils import reorder_bidi_for_display, validate_rtl_dependency
from ..typing import HWCImage

if TYPE_CHECKING:
    import sys
    from collections.abc import Callable, Iterable

    from omegaconf import DictConfig

    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias

RecLine: TypeAlias = tuple[str, float]
RecResult: TypeAlias = tuple[RecLine, WordInfo | None]
RecQueueItem: TypeAlias = tuple[float, HWCImage, "asyncio.Future[RecResult]"]


def generate_rec_buckets(max_width: int, count: int) -> tuple[int, ...]:
    """Generate count evenly spaced positive integer widths ending at max_width."""
    if type(max_width) is not int or type(count) is not int:
        raise ValueError("Maximum width and count must be integers")
    if max_width <= 0 or count <= 0:
        raise ValueError("Maximum width and count must be positive")
    if max_width % count:
        raise ValueError("Maximum width must be divisible by bucket count")
    step = max_width // count
    return tuple(range(step, max_width + 1, step))


class RecEngine:
    def __init__(
        self,
        cfg: DictConfig,
        buckets: Iterable[int],
        batch_size: int = 32,
        max_wait: float = 0.02,
        return_word_box: bool = False,
        concurrency: int = 1,
    ) -> None:
        if (
            type(batch_size) is not int
            or batch_size < 1
            or batch_size & (batch_size - 1)
        ):
            raise ValueError("Recognition batch size must be a power of two")
        if concurrency < 1:
            raise ValueError("Recognition concurrency must be positive")
        self._is_arabic = normalize_lang(cfg.lang_type) == LangRec.ARABIC.value
        self._rec_keys_path: str | Path | None = cfg.get("rec_keys_path")
        self._model_root_dir = Path(
            cfg.get("model_root_dir") or Path(__file__).resolve().parents[2] / "models"
        )
        self._file_info = FileInfo(
            engine_type=cfg.engine_type,
            ocr_version=cfg.ocr_version,
            task_type=cfg.task_type,
            lang_type=cfg.lang_type,
            model_type=cfg.model_type,
        )
        if self._is_arabic:
            validate_rtl_dependency()
        factory = cast(
            "Callable[[DictConfig], InferSession]", get_engine(cfg.engine_type)
        )
        self._session = factory(cfg)
        self._return_word_box = return_word_box
        self._widths = tuple(sorted(set(buckets)))
        self._batch_size = batch_size
        self._max_wait = max_wait
        self._queues: dict[int, deque[RecQueueItem]] = {
            width: deque() for width in self._widths
        }
        self._wakeup = asyncio.Event()
        self._concurrency = concurrency
        self._inflight: set[asyncio.Task[None]] = set()
        self._executor = ThreadPoolExecutor(max_workers=concurrency)
        self._worker: asyncio.Task[None] | None = None
        self._closed = False
        self._img_shape: tuple[int, int, int] = tuple(cfg.rec_img_shape)
        self._postprocess: CTCLabelDecode | None = None
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
        character, path = self._load_characters()
        self._postprocess = CTCLabelDecode(character=character, character_path=path)
        c, h, _ = self._img_shape
        batch_sizes = tuple(
            1 << exponent for exponent in range(self._batch_size.bit_length())
        )
        for w in self._widths:
            for batch_size in batch_sizes:
                await loop.run_in_executor(
                    self._executor,
                    self._session,
                    np.zeros((batch_size, c, h, w), np.float32),
                )
        self._worker = loop.create_task(self._run_batches())

    def _load_characters(self) -> tuple[list[str] | None, str | Path | None]:
        assert self._session is not None
        path = self._rec_keys_path
        if self._session.have_key():
            return self._session.get_character_list(), path
        if path and Path(path).exists():
            return None, path
        url = self._session.get_dict_key_url(self._file_info) or (
            "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v2.0.7/"
            "paddle/PP-OCRv4/rec/ch_PP-OCRv4_rec_infer/ppocr_keys_v1.txt"
        )
        path = self._model_root_dir / Path(url).name
        if not path.exists():
            DownloadFile.run(
                DownloadFileInput(
                    file_url=url, sha256=None, save_path=path, logger=logger
                )
            )
        return None, path

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
            now = loop.time()
            selected: int | None = None
            oldest = float("inf")
            deadline = float("inf")
            for width, queue in self._queues.items():
                while queue and queue[0][2].cancelled():
                    queue.popleft()
                if not queue:
                    continue
                arrived = queue[0][0]
                expires = arrived + self._max_wait
                deadline = min(deadline, expires)
                ready = self._closed or expires <= now
                if ready or len(queue) >= self._batch_size:
                    limit = min(len(queue), self._batch_size)
                    if any(entry[2].cancelled() for entry in islice(queue, limit)):
                        active: list[RecQueueItem] = []
                        while queue and len(active) < limit:
                            entry = queue.popleft()
                            if not entry[2].cancelled():
                                active.append(entry)
                        queue.extendleft(reversed(active))
                    if not ready:
                        ready = len(queue) >= self._batch_size
                if ready and arrived < oldest:
                    selected, oldest = width, arrived
            if selected is None:
                if deadline == float("inf"):
                    if self._closed:
                        return
                    await self._wakeup.wait()
                    continue
                try:
                    await asyncio.wait_for(self._wakeup.wait(), deadline - now)
                except asyncio.TimeoutError:
                    pass
                continue
            queue = self._queues[selected]
            batch_size = 1 << (min(len(queue), self._batch_size).bit_length() - 1)
            pending = [queue.popleft() for _ in range(batch_size)]
            task = asyncio.create_task(self._infer_and_resolve(selected, pending))
            self._inflight.add(task)

    async def _infer_and_resolve(self, width: int, pending: list[RecQueueItem]) -> None:
        try:
            results = await asyncio.get_running_loop().run_in_executor(
                self._executor, self._infer, width, [image for _, image, _ in pending]
            )
            for (_, _, future), result in zip(pending, results):
                if not future.done():
                    future.set_result(result)
        except Exception as exc:  # noqa: BLE001
            for _, _, future in pending:
                if not future.done():
                    future.set_exception(exc)

    async def recognize(self, images: list[HWCImage]) -> TextRecOutput:
        assert self._session is not None
        height = self._img_shape[1]
        futures: list[asyncio.Future[RecResult]] = []
        for image in images:
            width = math.ceil(height * image.shape[1] / image.shape[0])
            bucket = next((w for w in self._widths if w >= width), self._widths[-1])
            if self._closed:
                raise RuntimeError("Rec engine is closed")
            loop = asyncio.get_running_loop()
            future: asyncio.Future[RecResult] = loop.create_future()
            self._queues[bucket].append((loop.time(), image, future))
            self._wakeup.set()
            futures.append(future)
        results = await asyncio.gather(*futures)
        lines, words = zip(*results)
        texts, scores = zip(*lines)
        if self._is_arabic:
            texts = reorder_bidi_for_display(texts)
        return TextRecOutput(
            images,
            cast("tuple[str]", tuple(str(text) for text in texts)),
            list(scores),
            tuple(words),
            elapse=0.0,
        )

    def _infer(
        self,
        width: int,
        images: list[HWCImage],
    ) -> list[RecResult]:
        assert self._session is not None and self._postprocess is not None
        self._batch_count += 1
        self._sample_count += len(images)
        c, h, _ = self._img_shape
        tensor = np.zeros((len(images), c, h, width), np.float32)
        ratios = [img.shape[1] / img.shape[0] for img in images]
        max_ratio = width / h
        for i, image in enumerate(images):
            resized_width = min(width, math.ceil(h * ratios[i]))
            resized = cv2.resize(image, (resized_width, h)).astype(np.float32)
            tensor[i, :, :, :resized_width] = (
                resized.transpose(2, 0, 1) / 255 - 0.5
            ) / 0.5
        preds = self._session(tensor)[: len(images)]
        lines, words = self._postprocess(
            preds,
            self._return_word_box,
            wh_ratio_list=ratios,
            max_wh_ratio=max_ratio,
        )
        return [
            (line, words[i] if self._return_word_box else None)
            for i, line in enumerate(lines)
        ]

    async def close(self) -> None:
        self._closed = True
        self._wakeup.set()
        if self._worker is not None:
            await self._worker
        if self._inflight:
            await asyncio.gather(*self._inflight, return_exceptions=True)
        self._executor.shutdown(wait=True)
