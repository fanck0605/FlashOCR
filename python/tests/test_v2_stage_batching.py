from __future__ import annotations

import asyncio
import threading
import unittest
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
from omegaconf import OmegaConf

from rapidocr.utils.typings import OCRVersion
from rapidocr.v2.cls_pipeline import ClsPipeline
from rapidocr.v2.rec_pipeline import RecPipeline


class TestStageBatching(unittest.IsolatedAsyncioTestCase):
    async def test_cls_full_batch_wakes_before_deadline(self) -> None:
        cfg = OmegaConf.create(
            {
                "ocr_version": OCRVersion.PPOCRV4,
                "engine_type": "fake",
                "cls_thresh": 0.9,
                "label_list": ["0", "180"],
            }
        )
        with patch("rapidocr.v2.cls_pipeline.core.get_engine", return_value=Mock()):
            pipeline = ClsPipeline(cfg, batch_size=2, max_wait=10)
        self.assertFalse(hasattr(pipeline, "_cfg"))
        image = np.zeros((8, 8, 3), np.uint8)
        pipeline._infer = AsyncMock(return_value=[(image, ("0", 1.0))] * 2)
        pipeline._worker = asyncio.create_task(pipeline._run_batches())
        first = asyncio.create_task(pipeline.classify([image]))
        try:
            await asyncio.sleep(0.01)
            second = asyncio.create_task(pipeline.classify([image]))
            await asyncio.wait_for(asyncio.gather(first, second), 2)
            self.assertEqual(pipeline._infer.await_count, 1)
        finally:
            await asyncio.wait_for(pipeline.close(), 2)

    async def test_concurrency_and_close_drain(self) -> None:
        for stage in ("cls", "rec"):
            with self.subTest(stage=stage):
                await self.check_concurrency_and_close(stage)

    async def check_concurrency_and_close(self, stage: str) -> None:
        cfg = OmegaConf.create(
            {
                "ocr_version": OCRVersion.PPOCRV4,
                "engine_type": "fake",
                "lang_type": "ch",
                "task_type": "rec",
                "model_type": "mobile",
                "rec_img_shape": [3, 48, 320],
                "cls_thresh": 0.9,
                "label_list": ["0", "180"],
            }
        )
        if stage == "cls":
            with patch("rapidocr.v2.cls_pipeline.core.get_engine", return_value=Mock()):
                pipeline = ClsPipeline(cfg, batch_size=2, max_wait=10, concurrency=2)
        else:
            with patch("rapidocr.v2.rec_pipeline.core.get_engine", return_value=Mock()):
                pipeline = RecPipeline(
                    cfg, buckets=[32, 64], batch_size=2, max_wait=10, concurrency=2
                )
        self.assertFalse(hasattr(pipeline, "_cfg"))
        loop = asyncio.get_running_loop()
        release = threading.Event()
        saturated = asyncio.Event()
        lock = threading.Lock()
        running = 0
        peak = 0

        def infer(*args):
            nonlocal running, peak
            with lock:
                running += 1
                peak = max(peak, running)
                if running == 2:
                    loop.call_soon_threadsafe(saturated.set)
            if not release.wait(2):
                raise TimeoutError("Inference was not released")
            with lock:
                running -= 1
            if stage == "cls":
                return [(image, ("0", 1.0)) for image in args[0]]
            return [((str(int(image[0, 0, 0])), 1.0), None) for image in args[1]]

        if stage == "cls":

            async def cls_infer(images):
                return await loop.run_in_executor(pipeline._executor, infer, images)

            pipeline._infer = cls_infer
        else:
            pipeline._infer = Mock(side_effect=infer)
        futures = []
        for i in range(9):
            image = np.full((8, 8, 3), i, np.uint8)
            future = loop.create_future()
            futures.append(future)
            queue = (
                pipeline._queue
                if stage == "cls"
                else pipeline._queues[32 if i < 4 else 64]
            )
            queue.append((loop.time(), image, future))
        futures[-1].cancel()
        pipeline._worker = asyncio.create_task(pipeline._run_batches())
        try:
            await asyncio.wait_for(saturated.wait(), 2)
            closing = asyncio.create_task(pipeline.close())
            await asyncio.sleep(0)
            release.set()
            await asyncio.wait_for(closing, 2)
            results = await asyncio.wait_for(asyncio.gather(*futures[:-1]), 2)
            self.assertEqual(peak, 2)
            if stage == "rec":
                self.assertEqual(
                    [result[0][0] for result in results],
                    list(map(str, range(8))),
                )
                self.assertTrue(all(not queue for queue in pipeline._queues.values()))
            else:
                self.assertFalse(pipeline._queue)
        finally:
            release.set()
            await asyncio.wait_for(pipeline.close(), 2)
