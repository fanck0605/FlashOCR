from __future__ import annotations

import asyncio
import unittest
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
from omegaconf import OmegaConf

from rapidocr.ch_ppocr_det.utils import TextDetOutput
from rapidocr.v2.det_engine import DetEngine, DetResult, DetShape

if TYPE_CHECKING:
    from rapidocr.v2.typing import HWCImage


class TestDetBatching(unittest.IsolatedAsyncioTestCase):
    def make_pipeline(self, max_wait: float = 0.005) -> DetEngine:
        cfg = OmegaConf.create(
            {"limit_side_len": 960, "limit_type": "max", "engine_type": "fake"}
        )
        with patch("rapidocr.v2.det_engine.core.create_session", return_value=Mock()):
            pipeline = DetEngine(
                cfg,
                [(32, 32), (64, 64)],
                batch_size=2,
                max_wait=max_wait,
                concurrency=2,
            )
        self.assertFalse(hasattr(pipeline, "_cfg"))
        return pipeline

    async def test_full_and_partial_batches_across_buckets(self) -> None:
        pipeline = self.make_pipeline()

        async def infer(bucket: DetShape, images: list[HWCImage]) -> list[DetResult]:
            return [(TextDetOutput(img=image), []) for image in images]

        mock = AsyncMock(side_effect=infer)
        pipeline._infer = mock
        pipeline._worker = asyncio.create_task(pipeline._run_batches())
        images = [np.zeros((size, size, 3), np.uint8) for size in (32, 32, 64)]
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*(pipeline.detect(image) for image in images)), 2
            )
            self.assertEqual(mock.await_count, 2)
            batches = [
                (call.args[0], len(call.args[1])) for call in mock.await_args_list
            ]
            self.assertEqual(batches, [((32, 32), 2), ((64, 64), 1)])
            for image, (result, _) in zip(images, results):
                self.assertIs(result.img, image)
        finally:
            await asyncio.wait_for(pipeline.close(), 2)

    async def test_close_drains_queues_at_concurrency_limit(self) -> None:
        pipeline = self.make_pipeline(max_wait=10)
        release = asyncio.Event()
        saturated = asyncio.Event()
        running = 0
        peak = 0

        async def infer(bucket: DetShape, images: list[HWCImage]) -> list[DetResult]:
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            if running == 2:
                saturated.set()
            await release.wait()
            running -= 1
            return [(TextDetOutput(img=image), []) for image in images]

        pipeline._infer = AsyncMock(side_effect=infer)
        pipeline._worker = asyncio.create_task(pipeline._run_batches())
        image = np.zeros((32, 32, 3), np.uint8)
        requests = [asyncio.create_task(pipeline.detect(image)) for _ in range(9)]
        try:
            await asyncio.wait_for(saturated.wait(), 2)
            requests[-1].cancel()
            await asyncio.gather(requests[-1], return_exceptions=True)
            closing = asyncio.create_task(pipeline.close())
            await asyncio.sleep(0)
            release.set()
            await asyncio.wait_for(closing, 2)
            await asyncio.wait_for(asyncio.gather(*requests[:-1]), 2)
            self.assertEqual(peak, 2)
            self.assertTrue(all(not queue for queue in pipeline._queues.values()))
        finally:
            release.set()
            await asyncio.wait_for(pipeline.close(), 2)


if __name__ == "__main__":
    unittest.main()
