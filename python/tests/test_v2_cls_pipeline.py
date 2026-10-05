import asyncio
import threading
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from omegaconf import OmegaConf

from rapidocr.utils.typings import OCRVersion
from rapidocr.v2.cls_engine import ClsEngine


class TestClsEngine(unittest.TestCase):
    def test_cross_request_batching_rotation_and_fixed_shape(self) -> None:
        async def run() -> None:
            calls = []
            threads = []
            main_thread = threading.get_ident()

            def session(tensor: np.ndarray) -> np.ndarray:
                calls.append(tensor.shape)
                threads.append(threading.get_ident())
                labels = np.tile([0.99, 0.01], (len(tensor), 1))
                if tensor[0].any():
                    labels[0] = [0.01, 0.99]
                return labels

            cfg = OmegaConf.create(
                {
                    "ocr_version": OCRVersion.PPOCRV4,
                    "engine_type": "fake",
                    "cls_thresh": 0.9,
                    "label_list": ["0", "180"],
                }
            )
            with patch(
                "rapidocr.v2.cls_engine.core.get_engine",
                return_value=lambda cfg: session,
            ):
                pipeline = ClsEngine(cfg, batch_size=4, max_wait=0.005)
            await pipeline.start()
            image = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
            try:
                left, right = await asyncio.wait_for(
                    asyncio.gather(
                        pipeline.classify([image]), pipeline.classify([image.copy()])
                    ),
                    2,
                )
                assert len(calls) >= 2  # Warmup plus inference batches.
                assert all(shape == (4, 3, 48, 192) for shape in calls)
                assert all(thread != main_thread for thread in threads)
                assert left.cls_res[0][0] == "180"
                assert right.cls_res[0][0] == "0"
                np.testing.assert_array_equal(
                    left.img_list[0], cv2.rotate(image, cv2.ROTATE_180)
                )
                np.testing.assert_array_equal(right.img_list[0], image)
                np.testing.assert_array_equal(
                    image, np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
                )
                empty = await pipeline.classify([])
                assert empty.img_list == [] and empty.cls_res == []
            finally:
                await pipeline.close()

        asyncio.run(run())
