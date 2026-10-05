from __future__ import annotations

import asyncio
import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import numpy as np
import onnxruntime as ort
from omegaconf import OmegaConf

from rapidocr.v2 import FlashOCR
from rapidocr.v2.session.onnx import OnnxSession

ROOT = Path(__file__).resolve().parents[1]
CLS_MODEL = ROOT / "rapidocr/models/ch_ppocr_mobile_v2.0_cls_mobile.onnx"


def config(model: Path, cuda: bool):
    return OmegaConf.create(
        {
            "model_path": str(model),
            "engine_cfg": {
                "use_cuda": cuda,
                "intra_op_num_threads": 1,
                "cuda_ep_cfg": {"device_id": 0},
            },
        }
    )


@unittest.skipUnless(CLS_MODEL.is_file(), "Local classification model required")
class CpuSessionTests(unittest.TestCase):
    def test_cpu_and_close(self):
        session = OnnxSession(config(CLS_MODEL, False))
        tensor = np.zeros((1, 3, 48, 192), np.float32)
        self.assertEqual(session(tensor).shape, (1, 2))
        self.assertEqual(session.graph_shapes(), ())
        with self.assertRaises(ValueError):
            session(cast("Any", tensor.astype(np.float64)))
        session.close()
        session.close()
        with self.assertRaises(RuntimeError):
            session(tensor)


@unittest.skipUnless(
    os.environ.get("RAPIDOCR_TEST_CUDA_GRAPH") == "1",
    "Enable CUDA integration tests explicitly",
)
class CudaGraphTests(unittest.TestCase):
    def test_shapes_content_concurrency_and_ownership(self):
        ort.preload_dlls()
        rng = np.random.default_rng(42)
        for model, spatial in (
            (CLS_MODEL, (48, 192)),
            (ROOT / "rapidocr/models/PP-OCRv6_rec_small.onnx", (48, 480)),
            (ROOT / "rapidocr/models/PP-OCRv6_det_small.onnx", (160, 160)),
        ):
            with self.subTest(model=model.name):
                session = OnnxSession(config(model, True))
                reference = ort.InferenceSession(
                    str(model),
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )
                name = reference.get_inputs()[0].name
                tensors = [
                    rng.uniform(-1, 1, (batch, 3, *spatial)).astype(np.float32)
                    for batch in (2, 1, 2, 1)
                ]
                expected = [
                    reference.run(None, {name: tensor})[0] for tensor in tensors
                ]
                try:
                    first = session(tensors[0])
                    saved = first.copy()
                    with ThreadPoolExecutor(max_workers=3) as pool:
                        actual = list(pool.map(session, tensors))
                    for result, wanted in zip(actual, expected):
                        np.testing.assert_allclose(
                            result, cast("Any", wanted), rtol=1e-4, atol=1e-5
                        )
                    np.testing.assert_array_equal(first, saved)
                    self.assertEqual(
                        set(session.graph_shapes()), {tuple(t.shape) for t in tensors}
                    )
                finally:
                    session.close()
                self.assertEqual(session.graph_shapes(), ())
                del reference

    def test_complete_ocr(self):
        async def run():
            ocr = FlashOCR(
                params={"EngineConfig.onnxruntime.use_cuda": True},
                det_buckets=((480, 480),),
                rec_buckets=(480,),
                det_batch_size=1,
                cls_batch_size=2,
                rec_batch_size=2,
            )
            try:
                results = await ocr.batch([ROOT / "tests/test_files/ch_en_num.jpg"] * 3)
                self.assertTrue(results[0].txts)
                for result in results[1:]:
                    self.assertEqual(result.txts, results[0].txts)
                for stage in ("det", "cls", "rec"):
                    engine = getattr(ocr._engine, f"_{stage}_engine")
                    self.assertTrue(engine._session.graph_shapes())
            finally:
                await ocr.close()
            print("CUDA Graph OCR lines:", len(results[0].txts or ()))

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
