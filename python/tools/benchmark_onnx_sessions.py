from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rapidocr.v2 import FlashOCR


class CaptureSession:
    def __init__(self, session: Any) -> None:
        self.session = session
        self.tensors: list[np.ndarray[Any, np.dtype[np.float32]]] = []

    def __call__(self, tensor: np.ndarray[Any, np.dtype[np.float32]]) -> Any:
        self.tensors.append(tensor.copy())
        return self.session(tensor)


def replay(capture: CaptureSession, iterations: int) -> dict[str, Any]:
    ort_session: ort.InferenceSession = capture.session.session
    input_name = ort_session.get_inputs()[0].name
    output_names = [output.name for output in ort_session.get_outputs()]
    input_feeds = [{input_name: tensor} for tensor in capture.tensors]
    for input_feed in input_feeds:
        ort_session.run(output_names, input_feed)
    started = time.perf_counter()
    for _ in range(iterations):
        for input_feed in input_feeds:
            ort_session.run(output_names, input_feed)
    elapsed = time.perf_counter() - started
    return {
        "elapsed_s": elapsed,
        "image_workloads_s": iterations / elapsed,
        "session_calls_s": iterations * len(capture.tensors) / elapsed,
        "samples_s": iterations * sum(t.shape[0] for t in capture.tensors) / elapsed,
        "ms_per_image_workload": elapsed * 1000 / iterations,
    }


async def run(args: argparse.Namespace) -> None:
    ort.preload_dlls()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ocr = FlashOCR(params={"EngineConfig.onnxruntime.use_cuda": True})
    captures: dict[str, CaptureSession] = {}
    report: dict[str, Any] = {
        "image": str(args.image),
        "iterations": args.iterations,
        "timed_call": "onnxruntime.InferenceSession.run",
    }
    try:
        await ocr.start()
        for name in ("det", "cls", "rec"):
            engine = getattr(ocr._engine, f"_{name}_engine")
            session = engine._session
            if "CUDAExecutionProvider" not in session.session.get_providers():
                raise RuntimeError(f"{name}: CUDA provider unavailable")
            captures[name] = CaptureSession(session)
            engine._session = captures[name]
        result = await ocr(args.image)
        report["texts"] = list(result.txts or ())
        if not report["texts"] or any(not c.tensors for c in captures.values()):
            raise RuntimeError("Image did not produce tensors for all three stages")
        report["shapes"] = {}
        for name, capture in captures.items():
            report["shapes"][name] = [list(t.shape) for t in capture.tensors]
            for index, tensor in enumerate(capture.tensors):
                np.save(args.output_dir / f"{name}_{index}.npy", tensor)
                capture.session(tensor)
        print("Captured:", json.dumps(report["shapes"]), flush=True)
        if args.capture_only:
            manifest = {
                name: {
                    "model_path": str(
                        Path(capture.session.session._model_path).resolve()
                    ),
                    "tensors": [
                        f"{name}_{index}.npy" for index in range(len(capture.tensors))
                    ],
                }
                for name, capture in captures.items()
            }
            (args.output_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            return
        report["isolated"] = {}
        for name, capture in captures.items():
            report["isolated"][name] = replay(capture, args.iterations)
            print(name, json.dumps(report["isolated"][name]), flush=True)
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                name: executor.submit(replay, capture, args.iterations)
                for name, capture in captures.items()
            }
            report["concurrent"] = {
                name: future.result() for name, future in futures.items()
            }
        elapsed = time.perf_counter() - started
        report["concurrent_wall_s"] = elapsed
        report["concurrent_image_workloads_s"] = args.iterations / elapsed
        print("Concurrent:", json.dumps(report["concurrent"]), flush=True)
        print("Concurrent image workloads/s:", args.iterations / elapsed, flush=True)
    finally:
        await ocr.close()
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture and replay OCR ONNX input tensors."
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--capture-only", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/tmp/ocr-session-benchmark")
    )
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
