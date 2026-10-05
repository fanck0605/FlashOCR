from __future__ import annotations

import argparse
import asyncio
import json
import math
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

import onnxruntime as ort

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

from rapidocr import RapidOCR
from rapidocr.v2 import FlashOCR
from rapidocr.v2.rec_engine import generate_rec_buckets
from rapidocr.v2.session.onnx import OnnxSession

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

DEFAULT_IMAGE_DIR = PYTHON_ROOT / "tests" / "test_files"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class GPUMemorySampler:
    """Sample whole-device memory; WSL does not reliably expose process memory."""

    def __init__(self) -> None:
        self._phase = "baseline"
        self._samples: dict[str, list[int]] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._error: Exception | None = None
        self.baseline = self._read()
        self._thread.start()

    @staticmethod
    def _read() -> int:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--id=0",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return int(result.stdout.strip())

    def _sample(self) -> None:
        try:
            while not self._stop.is_set():
                with self._lock:
                    phase = self._phase
                used = self._read()
                with self._lock:
                    self._samples.setdefault(phase, []).append(used)
                self._stop.wait(0.2)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            self._error = exc

    def phase(self, name: str) -> None:
        used = self._read()
        with self._lock:
            self._samples.setdefault(self._phase, []).append(used)
            self._phase = name
            self._samples.setdefault(name, []).append(used)

    def finish(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join()
        if self._error is not None:
            raise self._error
        return {
            "scope": "GPU 0 whole-device memory, including other processes",
            "sample_interval_s": 0.2,
            "baseline_mib": self.baseline,
            "phases": {
                name: {
                    "samples": len(values),
                    "min_mib": min(values),
                    "peak_mib": max(values),
                    "mean_mib": sum(values) / len(values),
                    "end_mib": values[-1],
                    "peak_above_baseline_mib": max(values) - self.baseline,
                }
                for name, values in self._samples.items()
            },
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark RapidOCR and FlashOCR in separate processes."
    )
    parser.add_argument("image_dir", nargs="?", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--det-concurrency", type=int, default=1)
    parser.add_argument("--cls-concurrency", type=int, default=1)
    parser.add_argument("--rec-concurrency", type=int, default=1)
    parser.add_argument(
        "--max-images", type=int, default=0, help="Zero means all images."
    )
    parser.add_argument(
        "--repeat", type=int, default=1, help="Repeat the first selected image."
    )
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--max-wait", type=float, default=0.02)
    parser.add_argument("--cls-batch-size", type=int, default=16)
    parser.add_argument("--rec-batch-size", type=int, default=32)
    parser.add_argument("--rec-max-width", type=int, default=3840)
    parser.add_argument("--rec-bucket-count", type=int, default=8)
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--engine", choices=("both", "rapid", "flash"), default="both")
    parser.add_argument("--output", type=Path, default=Path("benchmark_ocr.json"))
    parser.add_argument(
        "--order", choices=("rapid-first", "flash-first"), default="rapid-first"
    )
    args = parser.parse_args()
    if (
        args.concurrency < 1
        or args.det_concurrency < 1
        or args.cls_concurrency < 1
        or args.rec_concurrency < 1
        or args.rounds < 1
        or args.max_images < 0
        or args.repeat < 1
        or args.max_wait < 0
        or args.cls_batch_size < 1
        or args.rec_batch_size < 1
        or args.rec_max_width < 1
        or args.rec_bucket_count < 1
    ):
        parser.error(
            "Concurrency, rounds, and repeat must be positive; max-images must be nonnegative"
        )
    return args


def find_images(image_dir: Path, max_images: int) -> list[Path]:
    images = sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    return images[:max_images] if max_images else images


def output_to_dict(result: Any) -> dict[str, Any]:
    return {
        "texts": list(result.txts) if result.txts is not None else [],
        "scores": list(result.scores) if result.scores is not None else [],
        "boxes": result.boxes.tolist() if result.boxes is not None else None,
    }


def check_providers(sessions: list[Any], cuda: bool) -> list[list[str]]:
    providers = [session.session.get_providers() for session in sessions]
    if cuda and any("CUDAExecutionProvider" not in names for names in providers):
        raise RuntimeError(f"CUDA requested but provider fell back: {providers}")
    return providers


async def benchmark_round(
    name: str,
    round_number: int,
    images: list[Path],
    concurrency: int,
    recognize: Callable[[Path], Awaitable[Any]],
) -> dict[str, Any]:
    paths = iter(enumerate(images))
    results: list[dict[str, Any]] = [{} for _ in images]
    completed = 0
    started = time.perf_counter()

    async def worker() -> None:
        nonlocal completed
        for index, path in paths:
            request_start = time.perf_counter()
            row: dict[str, Any]
            try:
                result = await recognize(path)
                row = output_to_dict(result)
            except Exception as exc:  # noqa: BLE001
                row = {"error": str(exc)}
            row.update(image=str(path), latency_s=time.perf_counter() - request_start)
            results[index] = row
            completed += 1
            if completed % 100 == 0 or completed == len(images):
                print(
                    f"{name} round {round_number}: {completed}/{len(images)}",
                    flush=True,
                )

    await asyncio.gather(*(worker() for _ in range(min(concurrency, len(images)))))
    elapsed = time.perf_counter() - started
    latencies = sorted(row["latency_s"] for row in results)
    summary = {
        "round": round_number,
        "images": len(images),
        "elapsed_s": elapsed,
        "throughput_images_s": len(images) / elapsed,
        "mean_request_latency_s": sum(latencies) / len(latencies),
        "p50_request_latency_s": latencies[math.ceil(len(latencies) * 0.5) - 1],
        "p95_request_latency_s": latencies[math.ceil(len(latencies) * 0.95) - 1],
        "errors": sum("error" in row for row in results),
        "text_lines": sum(len(row.get("texts", [])) for row in results),
        "empty_results": sum(
            not row.get("texts") and "error" not in row for row in results
        ),
    }
    print(json.dumps({"engine": name, **summary}), flush=True)
    return {"summary": summary, "results": results}


async def benchmark_rapid(
    images: list[Path],
    args: argparse.Namespace,
    params: dict[str, Any],
    memory: GPUMemorySampler | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    ocr = RapidOCR(params=params)
    # Load weights without inference so the first round sees new input shapes.
    ocr._load_det_model()
    ocr._load_cls_model()
    ocr._load_rec_model()
    assert (
        ocr.text_det is not None
        and ocr.text_cls is not None
        and ocr.text_rec is not None
    )
    init = time.perf_counter() - started
    if memory is not None:
        memory.phase("initialized")
    providers = check_providers(
        [ocr.text_det.session, ocr.text_cls.session, ocr.text_rec.session], args.cuda
    )
    loop = asyncio.get_running_loop()
    rounds = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:

        async def recognize(path: Path) -> Any:
            return await loop.run_in_executor(executor, ocr, path)

        for number in range(1, args.rounds + 1):
            if memory is not None:
                memory.phase(f"round_{number}")
            rounds.append(
                await benchmark_round(
                    "RapidOCR", number, images, args.concurrency, recognize
                )
            )
        if memory is not None:
            memory.phase("after_rounds")
    return {"init_s": init, "warmup_s": 0.0, "providers": providers, "rounds": rounds}


async def benchmark_flash(
    images: list[Path],
    args: argparse.Namespace,
    params: dict[str, Any],
    memory: GPUMemorySampler | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    if args.disable_cuda_graph:
        from rapidocr.inference_engine.onnxruntime import OrtInferSession

        for stage in ("det", "cls", "rec"):
            module = import_module(f"rapidocr.v2.{stage}_engine.core")
            module.__dict__["create_session"] = OrtInferSession
    ocr = FlashOCR(
        params=params,
        det_concurrency=args.det_concurrency,
        cls_concurrency=args.cls_concurrency,
        rec_concurrency=args.rec_concurrency,
        cls_batch_size=args.cls_batch_size,
        rec_batch_size=args.rec_batch_size,
        rec_buckets=generate_rec_buckets(args.rec_max_width, args.rec_bucket_count),
        max_wait=args.max_wait,
    )
    init = time.perf_counter() - started
    if memory is not None:
        memory.phase("warmup")
    rounds = []
    try:
        started = time.perf_counter()
        await ocr.start()
        warmup = time.perf_counter() - started
        if memory is not None:
            memory.phase("warmed")
        pipeline = ocr._engine
        assert pipeline._det_engine is not None
        assert pipeline._cls_engine is not None
        assert pipeline._rec_engine is not None
        providers = check_providers(
            [
                pipeline._det_engine._session,
                pipeline._cls_engine._session,
                pipeline._rec_engine._session,
            ],
            args.cuda,
        )
        for number in range(1, args.rounds + 1):
            if memory is not None:
                memory.phase(f"round_{number}")
            rounds.append(
                await benchmark_round("FlashOCR", number, images, args.concurrency, ocr)
            )
        if memory is not None:
            memory.phase("after_rounds")
        batch_stats = {
            "det": pipeline._det_engine.batch_stats(),
            "cls": pipeline._cls_engine.batch_stats(),
            "rec": pipeline._rec_engine.batch_stats(),
        }
        graph_shapes = {
            name: engine._session.graph_shapes()
            for name, engine in (
                ("det", pipeline._det_engine),
                ("cls", pipeline._cls_engine),
                ("rec", pipeline._rec_engine),
            )
            if isinstance(engine._session, OnnxSession)
        }
    finally:
        await ocr.close()
        if memory is not None:
            memory.phase("closed")
    return {
        "init_s": init,
        "warmup_s": warmup,
        "providers": providers,
        "rounds": rounds,
        "batch_stats": batch_stats,
        "graph_shapes": graph_shapes,
        "cuda_graph": not args.disable_cuda_graph,
    }


async def run(args: argparse.Namespace) -> int:
    images = find_images(args.image_dir, args.max_images)
    if not images:
        raise ValueError(f"No images found in {args.image_dir}")
    if args.repeat > 1:
        images = [images[0]] * args.repeat
    params = {
        "EngineConfig.onnxruntime.use_cuda": args.cuda,
        "Global.log_level": "error",
    }
    report: dict[str, Any] = {
        "image_dir": str(args.image_dir.resolve()),
        "images": len(images),
        "concurrency": args.concurrency,
        "stage_concurrency": {
            "det": args.det_concurrency,
            "cls": args.cls_concurrency,
            "rec": args.rec_concurrency,
        },
        "cuda": args.cuda,
        "order": args.order,
        "process_mode": "independent",
        "engines": {},
    }
    engines = [("RapidOCR", benchmark_rapid), ("FlashOCR", benchmark_flash)]
    if args.order == "flash-first":
        engines.reverse()
    if args.engine != "both":
        ort.preload_dlls()
        memory = GPUMemorySampler() if args.cuda else None
        if memory is not None:
            memory.phase("initialization")
        selected = "RapidOCR" if args.engine == "rapid" else "FlashOCR"
        name, benchmark = next(entry for entry in engines if entry[0] == selected)
        try:
            result = await benchmark(images, args, params, memory)
        finally:
            memory_report = memory.finish() if memory is not None else None
        result["gpu_memory"] = memory_report
        report["engines"][name] = result
    else:
        with tempfile.TemporaryDirectory(prefix="ocr-benchmark-") as directory:
            for name, _ in engines:
                child_output = Path(directory) / f"{name}.json"
                command = [
                    sys.executable,
                    "-u",
                    str(Path(__file__).resolve()),
                    str(args.image_dir),
                    "--engine",
                    "rapid" if name == "RapidOCR" else "flash",
                    "--concurrency",
                    str(args.concurrency),
                    "--det-concurrency",
                    str(args.det_concurrency),
                    "--cls-concurrency",
                    str(args.cls_concurrency),
                    "--rec-concurrency",
                    str(args.rec_concurrency),
                    "--rounds",
                    str(args.rounds),
                    "--max-images",
                    str(args.max_images),
                    "--repeat",
                    str(args.repeat),
                    "--max-wait",
                    str(args.max_wait),
                    "--cls-batch-size",
                    str(args.cls_batch_size),
                    "--rec-batch-size",
                    str(args.rec_batch_size),
                    "--rec-max-width",
                    str(args.rec_max_width),
                    "--rec-bucket-count",
                    str(args.rec_bucket_count),
                    "--output",
                    str(child_output),
                ]
                if args.cuda:
                    command.append("--cuda")
                if args.disable_cuda_graph:
                    command.append("--disable-cuda-graph")
                child = await asyncio.create_subprocess_exec(*command)
                code = await child.wait()
                if not child_output.exists():
                    raise RuntimeError(
                        f"{name} benchmark exited with {code} without a report"
                    )
                report["engines"].update(
                    json.loads(child_output.read_text())["engines"]
                )
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Report: {args.output.resolve()}", flush=True)
    return int(
        any(
            round_["summary"]["errors"]
            for engine in report["engines"].values()
            for round_ in engine["rounds"]
        )
    )


def main() -> int:
    return asyncio.run(run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
