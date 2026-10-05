from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import onnxruntime as ort

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

from rapidocr.v2 import RapidOCRv2

DEFAULT_IMAGE_DIR = PYTHON_ROOT / "tests" / "test_files"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RapidOCRv2 concurrently over test image files."
    )
    parser.add_argument(
        "image_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help=f"Directory to scan. Default: {DEFAULT_IMAGE_DIR}",
    )
    parser.add_argument(
        "--concurrency", type=int, default=8, help="Maximum concurrent requests."
    )
    parser.add_argument(
        "--max-images", type=int, default=0, help="Limit images; zero means all."
    )
    return parser.parse_args()


def find_images(image_dir: Path, max_images: int) -> list[Path]:
    images = sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    return images if max_images <= 0 else images[:max_images]


def output_to_dict(result: Any) -> dict[str, Any]:
    return {
        "texts": list(result.txts) if result.txts is not None else [],
        "scores": list(result.scores) if result.scores is not None else [],
        "boxes": result.boxes.tolist() if result.boxes is not None else None,
        "elapsed": list(result.elapse_list or []),
    }


async def run(image_paths: list[Path], concurrency: int) -> int:
    if concurrency < 1:
        raise ValueError("--concurrency must be positive")
    if not image_paths:
        print("No images found.")
        return 0

    semaphore = asyncio.Semaphore(concurrency)
    ort.preload_dlls()
    async with RapidOCRv2() as ocr:

        async def recognize(path: Path) -> tuple[Path, dict[str, Any], float]:
            async with semaphore:
                started = time.perf_counter()
                result = await ocr(path)
                return path, output_to_dict(result), time.perf_counter() - started

        results = await asyncio.gather(*(recognize(path) for path in image_paths))

    for path, result, elapsed in results:
        print(
            json.dumps(
                {"image": str(path), "elapsed": elapsed, **result}, ensure_ascii=False
            )
        )
    print(
        json.dumps(
            {"images": len(results), "concurrency": concurrency}, ensure_ascii=False
        )
    )
    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(
        run(
            find_images(args.image_dir, args.max_images),
            args.concurrency,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
