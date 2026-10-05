from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from PIL import Image

COCO_IMAGE_ROOT = "https://s3.amazonaws.com/images.cocodataset.org/train2014"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "images" / "coco2000"
_local = threading.local()


def select_images(annotation: Path, count: int, seed: int) -> list[dict[str, Any]]:
    with annotation.open(encoding="utf-8") as stream:
        images = list(json.load(stream)["imgs"].values())
    if not 0 < count <= len(images):
        raise ValueError(f"Count must be between 1 and {len(images)}")
    rng = random.Random(seed)
    rng.shuffle(images)
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for image in images:
        groups[image["width"], image["height"]].append(image)
    groups_list = list(groups.values())
    rng.shuffle(groups_list)
    selected = [group.pop() for group in groups_list][:count]
    if len(selected) < count:
        remaining = [image for group in groups_list for image in group]
        rng.shuffle(remaining)
        selected.extend(remaining[: count - len(selected)])
    return selected


def validate(data: bytes, image: dict[str, Any]) -> None:
    with Image.open(io.BytesIO(data)) as decoded:
        if decoded.size != (image["width"], image["height"]):
            raise ValueError(
                f"Image dimensions differ from annotation: {image['file_name']}"
            )
        decoded.verify()


def download(image: dict[str, Any], output: Path) -> dict[str, Any]:
    filename = image["file_name"]
    if Path(filename).name != filename:
        raise ValueError(f"Invalid filename: {filename}")
    url = f"{COCO_IMAGE_ROOT}/{filename}"
    path = output / filename
    if path.exists():
        data = path.read_bytes()
        try:
            validate(data, image)
        except (OSError, ValueError):
            data = b""
    else:
        data = b""
    if not data:
        if not hasattr(_local, "session"):
            _local.session = requests.Session()
        for attempt in range(4):
            try:
                response = _local.session.get(url, timeout=(10, 45))
                response.raise_for_status()
                data = response.content
                validate(data, image)
                temporary = path.with_suffix(".jpg.part")
                temporary.write_bytes(data)
                temporary.replace(path)
                break
            except (requests.RequestException, OSError, ValueError):
                if attempt == 3:
                    raise
                time.sleep(2**attempt)
    return {
        **image,
        "url": url,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download COCO-Text image samples covering distinct dimensions."
    )
    parser.add_argument("annotation", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    images = select_images(args.annotation, args.count, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    errors = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download, image, args.output): image for image in images
        }
        for completed, future in enumerate(as_completed(futures), 1):
            try:
                records.append(future.result())
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {"image": futures[future]["file_name"], "error": str(exc)}
                )
            if completed % 100 == 0 or completed == len(images):
                print(
                    f"{completed}/{len(images)} completed, {len(errors)} failures",
                    flush=True,
                )
    manifest = {
        "annotation": str(args.annotation),
        "seed": args.seed,
        "requested": args.count,
        "seconds": time.perf_counter() - started,
        "distinct_dimensions": len({(r["width"], r["height"]) for r in records}),
        "images": sorted(records, key=lambda r: r["file_name"]),
        "errors": errors,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"Downloaded {len(records)} images, {manifest['distinct_dimensions']} distinct dimensions to {args.output}"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
