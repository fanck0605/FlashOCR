from __future__ import annotations

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort


def gpu_memory() -> int:
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
    )
    return int(result.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay saved tensors with fresh native ORT sessions."
    )
    parser.add_argument("tensor_dir", type=Path)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--text-batch-size", type=int, default=0)
    parser.add_argument("--pad-text-batch-size", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--group-by-shape", action="store_true")
    parser.add_argument("--session-per-shape", action="store_true")
    parser.add_argument(
        "--conv-algo-search",
        choices=("EXHAUSTIVE", "HEURISTIC", "DEFAULT"),
        default="EXHAUSTIVE",
    )
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    if args.session_per_shape and args.profile:
        parser.error("session-per-shape profiling is not supported")
    if args.text_batch_size < 0:
        parser.error("text-batch-size must be nonnegative")
    if args.pad_text_batch_size < 0 or (
        args.pad_text_batch_size and args.text_batch_size
    ):
        parser.error(
            "padding size must be nonnegative and cannot be combined with text-batch-size"
        )
    ort.preload_dlls()
    manifest = json.loads((args.tensor_dir / "manifest.json").read_text())
    report: dict[str, Any] = {
        "baseline_gpu_mib": gpu_memory(),
        "iterations": args.iterations,
        "text_batch_size": args.text_batch_size,
        "pad_text_batch_size": args.pad_text_batch_size,
        "conv_algo_search": args.conv_algo_search,
        "group_by_shape": args.group_by_shape,
        "session_per_shape": args.session_per_shape,
    }
    workloads: dict[str, Any] = {}
    for name, entry in manifest.items():
        options = ort.SessionOptions()
        options.log_severity_level = 4
        options.enable_profiling = args.profile
        options.profile_file_prefix = str(args.tensor_dir / f"{name}_profile")
        if args.cpu_threads:
            options.intra_op_num_threads = args.cpu_threads
            options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            entry["model_path"],
            sess_options=options,
            providers=[
                (
                    "CUDAExecutionProvider",
                    {"cudnn_conv_algo_search": args.conv_algo_search},
                ),
                "CPUExecutionProvider",
            ],
        )
        if session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError(f"{name}: CUDA provider unavailable")
        input_name = session.get_inputs()[0].name
        outputs = [output.name for output in session.get_outputs()]
        feeds = [
            {input_name: np.load(args.tensor_dir / filename)}
            for filename in entry["tensors"]
        ]
        if args.text_batch_size and name in ("cls", "rec"):
            tensors = [feed[input_name] for feed in feeds]
            if any(t.shape[1:] != tensors[0].shape[1:] for t in tensors):
                raise ValueError(f"{name}: tensors must have matching spatial shapes")
            samples = np.concatenate(tensors, axis=0)
            indices = np.arange(args.text_batch_size) % len(samples)
            feeds = [{input_name: np.ascontiguousarray(samples[indices])}]
        valid_samples = sum(feed[input_name].shape[0] for feed in feeds)
        if args.pad_text_batch_size and name in ("cls", "rec"):
            padded_feeds = []
            for feed in feeds:
                tensor = feed[input_name]
                if len(tensor) > args.pad_text_batch_size:
                    raise ValueError("Padding size is smaller than an existing batch")
                padded = np.zeros(
                    (args.pad_text_batch_size, *tensor.shape[1:]), tensor.dtype
                )
                padded[: len(tensor)] = tensor
                padded_feeds.append({input_name: padded})
            feeds = padded_feeds
        shape_sessions: dict[tuple[int, ...], ort.InferenceSession] = {}
        calls = []
        for feed in feeds:
            shape = tuple(feed[input_name].shape)
            if args.session_per_shape:
                if shape not in shape_sessions:
                    shape_sessions[shape] = (
                        session
                        if not shape_sessions
                        else ort.InferenceSession(
                            entry["model_path"],
                            sess_options=options,
                            providers=[
                                (
                                    "CUDAExecutionProvider",
                                    {"cudnn_conv_algo_search": args.conv_algo_search},
                                ),
                                "CPUExecutionProvider",
                            ],
                        )
                    )
                current = shape_sessions[shape]
            else:
                current = session
            if current.get_providers()[0] != "CUDAExecutionProvider":
                raise RuntimeError(f"{name}: CUDA provider unavailable")
            current.run(outputs, feed)
            calls.append((current, feed))
        workloads[name] = (session, outputs, feeds, valid_samples, calls)
    report["warmed_gpu_mib"] = gpu_memory()

    def replay(name: str) -> dict[str, Any]:
        _, outputs, feeds, valid_samples, calls = workloads[name]
        started = time.perf_counter()
        if args.group_by_shape:
            for current, feed in calls:
                for _ in range(args.iterations):
                    current.run(outputs, feed)
        else:
            for _ in range(args.iterations):
                for current, feed in calls:
                    current.run(outputs, feed)
        elapsed = time.perf_counter() - started
        computed_samples_s = (
            args.iterations
            * sum(next(iter(feed.values())).shape[0] for feed in feeds)
            / elapsed
        )
        samples_s = args.iterations * valid_samples / elapsed
        return {
            "elapsed_s": elapsed,
            "image_workloads_s": samples_s / 18
            if name in ("cls", "rec")
            else samples_s,
            "samples_s": samples_s,
            "computed_samples_s": computed_samples_s,
            "session_calls_s": args.iterations * len(feeds) / elapsed,
            "shapes": [list(next(iter(feed.values())).shape) for feed in feeds],
        }

    report["isolated"] = {}
    for name in workloads:
        report["isolated"][name] = replay(name)
        print(name, json.dumps(report["isolated"][name]), flush=True)
    report["after_isolated_gpu_mib"] = gpu_memory()
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(workloads)) as executor:
        report["concurrent"] = dict(zip(workloads, executor.map(replay, workloads)))
    report["concurrent_wall_s"] = time.perf_counter() - started
    report["concurrent_image_workloads_s"] = min(
        result["image_workloads_s"] for result in report["concurrent"].values()
    )
    report["after_concurrent_gpu_mib"] = gpu_memory()
    if args.profile:
        report["profiles"] = {}
        for name, (session, _, _, _, _) in workloads.items():
            path = session.end_profiling()
            events = json.loads(Path(path).read_text())
            providers: dict[str, float] = {}
            operators: dict[str, float] = {}
            for event in events:
                provider = event.get("args", {}).get("provider")
                if provider:
                    providers[provider] = providers.get(provider, 0) + event.get(
                        "dur", 0
                    )
                    key = provider + ":" + event["args"].get("op_name", "unknown")
                    operators[key] = operators.get(key, 0) + event.get("dur", 0)
            report["profiles"][name] = {
                "path": path,
                "provider_duration_us": providers,
                "top_operators_us": sorted(
                    operators.items(), key=lambda pair: pair[1], reverse=True
                )[:10],
            }
    filename = (
        f"bare_batch{args.text_batch_size}_report.json"
        if args.text_batch_size
        else "bare_report.json"
    )
    if args.pad_text_batch_size:
        filename = f"bare_padded{args.pad_text_batch_size}_report.json"
    if args.session_per_shape:
        filename = "bare_session_per_shape_report.json"
    (args.tensor_dir / filename).write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
