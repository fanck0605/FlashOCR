from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tensor_dir", type=Path)
    parser.add_argument("--provider", choices=("cuda", "tensorrt"), default="cuda")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--det-batch-size", type=int, default=0)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("det", "cls", "rec"),
        default=("det", "cls", "rec"),
    )
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    if args.det_batch_size < 0:
        parser.error("det-batch-size must be nonnegative")
    if args.cuda_graph and args.provider != "cuda":
        parser.error("CUDA Graph test requires CUDA EP")
    ort.preload_dlls()
    manifest = json.loads((args.tensor_dir / "manifest.json").read_text())
    report = {}
    for stage, entry in manifest.items():
        if stage not in args.stages:
            continue
        options = ort.SessionOptions()
        options.enable_profiling = args.provider == "tensorrt"
        options.profile_file_prefix = str(args.tensor_dir / f"{stage}_trt_profile")
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        providers = [
            ("CUDAExecutionProvider", {"enable_cuda_graph": str(int(args.cuda_graph))}),
            "CPUExecutionProvider",
        ]
        expected = "CUDAExecutionProvider"
        if args.provider == "tensorrt":
            import onnx

            model = onnx.load(entry["model_path"])
            for info in (
                *model.graph.input,
                *model.graph.output,
                *model.graph.value_info,
            ):
                for axis, dim in enumerate(info.type.tensor_type.shape.dim):
                    if dim.dim_param:
                        dim.dim_param = f"{info.name}_axis_{axis}"
            model_path = args.tensor_dir / f"{stage}_trt.onnx"
            onnx.save(model, model_path)
            tensors = [
                np.load(args.tensor_dir / filename) for filename in entry["tensors"]
            ]
            input_name = model.graph.input[0].name
            shapes = np.asarray([tensor.shape for tensor in tensors])

            profiles = {
                key: input_name + ":" + "x".join(str(int(dim)) for dim in shape)
                for key, shape in (
                    ("min", shapes.min(axis=0)),
                    ("max", shapes.max(axis=0)),
                    ("opt", shapes.max(axis=0)),
                )
            }

            expected = "TensorrtExecutionProvider"
            providers.insert(
                0,
                (
                    expected,
                    {
                        "trt_engine_cache_enable": "True",
                        "trt_engine_cache_path": str(args.tensor_dir / "trt_cache"),
                        "trt_profile_min_shapes": profiles["min"],
                        "trt_profile_max_shapes": profiles["max"],
                        "trt_profile_opt_shapes": profiles["opt"],
                    },
                ),
            )
        session = ort.InferenceSession(
            str(model_path) if args.provider == "tensorrt" else entry["model_path"],
            sess_options=options,
            providers=providers,
        )
        if session.get_providers()[0] != expected:
            raise RuntimeError(f"{stage}: {expected} failed to load")
        name = session.get_inputs()[0].name
        calls = []
        samples = 0
        for graph_id, filename in enumerate(entry["tensors"]):
            tensor = np.load(args.tensor_dir / filename)
            if stage == "det" and args.det_batch_size:
                tensor = np.ascontiguousarray(
                    np.repeat(tensor[:1], args.det_batch_size, axis=0)
                )
            samples += len(tensor)
            value = ort.OrtValue.ortvalue_from_numpy(tensor, "cuda", 0)
            binding = session.io_binding()
            binding.bind_ortvalue_input(name, value)
            # Discover output shapes outside timing, then keep their GPU addresses stable.
            for output in session.get_outputs():
                binding.bind_output(output.name, "cuda")
            no_capture = ort.RunOptions()
            no_capture.add_run_config_entry("gpu_graph_id", "-1")
            session.run_with_iobinding(binding, no_capture)
            output_values = binding.get_outputs()
            reference = [output_value.numpy() for output_value in output_values]
            binding.clear_binding_outputs()
            for output, output_value in zip(session.get_outputs(), output_values):
                binding.bind_ortvalue_output(output.name, output_value)
            run_options = ort.RunOptions()
            run_options.add_run_config_entry(
                "gpu_graph_id", str(graph_id) if args.cuda_graph else "-1"
            )
            for _ in range(3):
                session.run_with_iobinding(binding, run_options)
            calls.append((binding, run_options, value, output_values, reference))
        started = time.perf_counter()
        for _ in range(args.iterations):
            for binding, run_options, _, _, _ in calls:
                session.run_with_iobinding(binding, run_options)
                for output in binding.get_outputs():
                    output.numpy()
        elapsed = time.perf_counter() - started
        for binding, run_options, _, output_values, reference in calls:
            session.run_with_iobinding(binding, run_options)
            for actual, expected_output in zip(output_values, reference):
                np.testing.assert_allclose(
                    actual.numpy(), expected_output, rtol=1e-4, atol=1e-5
                )
        report[stage] = {
            "samples_s": samples * args.iterations / elapsed,
            "elapsed_s": elapsed,
            "calls_s": len(calls) * args.iterations / elapsed,
        }
        if args.provider == "tensorrt":
            events = json.loads(Path(session.end_profiling()).read_text())
            counts = {}
            for event in events:
                provider = event.get("args", {}).get("provider")
                if provider:
                    counts[provider] = counts.get(provider, 0) + 1
            report[stage]["provider_node_calls"] = counts
        print(stage, json.dumps(report[stage]), flush=True)
        del calls, session
    suffix = f"_det_batch{args.det_batch_size}" if args.det_batch_size else ""
    path = (
        args.tensor_dir / f"{args.provider}_graph_{args.cuda_graph}{suffix}_report.json"
    )
    path.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
