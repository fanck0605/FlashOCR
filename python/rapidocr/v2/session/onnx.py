from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import onnxruntime as ort

from ...inference_engine.base import FileInfo, InferSession
from ...utils.download_file import DownloadFile, DownloadFileInput
from ...utils.log import logger

if TYPE_CHECKING:
    from omegaconf import DictConfig


@dataclass
class _Graph:
    input: ort.OrtValue
    outputs: list[ort.OrtValue]
    binding: ort.IOBinding
    options: ort.RunOptions


class OnnxSession:
    """Native ORT inference with one CUDA Graph per complete input shape."""

    def __init__(self, cfg: DictConfig) -> None:
        engine = cfg.engine_cfg
        self._cuda = bool(engine.get("use_cuda", False))
        self._device_id = int(engine.get("cuda_ep_cfg", {}).get("device_id", 0))
        self._lock = Lock()
        self._graphs: dict[tuple[int, ...], _Graph] = {}
        self._next_graph_id = 0
        self._session: ort.InferenceSession | None = None
        if self._cuda:
            preload = getattr(ort, "preload_dlls", None)
            if preload is not None:
                preload()
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("CUDA Graph requires onnxruntime-gpu with CUDA EP")
        supplied = cfg.get("session")
        if supplied is not None:
            if not isinstance(supplied, ort.InferenceSession):
                raise TypeError("session must be an ONNX Runtime InferenceSession")
            self._session = supplied
        else:
            options = ort.SessionOptions()
            options.log_severity_level = 3
            options.enable_cpu_mem_arena = bool(
                engine.get("enable_cpu_mem_arena", False)
            )
            for name in ("intra_op_num_threads", "inter_op_num_threads"):
                value = int(engine.get(name, -1))
                if value > 0:
                    setattr(options, name, value)
            providers: list[Any] = ["CPUExecutionProvider"]
            if self._cuda:
                cuda_options = dict(engine.get("cuda_ep_cfg", {}))
                cuda_options["enable_cuda_graph"] = "1"
                providers.insert(0, ("CUDAExecutionProvider", cuda_options))
            self._session = ort.InferenceSession(
                str(self._resolve_model(cfg)), sess_options=options, providers=providers
            )
        if self._cuda:
            if self._session.get_providers()[0] != "CUDAExecutionProvider":
                raise RuntimeError(
                    "CUDA EP failed to load; CPU fallback is not allowed"
                )
            provider_options = self._session.get_provider_options()[
                "CUDAExecutionProvider"
            ]
            if provider_options.get("enable_cuda_graph") not in ("1", "true", "True"):
                raise ValueError("Provided CUDA session must enable CUDA Graph")
            self._device_id = int(provider_options.get("device_id", self._device_id))
        inputs = self._session.get_inputs()
        if len(inputs) != 1 or inputs[0].type != "tensor(float)":
            raise ValueError("OCR models must have one float32 tensor input")
        self._input_name = inputs[0].name
        self._output_names = [output.name for output in self._session.get_outputs()]
        self._metadata = dict(self._session.get_modelmeta().custom_metadata_map)

    @staticmethod
    def _resolve_model(cfg: DictConfig) -> Path:
        path = cfg.get("model_path")
        if path is None:
            info = InferSession.get_model_url(
                FileInfo(
                    cfg.engine_type,
                    cfg.ocr_version,
                    cfg.task_type,
                    cfg.lang_type,
                    cfg.model_type,
                )
            )
            root = cfg.get("model_root_dir")
            if root is None:
                raise ValueError("model_path or model_root_dir must be provided")
            path = Path(root) / Path(info["model_dir"]).name
            DownloadFile.run(
                DownloadFileInput(
                    file_url=info["model_dir"],
                    sha256=info["SHA256"],
                    save_path=path,
                    logger=logger,
                )
            )
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def _capture(self, tensor: np.ndarray[Any, np.dtype[np.float32]]) -> _Graph:
        assert self._session is not None
        input_value = ort.OrtValue.ortvalue_from_numpy(tensor, "cuda", self._device_id)
        binding = self._session.io_binding()
        binding.bind_ortvalue_input(self._input_name, input_value)
        for name in self._output_names:
            binding.bind_output(name, "cuda", self._device_id)
        ordinary = ort.RunOptions()
        ordinary.add_run_config_entry("gpu_graph_id", "-1")
        self._session.run_with_iobinding(binding, ordinary)
        outputs = binding.get_outputs()
        binding.clear_binding_outputs()
        for name, output in zip(self._output_names, outputs):
            binding.bind_ortvalue_output(name, output)
        options = ort.RunOptions()
        options.add_run_config_entry("gpu_graph_id", str(self._next_graph_id))
        self._next_graph_id += 1
        self._session.run_with_iobinding(binding, options)
        binding.synchronize_outputs()
        return _Graph(input_value, outputs, binding, options)

    def __call__(
        self, tensor: np.ndarray[Any, np.dtype[np.float32]]
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        if tensor.dtype != np.float32 or not tensor.size:
            raise ValueError("Input must be a nonempty float32 tensor")
        tensor = np.ascontiguousarray(tensor)
        # ORT CUDA Graph does not support concurrent Run calls on one session.
        with self._lock:
            if self._session is None:
                raise RuntimeError("ONNX session is closed")
            if not self._cuda:
                return cast(
                    "np.ndarray[Any, np.dtype[np.float32]]",
                    self._session.run(self._output_names, {self._input_name: tensor})[
                        0
                    ],
                )
            shape = tuple(tensor.shape)
            graph = self._graphs.get(shape)
            if graph is None:
                graph = self._capture(tensor)
                self._graphs[shape] = graph
            else:
                graph.input.update_inplace(tensor)
                self._session.run_with_iobinding(graph.binding, graph.options)
            graph.binding.synchronize_outputs()
            return graph.outputs[0].numpy().copy()

    def graph_shapes(self) -> tuple[tuple[int, ...], ...]:
        with self._lock:
            return tuple(self._graphs)

    @property
    def session(self) -> ort.InferenceSession:
        if self._session is None:
            raise RuntimeError("ONNX session is closed")
        return self._session

    def have_key(self, key: str = "character") -> bool:
        return key in self._metadata

    def get_character_list(self, key: str = "character") -> list[str]:
        return self._metadata[key].splitlines()

    @staticmethod
    def get_dict_key_url(file_info: FileInfo) -> str:
        return InferSession.get_dict_key_url(file_info)

    def close(self) -> None:
        with self._lock:
            self._graphs.clear()
            self._session = None
