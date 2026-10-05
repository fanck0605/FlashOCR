from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ...inference_engine.base import InferSession, get_engine
from ...utils.typings import EngineType

if TYPE_CHECKING:
    from collections.abc import Callable

    from omegaconf import DictConfig

    from .onnx import OnnxSession


def create_session(cfg: DictConfig) -> InferSession | OnnxSession:
    if cfg.engine_type == EngineType.ONNXRUNTIME:
        from .onnx import OnnxSession

        return OnnxSession(cfg)
    factory = cast("Callable[[DictConfig], InferSession]", get_engine(cfg.engine_type))
    return factory(cfg)
