"""Engine registry. Only one engine is kept in memory at a time."""

from __future__ import annotations

import logging
import threading

from ..config import Settings
from ..core import Progress, UserError
from ..hardware import Hardware, detect
from .base import Engine
from .ltx import LTXEngine
from .mock import MockEngine
from .wan import Wan5BEngine, Wan14BEngine

log = logging.getLogger(__name__)

ENGINE_CLASSES: dict[str, type[Engine]] = {
    LTXEngine.id: LTXEngine,
    Wan5BEngine.id: Wan5BEngine,
    Wan14BEngine.id: Wan14BEngine,
    MockEngine.id: MockEngine,
}


class EngineManager:
    def __init__(self, settings: Settings, hw: Hardware | None = None) -> None:
        self.settings = settings
        self.hw = hw or detect()
        self.engines: dict[str, Engine] = {
            engine_id: cls(settings, self.hw)
            for engine_id, cls in ENGINE_CLASSES.items()
            if engine_id != MockEngine.id or settings.enable_mock
        }
        self._lock = threading.Lock()

    @property
    def default_engine(self) -> str:
        if self.settings.default_engine in self.engines:
            return self.settings.default_engine
        return next(iter(self.engines))

    def get(self, engine_id: str) -> Engine:
        try:
            return self.engines[engine_id]
        except KeyError:
            raise UserError(f"Unknown engine '{engine_id}'. Available: {', '.join(self.engines)}") from None

    def acquire(self, engine_id: str, progress: Progress) -> Engine:
        """Return the engine loaded and ready, unloading any other engine first to free memory."""
        engine = self.get(engine_id)
        with self._lock:
            for other in self.engines.values():
                if other is not engine and other.loaded:
                    log.info("Unloading %s to make room for %s", other.id, engine.id)
                    progress(0.0, f"Unloading {other.label}")
                    other.unload()
            if not engine.loaded:
                progress(0.0, f"Loading {engine.label}")
                engine.load(progress)
        return engine

    def infos(self) -> list[dict]:
        return [engine.info() for engine in self.engines.values()]
