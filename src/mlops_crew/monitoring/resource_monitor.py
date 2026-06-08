"""Lightweight process resource sampling for training runs."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import psutil


@dataclass(frozen=True)
class ResourceSample:
    """One CPU and memory observation taken during background sampling."""

    timestamp: float
    process_cpu_percent: float
    system_cpu_percent: float
    rss_mb: float
    available_memory_mb: float


class ResourceMonitor:
    """Sample CPU and memory in the background while a training job runs."""

    def __init__(self, interval_seconds: float = 1.0) -> None:
        self.interval_seconds = interval_seconds
        self._process = psutil.Process(os.getpid())
        self._samples: list[ResourceSample] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def samples(self) -> list[ResourceSample]:
        return self._samples

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("ResourceMonitor is already running")

        self._stop.clear()
        with self._lock:
            self._samples.clear()

        self._process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is None:
            return

        thread.join()
        if thread.is_alive():
            raise RuntimeError("Resource monitor thread did not stop")

        self._thread = None
        self._stop = threading.Event()

    def write_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            snapshot = [asdict(sample) for sample in self._samples]
        pd.DataFrame(snapshot).to_csv(path, index=False)

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            memory = self._process.memory_info()
            virtual_memory = psutil.virtual_memory()
            sample = ResourceSample(
                timestamp=time.time(),
                process_cpu_percent=float(
                    self._process.cpu_percent(interval=None)),
                system_cpu_percent=float(psutil.cpu_percent(interval=None)),
                rss_mb=float(memory.rss / (1024 * 1024)),
                available_memory_mb=float(
                    virtual_memory.available / (1024 * 1024)),
            )
            with self._lock:
                self._samples.append(sample)
            self._stop.wait(self.interval_seconds)
