
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass

__all__ = ["GPUInfo", "detect_gpus", "configured_gpu_gib"]


@dataclass(frozen=True)
class GPUInfo:
    index: int
    name: str
    total_mib: int
    free_mib: int
    driver: str = ""

    @property
    def total_gib(self) -> float:
        return round(self.total_mib / 1024, 1)


def detect_gpus() -> list[GPUInfo]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    query = "--query-gpu=index,name,memory.total,memory.free,driver_version"
    try:
        proc = subprocess.run(
            [exe, query, "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    gpus: list[GPUInfo] = []
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            gpus.append(GPUInfo(
                index=int(parts[0]), name=parts[1],
                total_mib=int(float(parts[2])), free_mib=int(float(parts[3])),
                driver=parts[4] if len(parts) > 4 else "",
            ))
        except ValueError:
            continue
    return gpus


def configured_gpu_gib() -> float | None:
    raw = os.environ.get("MIZAN_GPU_GIB")
    if not raw:
        return None
    try:
        value = float(raw)
        return value if value > 0 else None
    except ValueError:
        return None
