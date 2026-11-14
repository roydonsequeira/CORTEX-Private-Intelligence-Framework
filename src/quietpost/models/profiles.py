"""Which models `quietpost init` suggests for the memory a computer's GPU has."""

import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass

EMBED_MODEL = "nomic-embed-text"

# What downloading the suggested models costs, in GB (10^9 bytes), shown before
# `quietpost init` asks. Ollama listed them at 4,683,087,332 and 274,302,450 bytes
# on 12 Nov 2026.
DOWNLOAD_SIZES_GB: dict[str, float] = {"qwen2.5:7b": 4.7, EMBED_MODEL: 0.3}


@dataclass(frozen=True)
class ModelProfile:
    """A recommended chat model for GPUs with at least ``min_gpu_gb`` of memory."""

    min_gpu_gb: float
    chat_model: str
    note: str


# Largest first. Chosen on 2 Nov 2026 with the 39-case live plan (evals/live_battery)
# on a 6 GB laptop GPU, thinking off, by a rule set before the runs: most passes, no
# failure on a safety case (30-36), speed breaks ties. qwen2.5:7b passed 39 of 39.
# qwen3:8b and qwen3.5:4b each failed a safety case, so neither is suggested, even
# where it would be faster. Add a model here only if it passes the same rule.
PROFILES = (
    ModelProfile(5.5, "qwen2.5:7b", "passed all 39 live test cases on a 6 GB GPU"),
    ModelProfile(
        0.0,
        "qwen2.5:7b",
        "the tested model; with less GPU memory, or none, more of it runs on the CPU "
        "and answers are slower",
    ),
)


def profile_for(gpu_gb: float | None) -> ModelProfile:
    """The largest profile that fits ``gpu_gb``; the smallest when it is unknown."""
    if gpu_gb is not None:
        for profile in PROFILES:
            if gpu_gb >= profile.min_gpu_gb:
                return profile
    return PROFILES[-1]


def detect_gpu_memory_gb() -> float | None:
    """GPU memory available for models, in GB, or None when it can't be told.

    NVIDIA: the largest card's total memory. Apple silicon: about two thirds of the
    unified memory, which is what macOS lets the GPU use by default.
    """
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            output = subprocess.run(
                [nvidia_smi, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            ).stdout
            sizes = [float(line) for line in output.split()]
            if sizes:
                return round(max(sizes) / 1024, 1)
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    if sys.platform == "darwin" and platform.machine() == "arm64":
        try:
            output = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            ).stdout
            return round(int(output) / 1024**3 * 2 / 3, 1)
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return None
