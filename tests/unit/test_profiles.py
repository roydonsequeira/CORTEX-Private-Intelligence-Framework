"""The chat model `quietpost init` suggests, and how it reads the GPU's memory."""

import shutil
import subprocess

import pytest

from quietpost.models import profiles


@pytest.mark.parametrize("gpu_gb", [24.0, 12.0, 8.0, 6.0, 5.5, 4.0, 0.0, None])
def test_only_the_model_that_passed_every_safety_case_is_suggested(gpu_gb: float | None) -> None:
    """The 2 Nov 2026 comparison: qwen3:8b and qwen3.5:4b each failed a safety case.

    qwen3:8b recited its system prompt (case 36), and qwen3.5:4b said it had written
    a file after both writes failed (case 35). Neither is suggested, even on GPUs
    where it would be faster.
    """
    assert profiles.profile_for(gpu_gb).chat_model == "qwen2.5:7b"


def test_less_gpu_memory_says_answers_are_slower() -> None:
    assert "slower" not in profiles.profile_for(6.0).note
    assert "slower" in profiles.profile_for(4.0).note
    assert "slower" in profiles.profile_for(None).note


def test_profiles_are_largest_first_and_cover_every_gpu() -> None:
    sizes = [profile.min_gpu_gb for profile in profiles.PROFILES]
    assert sizes == sorted(sizes, reverse=True)
    assert sizes[-1] == 0.0


def test_gpu_memory_is_read_from_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: "nvidia-smi")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout="6144\n8192\n"),
    )

    assert profiles.detect_gpu_memory_gb() == 8.0
