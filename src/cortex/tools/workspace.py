"""Workspace path policy shared by the filesystem tool and the Python sandbox.

Both read files for the model, so both must agree on what "inside the
workspace" means; keeping the rules here stops the two from drifting apart.
"""

import re
from pathlib import Path

MAX_READ_BYTES = 1_048_576
# "C:/..." or "c:\..." (after backslashes are normalised to "/").
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:(/|$)")


def resolve_workspace_path(root: Path, raw_path: str) -> Path:
    """Resolve and validate a relative path under the workspace ``root``.

    Models often address the workspace root as "/", "./" or "" and prefix
    relative paths with "/"; those are normalised to the root. Real absolute
    paths (drive letters, UNC) and any ".." traversal are rejected, and the
    resolved path (symlinks followed) must still be inside the root.
    """
    cleaned = raw_path.strip().replace("\\", "/")
    denied = OSError(
        f"Access denied: '{raw_path}' is outside the CORTEX workspace. "
        "Only paths relative to the workspace root are allowed."
    )
    if cleaned in ("", ".", "/", "./", "~"):
        cleaned = "."
    elif cleaned.startswith("/") and not cleaned.startswith("//"):
        # "/README.md" usually means the workspace root, but "/etc/passwd"
        # is a real absolute path: only accept it if it exists in the workspace.
        relative = cleaned.lstrip("/")
        if not (root / relative).exists():
            raise denied
        cleaned = relative
    candidate = Path(cleaned)
    if ".." in candidate.parts:
        raise OSError(
            "Access denied: paths must stay inside the CORTEX workspace and must not "
            "contain '..'. Files outside the workspace cannot be read or written."
        )
    # Checked textually as well: on Linux, Path("C:/Windows") has no drive and
    # is not absolute, so a Windows-style path would otherwise be treated as a
    # relative folder named "C:" instead of being denied.
    if (
        candidate.is_absolute()
        or candidate.drive
        or cleaned.startswith("//")
        or _WINDOWS_DRIVE.match(cleaned)
    ):
        raise denied
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise denied
    return resolved


def ensure_not_hidden(root: Path, path: Path) -> None:
    """Raise OSError if ``path`` is, or is inside, a hidden entry under ``root``.

    Hidden files and folders (.env, .git, .venv, .cortex) hold secrets, history
    and CORTEX's own memory; no tool reads, lists or writes them, whatever the
    workspace root is.
    """
    relative = path.relative_to(root)
    if any(part.startswith(".") for part in relative.parts):
        raise OSError(
            f"Access denied: hidden files and folders are off-limits ({relative.as_posix()}); "
            ".env, .git and similar hold secrets and history."
        )


def check_readable_file(path: Path, display: str) -> None:
    """Raise OSError unless ``path`` is an existing regular file within the read limit."""
    if not path.exists():
        raise OSError(f"File not found: {display}")
    if not path.is_file():
        raise OSError(f"Not a file: {display}")
    if path.stat().st_size > MAX_READ_BYTES:
        raise OSError("File exceeds 1MB read limit.")


def ensure_text(raw: bytes, display: str) -> None:
    """Raise OSError if ``raw`` looks like a binary file (a NUL byte near the start)."""
    if b"\x00" in raw[:4096]:
        raise OSError(f"{display} looks like a binary file; only text can be read.")
