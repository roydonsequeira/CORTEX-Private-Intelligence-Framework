"""Workspace path policy shared by the filesystem tool and the Python sandbox.

Both read files for the model, so both must agree on what "inside the
workspace" means; keeping the rules here stops the two from drifting apart.
"""

import contextlib
import os
import re
import stat
import sys
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


# Opening a file the policy above approved. The check is by name, so another
# process could swap a folder on the path for a link to somewhere else (or put a
# file where none was) between the check and the open. These helpers open the
# file itself in a way that cannot be redirected, then check what was opened
# (#57). The agent cannot create links; this is hardening against other local
# programs.

_LINK_SWAPPED = (
    "Access denied: {display} changed after it was checked (a link now leads "
    "outside the workspace). It was not opened."
)
_HARD_LINKED = (
    "Access denied: {display} has another name outside the workspace (a hard "
    "link). It was not opened."
)
_O_NONBLOCK = getattr(os, "O_NONBLOCK", 0)  # a FIFO swapped in must not block the read


def read_workspace_file(root: Path, path: Path, display: str) -> bytes:
    """Read ``path``, which resolve_workspace_path placed inside ``root``.

    Raises OSError if the file is no longer a plain file inside the workspace,
    or is larger than the read limit.
    """
    fd = _open_confined(root, path, os.O_RDONLY | _O_NONBLOCK, display)
    with os.fdopen(fd, "rb") as handle:
        info = os.fstat(handle.fileno())
        _ensure_plain_file(info, display)
        if info.st_size > MAX_READ_BYTES:
            raise OSError("File exceeds 1MB read limit.")
        return handle.read(MAX_READ_BYTES)


def write_workspace_file(
    root: Path, path: Path, data: bytes, *, overwrite: bool, display: str
) -> None:
    """Write ``data`` to ``path`` inside ``root``, creating missing folders.

    A new file is created exclusively (O_EXCL), so without ``overwrite`` a file
    that appeared after the caller's check raises FileExistsError instead of
    being replaced. An existing file is replaced only if it is a plain file with
    no other hard link, and it is emptied only after that check.
    """
    _make_folders(root, path.parent, display)
    try:
        fd = _open_confined(root, path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, display)
    except FileExistsError:
        if not overwrite:
            raise
        fd = _open_confined(root, path, os.O_WRONLY, display)
    with os.fdopen(fd, "wb") as handle:
        _ensure_plain_file(os.fstat(handle.fileno()), display)
        handle.truncate(0)
        handle.write(data)


def _ensure_plain_file(info: os.stat_result, display: str) -> None:
    """Raise OSError unless the opened file is a regular file with one name."""
    if not stat.S_ISREG(info.st_mode):
        raise OSError(f"Not a file: {display}")
    if info.st_nlink > 1:
        raise OSError(_HARD_LINKED.format(display=display))


if sys.platform == "win32":
    import ctypes
    import msvcrt
    from ctypes import wintypes

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _get_final_path = _kernel32.GetFinalPathNameByHandleW
    _get_final_path.argtypes = [wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD]
    _get_final_path.restype = wintypes.DWORD

    def _final_path(fd: int) -> Path:
        """The path the open file really has, with every link and junction resolved."""
        handle = msvcrt.get_osfhandle(fd)
        size = 512
        while True:
            buffer = ctypes.create_unicode_buffer(size)
            length = _get_final_path(handle, buffer, size, 0)
            if length == 0:
                raise ctypes.WinError(ctypes.get_last_error())
            if length < size:
                break
            size = length  # too small: length is the size needed
        name = buffer.value
        if name.startswith("\\\\?\\UNC\\"):
            name = "\\\\" + name[len("\\\\?\\UNC\\") :]
        elif name.startswith("\\\\?\\"):
            name = name[len("\\\\?\\") :]
        return Path(name)

    def _make_folders(root: Path, folder: Path, display: str) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        if not folder.resolve().is_relative_to(root):
            raise OSError(_LINK_SWAPPED.format(display=display))

    def _open_confined(root: Path, path: Path, flags: int, display: str) -> int:
        """Open ``path``, then check the open handle's real path is inside ``root``.

        Windows cannot open relative to a folder handle from Python, so the
        check comes right after the open: a file reached through a junction
        swapped in on the way is closed before anything is read or written, and
        removed if this open just created it (it is still empty).
        """
        fd = os.open(path, flags | os.O_BINARY | os.O_NOINHERIT, 0o666)
        final = None
        try:
            final = _final_path(fd)
            if not final.is_relative_to(root):
                raise OSError(_LINK_SWAPPED.format(display=display))
        except BaseException:
            os.close(fd)
            if final is not None and flags & os.O_EXCL:
                with contextlib.suppress(OSError):
                    os.remove(final)
            raise
        return fd

else:
    _DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC

    def _is_link(name: str, dir_fd: int) -> bool:
        try:
            return stat.S_ISLNK(os.stat(name, dir_fd=dir_fd, follow_symlinks=False).st_mode)
        except OSError:
            return False

    def _open_at(name: str, flags: int, dir_fd: int, display: str) -> int:
        """Open ``name`` in the folder ``dir_fd`` without following a symlink.

        The error for a symlink differs by system and by flags (ELOOP, EMLINK,
        ENOTDIR), so the entry itself is checked to explain the refusal.
        """
        try:
            return os.open(name, flags | os.O_NOFOLLOW | os.O_CLOEXEC, 0o666, dir_fd=dir_fd)
        except OSError:
            if _is_link(name, dir_fd):
                raise OSError(_LINK_SWAPPED.format(display=display)) from None
            raise

    def _walk(root: Path, folders: tuple[str, ...], display: str, create: bool) -> int:
        """Open each folder below ``root`` in turn without following symlinks."""
        fd = os.open(root, _DIR_FLAGS)
        try:
            for folder in folders:
                if create:
                    with contextlib.suppress(FileExistsError):
                        os.mkdir(folder, dir_fd=fd)
                child = _open_at(folder, _DIR_FLAGS, fd, display)
                os.close(fd)
                fd = child
        except BaseException:
            os.close(fd)
            raise
        return fd

    def _make_folders(root: Path, folder: Path, display: str) -> None:
        os.close(_walk(root, folder.relative_to(root).parts, display, create=True))

    def _open_confined(root: Path, path: Path, flags: int, display: str) -> int:
        """Open ``path`` folder by folder from ``root``, following no symlink.

        A folder or file swapped for a symlink after the check fails to open,
        wherever the link points.
        """
        parts = path.relative_to(root).parts
        if not parts:
            raise OSError(f"Not a file: {display}")
        *folders, name = parts
        parent = _walk(root, tuple(folders), display, create=False)
        try:
            return _open_at(name, flags, parent, display)
        finally:
            os.close(parent)
