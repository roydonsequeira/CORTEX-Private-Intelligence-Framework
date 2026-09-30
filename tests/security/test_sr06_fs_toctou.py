"""SR-06 — path check and file open are not atomic (acknowledged, low priority).

Only a controlled temp directory is used. The race is made deterministic by
swapping a directory for a symlink right after the path policy has approved
it, which is what a second local process would have to win. The agent itself
cannot create links, so these are hardening tests.
"""

import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from cortex.tools import sandbox as sandbox_module
from cortex.tools.builtin import filesystem as fs_module
from cortex.tools.builtin.filesystem import FileSystemTool

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlink/hardlink semantics")

SECRET = "OUTSIDE-THE-WORKSPACE"


@pytest.fixture
def layout(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "ws"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "notes.txt").write_text("inside", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "notes.txt").write_text(SECRET, encoding="utf-8")
    return root, outside


def _swap_after(
    module: ModuleType, name: str, monkeypatch: pytest.MonkeyPatch, root: Path, outside: Path
) -> None:
    """Replace workspace/sub with a symlink to outside/ right after ``name`` approved it."""
    original = getattr(module, name)

    def racing(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        (root / "sub").rename(root / "sub-old")
        os.symlink(outside, root / "sub")
        return result

    monkeypatch.setattr(module, name, racing)


@pytest.mark.asyncio
@pytest.mark.xfail(strict=True, reason="#57 SR-06: open follows a symlink swapped in after the check")
async def test_read_after_symlink_swap_stays_in_workspace(
    layout: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, outside = layout
    _swap_after(fs_module, "ensure_not_hidden", monkeypatch, root, outside)
    result = await FileSystemTool(root).execute(action="read_file", path="sub/notes.txt")
    assert SECRET not in result.output


@pytest.mark.asyncio
@pytest.mark.xfail(strict=True, reason="#57 SR-06: write follows a symlink swapped in after the check")
async def test_write_after_symlink_swap_stays_in_workspace(
    layout: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, outside = layout
    _swap_after(fs_module, "ensure_not_hidden", monkeypatch, root, outside)
    await FileSystemTool(root).execute(action="write_file", path="sub/new.txt", content="x")
    assert not (outside / "new.txt").exists()


@pytest.mark.xfail(strict=True, reason="#57 SR-06: sandbox open() has the same check/use window")
def test_sandbox_open_after_symlink_swap_stays_in_workspace(
    layout: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, outside = layout
    _swap_after(sandbox_module, "check_readable_file", monkeypatch, root, outside)
    opener = sandbox_module._workspace_open(root.resolve())
    assert SECRET not in opener("sub/notes.txt").read()


@pytest.mark.asyncio
async def test_static_symlink_out_of_workspace_is_refused(layout: tuple[Path, Path]) -> None:
    """Existing control preserved: a symlink that already points outside is refused."""
    root, outside = layout
    os.symlink(outside / "notes.txt", root / "link.txt")
    result = await FileSystemTool(root).execute(action="read_file", path="link.txt")
    assert result.success is False and SECRET not in result.output


@pytest.mark.asyncio
@pytest.mark.xfail(strict=True, reason="#57 SR-06: a hardlink made by another process is read through")
async def test_hardlink_to_outside_file_is_not_read(layout: tuple[Path, Path]) -> None:
    root, outside = layout
    try:
        os.link(outside / "notes.txt", root / "hard.txt")
    except OSError:
        pytest.skip("hardlinks unsupported here")
    result = await FileSystemTool(root).execute(action="read_file", path="hard.txt")
    assert SECRET not in result.output


@pytest.mark.asyncio
@pytest.mark.xfail(strict=True, reason="#57 SR-06: exists() then write is not O_EXCL")
async def test_no_overwrite_holds_if_file_appears_after_the_check(
    layout: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = layout
    target = root / "late.txt"
    real_exists = Path.exists

    def exists_then_create(self: Path, *a: Any, **k: Any) -> bool:
        seen = real_exists(self, *a, **k)
        if self == target and not seen:
            target.write_text("user data", encoding="utf-8")
        return seen

    monkeypatch.setattr(Path, "exists", exists_then_create)
    await FileSystemTool(root).execute(action="write_file", path="late.txt", content="agent")
    monkeypatch.undo()
    assert target.read_text(encoding="utf-8") == "user data"
