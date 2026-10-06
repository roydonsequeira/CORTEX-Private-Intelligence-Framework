"""SR-06 — path check and file open are not atomic (#57, closed in 1.2.3).

Only a controlled temp directory is used. The race is made deterministic by
swapping a directory for a symlink right after the path policy has approved
it, which is what a second local process would have to win. The agent itself
cannot create links, so these are hardening tests. Each race test asserts that
the swap really happened, so a refactor that skips the hooked call cannot make
it pass by accident.
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
        if not (root / "sub").is_symlink():
            (root / "sub").rename(root / "sub-old")
            os.symlink(outside, root / "sub")
        return result

    monkeypatch.setattr(module, name, racing)


@pytest.mark.asyncio
async def test_read_after_symlink_swap_stays_in_workspace(
    layout: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, outside = layout
    _swap_after(fs_module, "ensure_not_hidden", monkeypatch, root, outside)
    result = await FileSystemTool(root).execute(action="read_file", path="sub/notes.txt")
    assert (root / "sub").is_symlink(), "the swap did not happen"
    assert result.success is False
    assert SECRET not in result.output


@pytest.mark.asyncio
async def test_write_after_symlink_swap_stays_in_workspace(
    layout: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, outside = layout
    _swap_after(fs_module, "ensure_not_hidden", monkeypatch, root, outside)
    result = await FileSystemTool(root).execute(action="write_file", path="sub/new.txt", content="x")
    assert (root / "sub").is_symlink(), "the swap did not happen"
    assert result.success is False
    assert not (outside / "new.txt").exists()


def test_sandbox_open_after_symlink_swap_stays_in_workspace(
    layout: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, outside = layout
    _swap_after(sandbox_module, "check_readable_file", monkeypatch, root, outside)
    opener = sandbox_module._workspace_open(root.resolve())
    with pytest.raises(OSError, match="changed after it was checked"):
        opener("sub/notes.txt")
    assert (root / "sub").is_symlink(), "the swap did not happen"


@pytest.mark.asyncio
async def test_static_symlink_out_of_workspace_is_refused(layout: tuple[Path, Path]) -> None:
    """Existing control preserved: a symlink that already points outside is refused."""
    root, outside = layout
    os.symlink(outside / "notes.txt", root / "link.txt")
    result = await FileSystemTool(root).execute(action="read_file", path="link.txt")
    assert result.success is False and SECRET not in result.output


@pytest.mark.asyncio
async def test_hardlink_to_outside_file_is_not_read(layout: tuple[Path, Path]) -> None:
    root, outside = layout
    try:
        os.link(outside / "notes.txt", root / "hard.txt")
    except OSError:
        pytest.skip("hardlinks unsupported here")
    result = await FileSystemTool(root).execute(action="read_file", path="hard.txt")
    assert result.success is False
    assert SECRET not in result.output


@pytest.mark.asyncio
async def test_hardlinked_file_is_not_overwritten(layout: tuple[Path, Path]) -> None:
    """Replacing a hard-linked file would change the outside file too."""
    root, outside = layout
    try:
        os.link(outside / "notes.txt", root / "hard.txt")
    except OSError:
        pytest.skip("hardlinks unsupported here")
    tool = FileSystemTool(root)
    result = await tool.execute(action="write_file", path="hard.txt", content="x", overwrite=True)
    assert result.success is False
    assert (outside / "notes.txt").read_text(encoding="utf-8") == SECRET


@pytest.mark.asyncio
async def test_no_overwrite_holds_if_file_appears_after_the_check(
    layout: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = layout
    target = root / "late.txt"
    real_exists = Path.exists
    appeared = []

    def exists_then_create(self: Path, *a: Any, **k: Any) -> bool:
        seen = real_exists(self, *a, **k)
        if self == target and not seen:
            target.write_text("user data", encoding="utf-8")
            appeared.append(self)
        return seen

    monkeypatch.setattr(Path, "exists", exists_then_create)
    result = await FileSystemTool(root).execute(action="write_file", path="late.txt", content="agent")
    monkeypatch.undo()
    assert appeared, "the file did not appear after the check"
    assert result.success is False
    assert target.read_text(encoding="utf-8") == "user data"
