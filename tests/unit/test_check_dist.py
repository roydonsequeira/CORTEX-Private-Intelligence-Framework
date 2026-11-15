"""scripts/check_dist.py, the check that runs before a release is published."""

import io
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest
from scripts import check_dist

VERSION = "2.0.0a1"
COMMAND = "[console_scripts]\nquietpost = quietpost.cli:main\n"
SOURCE = (
    "__init__.py",
    "cli.py",
    "api/__init__.py",
    "api/web.py",
    "tools/__init__.py",
    "tools/plugins/__init__.py",
    "tools/plugins/calculator.py",
    "tools/plugins/README.md",
    "web/index.html",
    "web/_next/static/chunks/app.js",
)
# Everything in SOURCE but the plugins README, which is left out on purpose.
SHIPPED = tuple(name for name in SOURCE if name not in check_dist.LEFT_OUT)


def _metadata(version: str = VERSION, name: str = "quietpost") -> str:
    return f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n\nThe README.\n"


def _make_root(tmp_path: Path, version: str = VERSION, source: tuple[str, ...] = SOURCE) -> Path:
    root = tmp_path / "repo"
    package = root / "src" / "quietpost"
    for name in source:
        (package / name).parent.mkdir(parents=True, exist_ok=True)
        (package / name).write_text("", encoding="utf-8")
    # Left behind by running the code; never part of a release. The second is what
    # an interrupted bytecode write leaves.
    (package / "__pycache__").mkdir(exist_ok=True)
    (package / "__pycache__" / "cli.cpython-313.pyc").write_bytes(b"")
    (package / "__pycache__" / "cli.cpython-313.pyc.1953757952").write_bytes(b"")
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "quietpost"\nversion = "{version}"\n', encoding="utf-8"
    )
    return root


def _make_wheel(
    dist: Path,
    files: tuple[str, ...] = SHIPPED,
    *,
    version: str = VERSION,
    tags: str = "py3-none-any",
    metadata: str | None = None,
    entry_points: str | None = COMMAND,
    extra: tuple[str, ...] = (),
) -> Path:
    dist.mkdir(exist_ok=True)
    path = dist / f"quietpost-{version}-{tags}.whl"
    info = f"quietpost-{version}.dist-info"
    with zipfile.ZipFile(path, "w") as wheel:
        for name in files:
            wheel.writestr(f"quietpost/{name}", "")
        for name in extra:
            wheel.writestr(name, "")
        wheel.writestr(f"{info}/METADATA", _metadata(version) if metadata is None else metadata)
        if entry_points is not None:
            wheel.writestr(f"{info}/entry_points.txt", entry_points)
        wheel.writestr(f"{info}/RECORD", "")
    return path


def _make_sdist(
    dist: Path,
    files: tuple[str, ...] = SOURCE,
    *,
    version: str = VERSION,
    metadata: str | None = None,
) -> Path:
    dist.mkdir(exist_ok=True)
    path = dist / f"quietpost-{version}.tar.gz"
    top = f"quietpost-{version}"
    with tarfile.open(path, "w:gz") as sdist:

        def add(name: str, text: str = "") -> None:
            data = text.encode("utf-8")
            member = tarfile.TarInfo(f"{top}/{name}")
            member.size = len(data)
            sdist.addfile(member, io.BytesIO(data))

        add("PKG-INFO", _metadata(version) if metadata is None else metadata)
        add("pyproject.toml")
        for name in files:
            add(f"src/quietpost/{name}")
    return path


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return _make_root(tmp_path)


@pytest.fixture
def dist(tmp_path: Path) -> Path:
    return tmp_path / "dist"


def _one(problems: list[str], text: str) -> None:
    assert len(problems) == 1, problems
    assert text in problems[0]


def test_a_complete_build_passes_with_or_without_the_tag(root: Path, dist: Path) -> None:
    _make_wheel(dist)
    _make_sdist(dist)

    assert check_dist.check(dist, None, root) == []
    assert check_dist.check(dist, f"v{VERSION}", root) == []


def test_main_says_what_it_checked(
    root: Path, dist: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(check_dist, "ROOT", root)
    _make_wheel(dist)
    _make_sdist(dist)

    assert check_dist.main([str(dist), "--tag", f"v{VERSION}"]) == 0

    out = capsys.readouterr().out
    assert f"quietpost-{VERSION}-py3-none-any.whl: 12 files, 2 of them the UI" in out
    assert "plugins: calculator.py" in out
    assert f"They match the tag v{VERSION}." in out


def test_main_lists_every_problem_and_fails(
    root: Path, dist: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(check_dist, "ROOT", root)
    _make_wheel(dist, entry_points=None)
    _make_sdist(dist, files=())

    assert check_dist.main([str(dist), "--tag", "v9.9.9"]) == 1

    err = capsys.readouterr().err
    assert "doesn't match pyproject.toml's version" in err
    assert "doesn't install the command" in err
    assert "a wheel built from it" in err
    assert "3 problems; don't publish these files." in err


def test_main_needs_an_existing_folder(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert check_dist.main([str(tmp_path / "missing")]) == 1
    assert "is not a folder" in capsys.readouterr().err


@pytest.mark.parametrize("tag", ["2.0.0a1", "v2.0.0a2", "v2.0.0-alpha.1", "release-2.0.0a1"])
def test_the_tag_must_be_v_and_the_exact_version(root: Path, dist: Path, tag: str) -> None:
    _make_wheel(dist)
    _make_sdist(dist)

    _one(check_dist.check(dist, tag, root), f"tag the release 'v{VERSION}'")


@pytest.mark.parametrize(
    ("version", "text"),
    [
        ("2.0.0-alpha.1", "write it as '2.0.0a1', its normal form"),
        ("2.0.0.A1", "write it as '2.0.0a1', its normal form"),
        ("two", "isn't a valid version"),
        ("2.0.0a1+laptop", "has a local part, which PyPI refuses"),
    ],
)
def test_the_version_must_be_publishable_as_written(version: str, text: str) -> None:
    _one(check_dist.check_version(version, None), text)


def test_a_version_that_differs_from_the_files_is_caught(tmp_path: Path, dist: Path) -> None:
    """Files left in dist/ by an earlier build carry the old version."""
    root = _make_root(tmp_path, version="2.0.0a2")
    _make_wheel(dist)
    _make_sdist(dist)

    problems = check_dist.check(dist, None, root)

    assert any("isn't the wheel of quietpost 2.0.0a2" in problem for problem in problems)
    assert any("isn't the sdist of quietpost 2.0.0a2" in problem for problem in problems)


def test_a_build_that_skipped_the_ui_fails_although_the_source_has_none_either(
    tmp_path: Path, dist: Path
) -> None:
    no_ui = tuple(name for name in SOURCE if not name.startswith("web/"))
    root = _make_root(tmp_path, source=no_ui)
    _make_wheel(dist, tuple(name for name in no_ui if name not in check_dist.LEFT_OUT))
    _make_sdist(dist, no_ui)

    problems = check_dist.check(dist, None, root)

    assert len(problems) == 2, problems
    assert "has no UI (quietpost/web/index.html)" in problems[0]
    assert "npm run build:bundle" in problems[0]
    assert "a wheel built from it would have none either" in problems[1]


def test_a_file_missing_from_the_wheel_is_named(root: Path, dist: Path) -> None:
    _make_wheel(dist, tuple(name for name in SHIPPED if name != "api/web.py"))
    _make_sdist(dist)

    _one(check_dist.check(dist, None, root), "lacks these files of src/quietpost: ['api/web.py']")


def test_a_ui_file_missing_from_the_wheel_is_named(root: Path, dist: Path) -> None:
    _make_wheel(dist, tuple(name for name in SHIPPED if not name.endswith("app.js")))
    _make_sdist(dist)

    _one(check_dist.check(dist, None, root), "['web/_next/static/chunks/app.js']")


def test_the_shipped_calculator_plugin_is_required(tmp_path: Path, dist: Path) -> None:
    """Even when the source lost it as well, as when it is renamed."""
    source = tuple(name for name in SOURCE if not name.endswith("calculator.py"))
    root = _make_root(tmp_path, source=source)
    _make_wheel(dist, tuple(name for name in source if name not in check_dist.LEFT_OUT))
    _make_sdist(dist, source)

    _one(check_dist.check(dist, None, root), "lacks quietpost/tools/plugins/calculator.py")


def test_files_the_source_lacks_are_reported(root: Path, dist: Path) -> None:
    """Such as a module deleted from the source but left in an old build folder."""
    _make_wheel(dist, (*SHIPPED, "old_module.py"))
    _make_sdist(dist)

    _one(check_dist.check(dist, None, root), "has files that src/quietpost doesn't: ['old_module.py']")


@pytest.mark.parametrize("stray", ["palindrome.py", "tests/test_cli.py", "quietpost.egg-info/x"])
def test_files_outside_the_package_are_reported(root: Path, dist: Path, stray: str) -> None:
    _make_wheel(dist, extra=(stray,))
    _make_sdist(dist)

    _one(check_dist.check(dist, None, root), f"files outside quietpost/ and quietpost-{VERSION}")


@pytest.mark.parametrize("entry_points", [None, "[console_scripts]\ncortex = quietpost.cli:main\n"])
def test_the_wheel_must_install_the_quietpost_command(
    root: Path, dist: Path, entry_points: str | None
) -> None:
    _make_wheel(dist, entry_points=entry_points)
    _make_sdist(dist)

    _one(check_dist.check(dist, None, root), "doesn't install the command")


def test_the_wheel_must_be_pure_python(root: Path, dist: Path) -> None:
    _make_wheel(dist, tags="cp312-cp312-win_amd64")
    _make_sdist(dist)

    _one(check_dist.check(dist, None, root), "isn't a pure Python wheel")


@pytest.mark.parametrize(
    ("metadata", "text"),
    [
        (_metadata("2.0.0a0"), "METADATA has version '2.0.0a0'"),
        (_metadata(name="cortex"), "METADATA names the project 'cortex'"),
        ("", "METADATA names the project ''"),
    ],
)
def test_the_wheel_metadata_must_match(root: Path, dist: Path, metadata: str, text: str) -> None:
    _make_wheel(dist, metadata=metadata)
    _make_sdist(dist)

    problems = check_dist.check(dist, None, root)

    assert any(text in problem for problem in problems), problems


def test_the_sdist_metadata_must_match(root: Path, dist: Path) -> None:
    _make_wheel(dist)
    _make_sdist(dist, metadata=_metadata("2.0.0a0"))

    _one(check_dist.check(dist, None, root), "PKG-INFO has version '2.0.0a0'")


@pytest.mark.parametrize(
    ("old_files", "counts"),
    [
        (("quietpost-2.0.0a0-py3-none-any.whl",), "2 wheels and 1 sdist."),
        (("quietpost-2.0.0a0.tar.gz",), "1 wheel and 2 sdists."),
    ],
)
def test_the_folder_must_hold_one_wheel_and_one_sdist(
    root: Path, dist: Path, old_files: tuple[str, ...], counts: str
) -> None:
    _make_wheel(dist)
    _make_sdist(dist)
    for name in old_files:
        (dist / name).write_bytes(b"")

    _one(check_dist.check(dist, None, root), counts)


def test_an_empty_folder_fails(root: Path, dist: Path) -> None:
    dist.mkdir()

    _one(check_dist.check(dist, None, root), "0 wheels and 0 sdists")


def test_the_named_files_exist_in_this_repo() -> None:
    """A rename here must update the check, or every release would fail it."""
    package = check_dist.ROOT / "src" / "quietpost"

    for name in [*check_dist.REQUIRED, *check_dist.LEFT_OUT]:
        if name != check_dist.UI_ENTRY:  # built by `npm run build:bundle`, absent in CI
            assert (package / name).is_file(), name
    scripts = tomllib.loads((check_dist.ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["scripts"]
    assert [f"{name} = {target}" for name, target in scripts.items()] == [check_dist.CONSOLE_SCRIPT]
