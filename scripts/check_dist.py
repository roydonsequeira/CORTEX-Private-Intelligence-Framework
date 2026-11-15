"""Checks the files `python -m build` wrote, before they are published.

    python scripts/check_dist.py dist --tag v2.0.0a1

It lists every problem and exits with 1 unless:

- the folder holds exactly one wheel and one sdist of quietpost;
- both carry the version in pyproject.toml, written in its normal form, and the
  tag, when given, is "v" followed by that version;
- the wheel is pure Python and installs the `quietpost` command;
- the wheel holds every file of src/quietpost, the bundled UI and the shipped
  plugins among them (apart from those in LEFT_OUT), and nothing outside the
  package and its metadata;
- the sdist holds the bundled UI too, so a wheel built from it has the UI.

The release workflow (.github/workflows/release.yml) runs it after the build.
RELEASING.md shows how to run it on your own computer.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import tomllib
import zipfile
from email.parser import HeaderParser
from pathlib import Path

from packaging.utils import (
    InvalidSdistFilename,
    InvalidWheelFilename,
    canonicalize_name,
    parse_sdist_filename,
    parse_wheel_filename,
)
from packaging.version import InvalidVersion, Version

ROOT = Path(__file__).resolve().parent.parent
PROJECT = "quietpost"
# Files in src/quietpost that are deliberately not installed.
LEFT_OUT = frozenset({"tools/plugins/README.md"})
# Files a release can't do without, checked by name so that a build that skipped
# a step fails even though the source tree lacks the files as well.
UI_ENTRY = "web/index.html"
REQUIRED = (UI_ENTRY, "tools/plugins/calculator.py", "cli.py")
CONSOLE_SCRIPT = "quietpost = quietpost.cli:main"


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}{'' if number == 1 else 's'}"


def source_files(package_dir: Path) -> set[str]:
    """Every file of the source package, as a POSIX path relative to it."""
    return {
        path.relative_to(package_dir).as_posix()
        for path in package_dir.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    }


def check_version(version: str, tag: str | None) -> list[str]:
    """The version must be valid, in normal form and publishable; the tag must match it."""
    try:
        parsed = Version(version)
    except InvalidVersion:
        return [f"pyproject.toml has version {version!r}, which isn't a valid version (PEP 440)."]
    problems = []
    if str(parsed) != version:
        problems.append(
            f"pyproject.toml has version {version!r}; write it as {str(parsed)!r}, its normal "
            "form, so the tag, the file names and PyPI all show the same text."
        )
    if parsed.local is not None:
        problems.append(f"Version {version!r} has a local part, which PyPI refuses.")
    if tag is not None and tag != f"v{version}":
        problems.append(
            f"The tag {tag!r} doesn't match pyproject.toml's version {version!r}: "
            f"tag the release 'v{version}', or change the version."
        )
    return problems


def _metadata_problems(where: str, text: str, version: str) -> list[str]:
    fields = HeaderParser().parsestr(text)
    problems = []
    name = fields.get("Name") or ""
    if canonicalize_name(name) != PROJECT:
        problems.append(f"{where} names the project {name!r}, not {PROJECT!r}.")
    if fields.get("Version") != version:
        problems.append(f"{where} has version {fields.get('Version')!r}, not {version!r}.")
    return problems


def check_wheel(path: Path, version: str, expected: set[str]) -> list[str]:
    """The wheel's name, metadata, command and files."""
    try:
        name, wheel_version, _build, tags = parse_wheel_filename(path.name)
    except (InvalidWheelFilename, InvalidVersion) as exc:
        return [f"{path.name} isn't a valid wheel name: {exc}"]
    problems = []
    if name != PROJECT or str(wheel_version) != version:
        problems.append(f"{path.name} isn't the wheel of {PROJECT} {version}.")
    if {str(tag) for tag in tags} != {"py3-none-any"}:
        problems.append(f"{path.name} isn't a pure Python wheel (py3-none-any).")

    dist_info = f"{PROJECT}-{version}.dist-info"
    with zipfile.ZipFile(path) as wheel:
        names = {name for name in wheel.namelist() if not name.endswith("/")}
        metadata = f"{dist_info}/METADATA"
        if metadata in names:
            problems += _metadata_problems(
                f"{path.name}'s METADATA", wheel.read(metadata).decode("utf-8"), version
            )
        else:
            problems.append(f"{path.name} has no {metadata}.")
        entry_points = f"{dist_info}/entry_points.txt"
        commands = (
            wheel.read(entry_points).decode("utf-8").splitlines()
            if entry_points in names
            else []
        )
        if CONSOLE_SCRIPT not in (line.strip() for line in commands):
            problems.append(f"{path.name} doesn't install the command '{CONSOLE_SCRIPT}'.")

    prefix = f"{PROJECT}/"
    packaged = {name.removeprefix(prefix) for name in names if name.startswith(prefix)}
    if UI_ENTRY not in packaged:
        problems.append(
            f"{path.name} has no UI ({prefix}{UI_ENTRY}). Build it into the package "
            "first: `npm ci` and `npm run build:bundle` in ui/."
        )
    problems += [
        f"{path.name} lacks {prefix}{required}."
        for required in REQUIRED
        if required != UI_ENTRY and required not in packaged
    ]
    missing = sorted(expected - LEFT_OUT - packaged - set(REQUIRED))
    if missing:
        problems.append(f"{path.name} lacks these files of src/{PROJECT}: {missing}")
    unexpected = sorted(packaged - expected)
    if unexpected:
        problems.append(f"{path.name} has files that src/{PROJECT} doesn't: {unexpected}")
    outside = sorted(name for name in names if not name.startswith((prefix, f"{dist_info}/")))
    if outside:
        problems.append(f"{path.name} has files outside {prefix} and {dist_info}/: {outside}")
    return problems


def check_sdist(path: Path, version: str) -> list[str]:
    """The sdist's name and metadata, and the UI inside it."""
    try:
        name, sdist_version = parse_sdist_filename(path.name)
    except (InvalidSdistFilename, InvalidVersion) as exc:
        return [f"{path.name} isn't a valid sdist name: {exc}"]
    problems = []
    if name != PROJECT or str(sdist_version) != version:
        problems.append(f"{path.name} isn't the sdist of {PROJECT} {version}.")
    top = f"{PROJECT}-{version}"
    with tarfile.open(path, "r:gz") as sdist:
        names = set(sdist.getnames())
        pkg_info = f"{top}/PKG-INFO"
        member = sdist.extractfile(pkg_info) if pkg_info in names else None
        if member is None:
            problems.append(f"{path.name} has no {pkg_info}.")
        else:
            problems += _metadata_problems(
                f"{path.name}'s PKG-INFO", member.read().decode("utf-8"), version
            )
    if f"{top}/src/{PROJECT}/{UI_ENTRY}" not in names:
        problems.append(
            f"{path.name} has no UI (src/{PROJECT}/{UI_ENTRY}), so a wheel built from it "
            "would have none either."
        )
    return problems


def check(dist_dir: Path, tag: str | None, root: Path) -> list[str]:
    """Every problem with the files in dist_dir; an empty list means they can be published."""
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    version = str(pyproject["project"]["version"])
    problems = check_version(version, tag)
    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        problems.append(
            f"{dist_dir} should hold one wheel and one sdist; it holds "
            f"{_count(len(wheels), 'wheel')} and {_count(len(sdists), 'sdist')}. "
            "Empty it and build again."
        )
        return problems
    problems += check_wheel(wheels[0], version, source_files(root / "src" / PROJECT))
    problems += check_sdist(sdists[0], version)
    return problems


def _summary(dist_dir: Path) -> str:
    wheel = next(dist_dir.glob("*.whl"))
    with zipfile.ZipFile(wheel) as files:
        names = [name for name in files.namelist() if not name.endswith("/")]
    ui = sum(1 for name in names if name.startswith(f"{PROJECT}/web/"))
    plugin_dir = f"{PROJECT}/tools/plugins/"
    plugins = sorted(
        name.removeprefix(plugin_dir)
        for name in names
        if name.startswith(plugin_dir) and name != f"{plugin_dir}__init__.py"
    )
    return f"{wheel.name}: {len(names)} files, {ui} of them the UI; plugins: {', '.join(plugins)}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check the files `python -m build` wrote, before they are published."
    )
    parser.add_argument("dist_dir", type=Path, help="the folder `python -m build` wrote to")
    parser.add_argument("--tag", help="the release tag, such as v2.0.0a1")
    args = parser.parse_args(argv)
    if not args.dist_dir.is_dir():
        print(f"{args.dist_dir} is not a folder.", file=sys.stderr)
        return 1
    problems = check(args.dist_dir, args.tag, ROOT)
    if problems:
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        print(f"{_count(len(problems), 'problem')}; don't publish these files.", file=sys.stderr)
        return 1
    print(_summary(args.dist_dir))
    print("Both files check out." + (f" They match the tag {args.tag}." if args.tag else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
