"""The README: links that work on PyPI, and a settings table that matches the code.

PyPI shows README.md as the project page and resolves no relative link, so every
link is absolute, and a link into this repository must name a file that exists.
"""

import re
import tomllib
from pathlib import Path

from pydantic import BaseModel

from quietpost.config.settings import _RENAMED_MODEL_SETTINGS, Settings

ROOT = Path(__file__).resolve().parents[2]
README = (ROOT / "README.md").read_text(encoding="utf-8")
REPOSITORY: str = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
    "urls"
]["Repository"]
# How the README points into this repository, and whether each kind names a file.
_INTO_REPOSITORY = {
    f"{REPOSITORY}/blob/main/": True,
    f"{REPOSITORY}/tree/main/": False,
    REPOSITORY.replace("https://github.com/", "https://raw.githubusercontent.com/") + "/main/": True,
}


def _prose(markdown: str) -> str:
    """The text outside code blocks and code spans, where a link is a link."""
    without_blocks = re.sub(r"^```.*?^```", "", markdown, flags=re.DOTALL | re.MULTILINE)
    return re.sub(r"`[^`\n]*`", "", without_blocks)


def _link_targets() -> list[str]:
    return re.findall(r"\]\(([^)\s]+)\)", _prose(README))


def _anchor(heading: str) -> str:
    """The anchor GitHub gives a heading."""
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def test_every_link_is_absolute_so_it_works_on_pypi() -> None:
    targets = _link_targets()

    assert len(targets) > 20  # the pattern still finds the links
    assert [target for target in targets if not target.startswith("https://")] == []


def test_links_into_the_repository_name_files_and_headings_that_exist() -> None:
    checked = 0
    for target in _link_targets():
        for prefix, names_a_file in _INTO_REPOSITORY.items():
            if not target.startswith(prefix):
                continue
            path, _, anchor = target.removeprefix(prefix).partition("#")
            local = ROOT / path
            assert local.is_file() if names_a_file else local.is_dir(), target
            if anchor:
                text = local.read_text(encoding="utf-8")
                headings = re.findall(r"^#+ (.+)$", text, flags=re.MULTILINE)
                assert anchor in {_anchor(heading) for heading in headings}, target
            checked += 1

    assert checked > 15


def _documented_settings() -> set[str]:
    table = README.split("<summary>All settings and their defaults</summary>", 1)[1]
    names: set[str] = set()
    for row in table.split("</details>", 1)[0].splitlines():
        if row.startswith("| `"):
            names.update(re.findall(r"`([a-z_.]+)`", row.split("|")[1]))
    return names


def _settings_in_code() -> set[str]:
    names: set[str] = set()
    for name, field in Settings.model_fields.items():
        if name in _RENAMED_MODEL_SETTINGS:
            continue
        annotation = field.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            names.update(f"{name}.{nested}" for nested in annotation.model_fields)
        else:
            names.add(name)
    return names


def test_the_settings_table_lists_every_setting_and_nothing_else() -> None:
    assert _documented_settings() == _settings_in_code()
    # The 1.x names are still read, and a sentence under the table says so.
    for old_name in _RENAMED_MODEL_SETTINGS:
        assert f"`{old_name}`" in README
