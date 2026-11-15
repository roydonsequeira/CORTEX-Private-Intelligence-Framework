"""The release workflow's safety rules, so that an edit can't loosen one unnoticed.

.github/workflows/release.yml publishes to PyPI. Its build job runs hundreds of
npm packages and the project's own build, so it gets no publishing rights; the
publish job gets them, and runs only the two actions it needs.
"""

import re
import tomllib
from pathlib import Path
from typing import Any

import yaml
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"
BUILD_TOOLS = ROOT / ".github" / "release" / "requirements.txt"
# owner/repo@ followed by a full commit hash, which nobody can move to other code.
PINNED = re.compile(r"^[\w.-]+/[\w.-]+@[0-9a-f]{40}$")


def _workflow() -> dict[Any, Any]:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(workflow, dict)
    return workflow


def _steps(job: str) -> list[dict[str, Any]]:
    steps = _workflow()["jobs"][job]["steps"]
    assert isinstance(steps, list)
    return steps


def _pinned_build_tools() -> dict[str, str]:
    """Name and version of each requirement, each line continuation joined."""
    text = BUILD_TOOLS.read_text(encoding="utf-8").replace("\\\n", " ")
    pins: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = re.match(
            r"^([A-Za-z0-9_.-]+)==([^\s;]+)(\s*;[^-]+)?(\s+--hash=sha256:[0-9a-f]{64})+\s*$", line
        )
        assert match, f"not pinned with == and a sha256 hash: {line}"
        pins[canonicalize_name(match[1])] = match[2]
    return pins


def test_it_runs_for_version_tags_and_by_hand_only() -> None:
    # YAML 1.1 reads the key `on` as the boolean true.
    assert _workflow()[True] == {"push": {"tags": ["v*"]}, "workflow_dispatch": None}


def test_each_job_has_only_the_permissions_it_lists() -> None:
    workflow = _workflow()

    assert workflow["permissions"] == {}
    assert set(workflow["jobs"]) == {"build", "publish"}
    assert workflow["jobs"]["build"]["permissions"] == {"contents": "read"}
    assert workflow["jobs"]["publish"]["permissions"] == {"id-token": "write"}


def test_only_a_pushed_tag_publishes_and_only_in_the_pypi_environment() -> None:
    publish = _workflow()["jobs"]["publish"]

    assert publish["needs"] == "build"
    assert publish["if"] == "github.event_name == 'push' && github.ref_type == 'tag'"
    assert publish["environment"]["name"] == "pypi"


def test_the_publish_job_runs_none_of_the_projects_code() -> None:
    steps = _steps("publish")

    assert all("run" not in step for step in steps)
    assert [step["uses"].split("@")[0] for step in steps] == [
        "actions/download-artifact",
        "pypa/gh-action-pypi-publish",
    ]
    # Attestations and the metadata check are on by default; nothing may switch them off.
    assert all("with" not in step for step in steps if "pypi-publish" in step["uses"])


def test_every_action_is_pinned_to_a_commit() -> None:
    for job in _workflow()["jobs"].values():
        for step in job["steps"]:
            if "uses" in step:
                assert PINNED.match(step["uses"]), step["uses"]


def test_no_dependency_cache_can_reach_the_release() -> None:
    for step in _steps("build"):
        uses = step.get("uses", "")
        settings = step.get("with", {})
        assert not uses.startswith("actions/cache")
        assert "cache" not in settings, uses
        if uses.startswith("actions/setup-node@"):
            # setup-node caches on its own when package.json names a package manager.
            assert settings["package-manager-cache"] is False


def test_the_checkout_keeps_no_credentials() -> None:
    checkouts = [
        step for step in _steps("build") if step.get("uses", "").startswith("actions/checkout@")
    ]

    assert len(checkouts) == 1
    assert checkouts[0]["with"]["persist-credentials"] is False


def test_scripts_get_workflow_values_only_through_the_environment() -> None:
    """A tag name such as v1$(curl …) must not become part of a shell script."""
    for job in _workflow()["jobs"].values():
        for step in job["steps"]:
            assert "${{" not in step.get("run", ""), step


def test_the_build_job_builds_the_ui_then_checks_the_files_against_the_tag() -> None:
    scripts = "\n".join(step.get("run", "") for step in _steps("build"))

    assert scripts.index("npm ci") < scripts.index("npm run build:bundle")
    assert scripts.index("npm run build:bundle") < scripts.index("python -m build --no-isolation")
    assert 'python scripts/check_dist.py dist --tag "$TAG"' in scripts
    assert "python -m pip install --require-hashes -r .github/release/requirements.txt" in scripts
    assert _steps("build")[-1]["uses"].startswith("actions/upload-artifact@")


def test_the_build_tools_are_pinned_with_hashes() -> None:
    assert set(_pinned_build_tools()) == {
        "build",
        "colorama",  # Windows only
        "packaging",
        "pyproject-hooks",
        "setuptools",
    }


def test_the_pinned_build_tools_satisfy_pyproject() -> None:
    """Otherwise `python -m build --no-isolation` stops at its dependency check."""
    pins = _pinned_build_tools()
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    for line in pyproject["build-system"]["requires"]:
        requirement = Requirement(line)
        name = canonicalize_name(requirement.name)
        assert name in pins, line
        assert requirement.specifier.contains(pins[name]), (line, pins[name])
