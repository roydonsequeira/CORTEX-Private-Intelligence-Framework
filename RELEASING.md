# Releasing Quietpost

Releases are published to PyPI by `.github/workflows/release.yml` when a version
tag such as `v2.0.0a1` is pushed. The workflow builds the UI into the package,
builds the sdist and then the wheel from it, and runs `scripts/check_dist.py` on
both. It publishes only after you approve the run. Nothing on PyPI needs a token:
the workflow signs in with
[trusted publishing](https://docs.pypi.org/trusted-publishers/), and every file
gets a signed attestation that names the repository and the workflow that
published it.

## Once: set up PyPI and GitHub

1. **On PyPI**, add a pending publisher: your account, *Publishing*,
   *Add a new pending publisher*, *GitHub*. Fill in:

   | Field | Value |
   |---|---|
   | PyPI Project Name | `quietpost` |
   | Owner | `roydonsequeira` |
   | Repository name | `CORTEX-Private-Intelligence-Framework` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

   A pending publisher doesn't reserve the name. Until the first upload, anyone can
   register `quietpost`, and the pending publisher then stops working.

2. **On GitHub**, create the environment before the first tag: *Settings*,
   *Environments*, *New environment*, `pypi`. Under *Deployment protection rules*,
   add yourself as a required reviewer. Under *Deployment branches and tags*,
   allow only tags matching `v*`. A workflow that names a missing environment
   creates it with no rules at all, so do this first.

3. If the repository is renamed, change the repository name in the publisher's
   settings on PyPI too. It is part of what PyPI checks.

## Each release

1. Set `version` in `pyproject.toml` in its normal form, such as `2.0.0a1` (not
   `2.0.0-alpha.1`), and move the CHANGELOG's *Unreleased* section under a heading
   for that version and date. Merge it to `main` with CI green.

2. Dry run on GitHub: *Actions*, *Release*, *Run workflow*, on `main`. It builds and
   checks the files, attaches them to the run as the `dist` artifact, and publishes
   nothing.

   Or on your own computer, in a virtual environment:

   ```bash
   cd ui && npm ci && npm run build:bundle && cd ..
   python -m pip install --require-hashes -r .github/release/requirements.txt
   python -m build --no-isolation
   python scripts/check_dist.py dist --tag v2.0.0a1
   ```

   Empty `dist/` first: the check fails when it holds more than one wheel or sdist.

3. Tag the commit and push the tag:

   ```bash
   git tag -a v2.0.0a1 -m "Quietpost 2.0.0a1"
   git push origin v2.0.0a1
   ```

   The build fails if the tag isn't `v` followed by the version in `pyproject.toml`.

4. The run stops at *Publish to PyPI* until you approve it: *Review deployments*,
   `pypi`, *Approve and deploy*.

5. Check the release:

   ```bash
   pipx install quietpost==2.0.0a1
   pypi-attestations verify pypi --repository https://github.com/roydonsequeira/CORTEX-Private-Intelligence-Framework pypi:quietpost-2.0.0a1-py3-none-any.whl
   ```

   `pypi-attestations` is a separate tool (`pipx install pypi-attestations`).

PyPI never accepts a file name twice, even after the file is deleted. If a release
turns out broken, yank it on PyPI, fix the problem and release the next version,
such as `2.0.0a2`.

## Why the workflow is built this way

- **The job that builds has no publishing rights.** It runs `npm ci` (hundreds of
  packages) and the project's own build, so it may only read the repository. The
  publish job gets the one permission trusted publishing needs, `id-token: write`,
  and runs none of the project's code: it downloads the checked files and runs
  PyPA's publish action.
- **Every action is pinned to a commit**, not to a tag that could be moved to other
  code. Dependabot updates the pins.
- **The build tools are pinned with hashes** in `.github/release/requirements.txt`,
  so a changed file on PyPI fails the build. `python -m build --no-isolation` uses
  exactly those versions. The pinned setuptools must satisfy `build-system.requires`
  in `pyproject.toml`; a unit test checks that.
- **No dependency caches.** A cache saved by another run could otherwise change what
  gets published.
- **Workflow values reach scripts only through environment variables**, so a tag
  name can't become part of a shell command.

`tests/unit/test_release_workflow.py` checks each of these rules, so an edit that
loosens one fails CI.
