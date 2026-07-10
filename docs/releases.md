# Versioning and releases

## Version number

- Current line is **0.2.x** (0.1.x — initial releases, 0.2.0 — the
  queue-based rewrite).
- The version lives in **two places and must stay in sync**:
  - `pyproject.toml` → `version = "X.Y.Z"`
  - `aiologging/__init__.py` → `__version__ = "X.Y.Z"`
- Do **not** bump versions, create tags or releases unprompted — the
  maintainer decides the number and the moment.

## Commit messages

Conventional-commits style, with the version in parentheses when the
commit is the version bump:

```
feat: Telegram handler with 4096-char chunking and 429 retry_after (0.2.2)
feat!: queue-based async logging with strict logging-compatible API (0.2.0)
ci: update actions, run examples; fix Makefile and .gitignore
```

- `type: summary (X.Y.Z)` — feature/fix commits that ship a version;
- `type: summary` — everything else;
- types seen in history: `feat`, `feat!` (breaking), `ci`, `release`,
  `docs`.

A version-bump commit contains the code, tests, example, README
updates **and** both version strings — one commit per release.

## Release format

Uniform for every release (enforced by `make release`):

- **tag**: `vX.Y.Z`
- **GitHub Release title**: `vX.Y.Z` (same as the tag)
- **release notes**: the `### X.Y.Z` section of the README changelog,
  verbatim — the changelog is the single source of truth, write it
  once per release and nowhere else.

## Pipeline

- **CI** (`.github/workflows/ci.yml`) runs on every push/PR to
  `master`/`main`/`develop`: flake8 + mypy + pytest (3.9–3.14),
  all examples, package build with `twine check`.
- **Release** (`.github/workflows/release.yml`) publishes to PyPI
  when a **GitHub Release is published** (or via manual
  `workflow_dispatch`). Pushing a version-bump commit alone does not
  publish anything.

## Release steps (when asked)

1. Bump both version strings, add the `### X.Y.Z` changelog section
   to README; run the full local gate
   (`make quick-test quick-mypy`, flake8, run the touched examples).
2. Commit as `type: summary (X.Y.Z)`.
3. Run `make release` (`scripts/release.sh`). It verifies the
   preconditions (clean tree, `master`, versions in sync, changelog
   section present, tag/release not taken), re-runs the quick gate
   plus `stress-quick`, pushes, waits for CI to go green, then
   creates and publishes the GitHub Release in the uniform format —
   which triggers the PyPI upload.
