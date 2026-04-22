## What's New in v7.15.4

### Audit follow-up release

A deep comprehensive audit (5 parallel analyst agents + targeted spot-checks) confirmed the agent wiring is clean, the 655-test suite still fully covers all Rule 8 hard-caps, and the 37-file reference corpus has no orphans. This release fixes the tier-1 issues that audit did surface.

### Fixed

- **Silent data-loss path removed** — `common.load_json` now raises `JSONCorruptError` on malformed JSON instead of printing a warning and returning `None`. Previously a corrupt session file was indistinguishable from a missing one, and the next save would silently overwrite it. Missing and empty files still return `None` as before.
- **Lossy model conversion now warns** — `parametric_identifier.to_simulator_format()` emits a `UserWarning` when converting ARMAX (MA dropped) or NARMAX (basis collapsed) fits to the simulator's ARX schema. Pure ARX conversion remains silent.
- **Phase 0.7 `--glossary` flag documented** — the `scope_auditor.py enumerate --glossary` flag (shipped in v7.15.0, documented only in CLAUDE.md) is now visible in the `SKILL.md` Phase 0.7 protocol example.
- **PSYCH tier reference date-stamped** — `references/psych-tier-protocol.md` preamble refreshed; OCEAN/Dark Triad/MICE mechanics unchanged.

### Version consistency

Bumped all in-repo version strings from v7.15.2 to **v7.15.4** (Makefile, build.ps1, README badge, SKILL.md header, CLAUDE.md, and three script docstrings). v7.15.3 was a CHANGELOG-only doc release whose version strings were overlooked at the time; consolidated here.

### Deferred

Non-trivial improvements identified by the audit and scheduled for follow-up plans: structured JSON output for trackers, CRLF normalization on Windows, Evidence Rule 5 in-tracker enforcement, Phase 0.3/0.7 "None found" code-level gate enforcement, GitHub Actions CI workflow, cross-phase `$SM gate <phase>` harness. See `plans/plan_2026-04-22_af06c208/findings/deferred-opportunities.md`.

### Full Changelog

See [CHANGELOG.md](https://github.com/NikolasMarkou/epistemic-deconstructor/blob/main/CHANGELOG.md) for complete version history.

## Install

**Claude Code:**
```
git clone https://github.com/NikolasMarkou/epistemic-deconstructor.git ~/.claude/skills/epistemic-deconstructor
```

**Or update existing:**
```
cd ~/.claude/skills/epistemic-deconstructor && git pull
```
