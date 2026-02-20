## What's New

### Session File I/O Routing (Breaking Change)
All session file operations now route through `session_manager.py write`/`read`/`path` subcommands. Agents pass filenames only — the session manager resolves absolute paths internally. This eliminates the class of Write tool errors caused by relative paths.

SKILL.md has been rewritten to use `$SM write`/`$SM read` for all GATE IN statements, EXIT GATE checklists, and phase activities.

### Evidence Calibration Rules
New enforceable rules in SKILL.md prevent systematic Bayesian tracking errors:
- **LR caps**: Max LR=5.0 in Phases 0-1, LR=10.0 in Phases 2+
- **No batch evidence**: Each data point gets its own update call
- **Adversarial hypothesis**: At least one hypothesis must test data reliability
- **Consensus cap**: Forecaster/institutional consensus capped at LR=2.5
- **Disconfirm before confirm**: Before any hypothesis exceeds 0.80, apply disconfirming evidence
- **Prior discipline**: Mutually exclusive priors must sum to 1.0

New `evidence-calibration.md` reference (220 lines) with full LR scale, cap rules, anti-bundling guidance, and tracker preset reference tables.

### Other Changes
- `--base-dir` flag on session_manager.py decouples skill installation from analysis output
- Path traversal protection on write/read/path commands
- README rewritten with session management section, evidence rules section, updated project stats
- Version bumped across Makefile and build.ps1
- 2 session_manager tests fixed to match new read_pointer() return type
- 191 tests passing

## Install

**Claude Code:**
```
git clone https://github.com/NikolasMarkou/epistemic-deconstructor.git ~/.claude/skills/epistemic-deconstructor
```

**Or update existing:**
```
cd ~/.claude/skills/epistemic-deconstructor && git pull
```

**Full changelog**: https://github.com/NikolasMarkou/epistemic-deconstructor/blob/main/CHANGELOG.md
