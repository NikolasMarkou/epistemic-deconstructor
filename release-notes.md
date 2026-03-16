## What's New in v7.4.3

### Bug Fixes
- **KILLED→REFUTED rename**: `bayesian_tracker.py` renamed `Status.KILLED` to `Status.REFUTED` for consistency with `belief_tracker.py`'s `TraitStatus.REFUTED`. All references updated.
- **LR preset correction**: `bayesian_tracker.py` `strong_confirm` preset reduced from 10.0 to 5.0, matching SKILL.md evidence rules (MAX LR = 5.0 for Phases 0-1).
- **Saturation warning boundaries**: Both trackers now fire warnings *before* thresholds (0.85–0.90 confirmation, 0.05–0.10/0.10–0.15 refutation).
- **PSYCH tier LR cap documented**: `smoking_gun` preset (LR=20.0) now explicitly documented as a PSYCH-tier exception in the LR Cap Rules table.
- **TraitStatus enum**: Added `UNASSESSED` to `TraitStatus` enum — profile methods now use the enum instead of raw strings.
- **Documentation**: Fixed stale comment ("KILLED"→"REFUTED"), updated CLAUDE.md state block template, corrected evidence-calibration.md preset values.

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
