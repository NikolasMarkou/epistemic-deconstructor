## What's New in v7.3.0

### Documentation Accuracy Fixes
Deep audit of the entire repository revealed and fixed three documentation issues:

- **Version string sync**: SKILL.md and CLAUDE.md were still reporting v6.9 while the actual release was v7.2.0. Now all version references are consistent.
- **Missing forecast_modeler.py in README**: The script existed, worked, and had full test coverage, but was missing from the README project structure tree and had no CLI Tools documentation section. Added both.
- **Missing `falsify` preset in CLAUDE.md**: The `belief_tracker.py` supports a `falsify` preset (LR=0.0) that was undocumented in the CLAUDE.md presets list.

### Audit Results (all passing)
- 385/385 unit tests pass
- `make validate` passes
- All 63 functional CLI commands tested across 8 scripts
- All 44 documented files verified present
- All cross-references between SKILL.md, CLAUDE.md, and 30 reference files resolve

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
