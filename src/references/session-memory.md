# Session Memory Protocol

Persist analysis state to filesystem. Context window = RAM. Files = Disk.
Write to disk immediately. The context window will rot. The files won't.

## Table of Contents
- [Bootstrap](#bootstrap)
- [Directory Structure](#directory-structure)
- [File Purposes](#file-purposes)
- [Persistence Rules](#persistence-rules)
- [Tracker Integration](#tracker-integration)
- [Mandatory Re-reads](#mandatory-re-reads)
- [Recovery from Context Loss](#recovery-from-context-loss)
- [Phase Output Templates](#phase-output-templates)
- [Cross-Analysis Knowledge](#cross-analysis-knowledge)

## Bootstrap

**Script location**: `<skill-dir>/scripts/session_manager.py` — resolve `<skill-dir>` to the absolute path of the skill installation directory.

**Session location**: `--base-dir` MUST point to the **user's project directory**, NOT the skill directory. Sessions created inside the skill directory will fail due to sandbox write restrictions.

```bash
# <skill-dir> = absolute path to skill installation
# <project-dir> = user's working directory
python3 <skill-dir>/scripts/session_manager.py --base-dir <project-dir> new "System description"
python3 <skill-dir>/scripts/session_manager.py --base-dir <project-dir> resume
python3 <skill-dir>/scripts/session_manager.py --base-dir <project-dir> status
python3 <skill-dir>/scripts/session_manager.py --base-dir <project-dir> close
python3 <skill-dir>/scripts/session_manager.py --base-dir <project-dir> list
```

`new` refuses if active session exists — use `resume`, `close`, or `new --force`.

### Absolute Path Rule (CRITICAL)

The Write/Read/Edit tools REQUIRE absolute paths starting with `/`. Relative paths will fail.

Both `new` and `resume` write the absolute session path to `<project-dir>/analyses/.session_dir`. **Before any file operation, read this file:**

```bash
Read("<project-dir>/analyses/.session_dir")
# Returns e.g.: /home/user/project/analyses/analysis_2026-02-20_abc123
```

Use the path from that file as prefix for every session file operation:
```
WRONG: Write("analyses/analysis_2026-02-20_abc123/state.md")
RIGHT: Write("/home/user/project/analyses/analysis_2026-02-20_abc123/state.md")
```

After bootstrap → begin Phase 0. Fill `SESSION_DIR/analysis_plan.md` with setup outputs.

## Directory Structure

```
analyses/
├── .current_analysis              # → active session directory name
├── FINDINGS.md                    # Consolidated observations across sessions
├── DECISIONS.md                   # Consolidated decisions across sessions
└── analysis_YYYY-MM-DD_XXXXXXXX/  # {session-dir}
    ├── state.md                   # Current phase, tier, hypotheses summary
    ├── analysis_plan.md           # Phase 0 output (system, questions, initial H)
    ├── decisions.md               # Append-only: hypothesis pivots, approach changes
    ├── observations.md            # Index of observations + key constraints
    ├── observations/              # Detailed observation files
    ├── progress.md                # Phase completion + stop conditions
    ├── phase_outputs/             # Structured output per completed phase
    │   ├── phase_0.md             # Setup & Frame output
    │   ├── phase_1.md             # Boundary Mapping output
    │   └── ...                    # One file per completed phase
    ├── validation.md              # Phase 5 validation results
    ├── hypotheses.json            # Bayesian tracker state (via --file flag)
    ├── beliefs.json               # PSYCH tier state (via --file flag)
    ├── rapid_assessment.json      # RAPID tier state (via --file flag)
    └── summary.md                 # Written at session close
```

## File Purposes

| File | What It Tracks | When Updated |
|------|---------------|--------------|
| `state.md` | Phase, tier, fidelity, hypothesis count, confidence, transitions | Every phase transition |
| `analysis_plan.md` | System type, access, question pyramid, initial hypotheses | Phase 0 |
| `decisions.md` | Why hypotheses were pursued/dropped, tier changes | On hypothesis pivots |
| `observations.md` | Index of probe results, patterns, data | During Phases 1-4 |
| `observations/` | Detailed observation files (one per topic) | During Phases 1-4 |
| `progress.md` | Phase completion status, stop condition checklists | Every phase transition |
| `phase_outputs/` | Structured deliverable per phase | On phase completion |
| `validation.md` | Validation hierarchy, residual diagnostics, baseline | Phase 5 |
| `summary.md` | Final analysis report | Session close |

## Persistence Rules

All file paths below are relative to `SESSION_DIR` (the absolute path from `session_manager.py`). Always use `SESSION_DIR/filename` in Read/Write/Edit calls.

### What to Write and When

| Event | Write To | Content |
|-------|----------|---------|
| Phase 0 complete | `SESSION_DIR/analysis_plan.md`, `SESSION_DIR/phase_outputs/phase_0.md`, `SESSION_DIR/state.md`, `SESSION_DIR/progress.md` | Setup outputs, question pyramid, initial hypotheses |
| New observation | `SESSION_DIR/observations.md` index + `SESSION_DIR/observations/{topic}.md` | Probe result, pattern, data point |
| Hypothesis update | Run tracker with `--file SESSION_DIR/hypotheses.json` | Bayesian update |
| Hypothesis pivot | `SESSION_DIR/decisions.md` | Why old H dropped, why new H adopted |
| Phase N complete | `SESSION_DIR/phase_outputs/phase_N.md`, `SESSION_DIR/state.md`, `SESSION_DIR/progress.md` | Phase deliverable, updated state |
| Tier change | `SESSION_DIR/decisions.md`, `SESSION_DIR/state.md` | Rationale for escalation/de-escalation |
| Session end | `SESSION_DIR/summary.md`, then `session_manager.py close` | Final report, merge to consolidated |

### Flush Frequency
- After every 2 observations → update `SESSION_DIR/observations.md` index
- After every phase transition → update `SESSION_DIR/state.md` + `SESSION_DIR/progress.md`
- After every hypothesis update → run tracker CLI (auto-persists to JSON)

## Tracker Integration

Redirect existing tracker scripts to the session directory using the **absolute paths** printed by `session_manager.py new`. All scripts are at `<skill-dir>/scripts/`.

```bash
# System analysis hypotheses (use absolute path from session_manager.py output)
python3 <skill-dir>/scripts/bayesian_tracker.py --file /absolute/path/to/analyses/{session-dir}/hypotheses.json add "H1" --prior 0.6
python3 <skill-dir>/scripts/bayesian_tracker.py --file /absolute/path/to/analyses/{session-dir}/hypotheses.json update H1 "Evidence" --preset strong_confirm
python3 <skill-dir>/scripts/bayesian_tracker.py --file /absolute/path/to/analyses/{session-dir}/hypotheses.json report

# PSYCH tier traits
python3 <skill-dir>/scripts/belief_tracker.py --file /absolute/path/to/analyses/{session-dir}/beliefs.json add "High Neuroticism" --prior 0.5
python3 <skill-dir>/scripts/belief_tracker.py --file /absolute/path/to/analyses/{session-dir}/beliefs.json profile

# RAPID tier assessment
python3 <skill-dir>/scripts/rapid_checker.py --file /absolute/path/to/analyses/{session-dir}/rapid_assessment.json start "Claim"
python3 <skill-dir>/scripts/rapid_checker.py --file /absolute/path/to/analyses/{session-dir}/rapid_assessment.json verdict
```

## Mandatory Re-reads

These files are active working memory. Re-read during the conversation, not just at start.

All paths below use `SESSION_DIR` = the absolute path printed by `session_manager.py new` or `resume`.

| When | Read | Why |
|------|------|-----|
| Before starting any phase | `SESSION_DIR/state.md`, `SESSION_DIR/progress.md` | Confirm phase, check stop conditions |
| Before updating hypotheses | `SESSION_DIR/decisions.md` | Don't repeat failed approaches |
| Before Phase 0 or tier change | `<project-dir>/analyses/FINDINGS.md`, `<project-dir>/analyses/DECISIONS.md` | Cross-analysis context |
| After >10 turns in same phase | `SESSION_DIR/state.md`, latest `SESSION_DIR/phase_outputs/` | Reorient, check for drift |
| On context loss / new conversation | All files via `session_manager.py resume` | Full state recovery |

## Recovery from Context Loss

1. Run `python3 <skill-dir>/scripts/session_manager.py --base-dir <project-dir> resume`
2. Read `<project-dir>/analyses/.session_dir` → get absolute session path. ALL reads below use this path as prefix.
3. Read `SESSION_DIR/state.md` → current phase, tier, hypothesis count
4. Read `SESSION_DIR/analysis_plan.md` → system type, questions, fidelity target
5. Read `SESSION_DIR/decisions.md` → what was tried and why
6. Read `SESSION_DIR/progress.md` → which phases complete, which remaining
7. Read latest `SESSION_DIR/phase_outputs/` files → completed phase deliverables
8. Read `SESSION_DIR/observations.md` → accumulated data
9. Resume from current phase. **Never start over.**

## Phase Output Templates

Write one file per completed phase to `SESSION_DIR/phase_outputs/`.

### phase_0.md
```markdown
# Phase 0: Setup & Frame
## System: [description]
## Tier: [selected tier]
## Fidelity: [L1-L5]
## Question Pyramid
- L1 (DO): [question]
- L2 (HOW): [question]
- ...
## Initial Hypotheses
- H1: [statement] (prior: X)
- H2: [statement] (prior: X)
- H3: [adversarial/deceptive] (prior: X)
## Adversarial Pre-check: [result]
## Stop Conditions: [all checked]
```

### phase_1.md
```markdown
# Phase 1: Boundary Mapping
## I/O Channels: [count] characterized
## Probe Database: [count] entries
| Stimulus | Response | Notes |
|----------|----------|-------|
## Data Quality: [coherence score]
## Edge Cases Tested: [list]
## Stop Conditions: [checklist]
```

### phase_2.md through phase_5.md
Follow similar pattern: section per activity, results table, stop condition checklist.

### PSYCH tier: phase_0p.md through phase_5p.md
Same pattern with PSYCH-specific outputs (OCEAN, Dark Triad, MICE/RASP).

## EXIT GATE Checklist per Phase

This section mirrors the SKILL.md File Write Matrix in expanded form. If SKILL.md is the enforcement rule, this is the detailed specification of what each file should contain after each phase.

**All filenames below are relative to `SESSION_DIR`.** When reading or writing, always use `SESSION_DIR/filename` (absolute path).

### After Phase 0 (Setup & Frame)

| File | Expected Content |
|------|-----------------|
| `analysis_plan.md` | ALL fields populated: system description, access level, adversary status, tier, fidelity target, question pyramid (L1 through target), initial hypotheses list, adversarial pre-check result, cognitive vulnerabilities. **No placeholder text.** |
| `hypotheses.json` | ≥3 hypotheses via `bayesian_tracker.py add`, including ≥1 adversarial hypothesis. Priors sum to 1.0 for mutually exclusive sets. |
| `decisions.md` | Tier selection entry: chosen tier + trade-off rationale ("X at the cost of Y") |
| `state.md` | Phase=0 complete, Tier=selected, Fidelity=set, Active Hypotheses=N, Lead=HN, Confidence=Low |
| `progress.md` | Phase 0 in Completed; Phase 1 in In Progress; remaining phases listed |
| `phase_outputs/phase_0.md` | Setup deliverables: system, tier, fidelity, question pyramid, hypotheses, pre-check, cognitive traps |

### After Phase 1 (Boundary Mapping)

| File | Expected Content |
|------|-----------------|
| `observations/` | ≥3 observation files (LITE: ≥1), named `obs_NNN_topic.md` |
| `observations.md` | Index table listing all observation files with ID, title, phase, key finding |
| `hypotheses.json` | ≥1 evidence update per active hypothesis via `bayesian_tracker.py update` |
| `state.md` | Phase=1 complete, updated hypothesis count and lead |
| `progress.md` | Phase 1 in Completed; Phase 2 in In Progress |
| `phase_outputs/phase_1.md` | I/O channels, probe database, data quality, edge cases, stop conditions |

### After Phase 2 (Causal Analysis)

| File | Expected Content |
|------|-----------------|
| `observations/` | New findings added; `observations.md` index updated |
| `hypotheses.json` | Falsification evidence applied; ≥1 hypothesis refuted or significantly weakened |
| `decisions.md` | Causal model choices logged (if any) |
| `state.md` | Phase=2 complete, lead hypothesis updated |
| `progress.md` | Phase 2 in Completed; Phase 3 in In Progress |
| `phase_outputs/phase_2.md` | Causal graph, dependency matrix, falsification results |

### After Phases 3-4 (Parametric ID, Model Synthesis)

Follow same pattern: `state.md` + `progress.md` + `phase_outputs/phase_N.md` + `hypotheses.json` updates. Phase 3 adds model documentation; Phase 4 adds composition and emergence results.

### After Phase 5 (Validation & Report)

| File | Expected Content |
|------|-----------------|
| `validation.md` | Validation hierarchy table (check, method, result, evidence), verdict |
| `summary.md` | Final analysis report referencing session files, including state block |
| `hypotheses.json` | Final posteriors |
| `state.md` | Phase=5 complete, final confidence |
| `progress.md` | All phases marked complete |
| `phase_outputs/phase_5.md` | Validation report |

---

## Cross-Analysis Knowledge

On `session_manager.py close`:
- `observations.md` merges to `analyses/FINDINGS.md` (newest first)
- `decisions.md` merges to `analyses/DECISIONS.md` (newest first)
- Per-session headings demoted (## → ###)
- Observation links rewritten to include session directory

At start of new analysis (use absolute paths — `<project-dir>/analyses/`):
- Read `<project-dir>/analyses/FINDINGS.md` for patterns from prior analyses
- Read `<project-dir>/analyses/DECISIONS.md` for approaches that worked/failed before
- Apply prior knowledge to Phase 0 hypothesis seeding
