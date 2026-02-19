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

```bash
# Create new analysis session
python scripts/session_manager.py new "System description"

# Resume in new conversation
python scripts/session_manager.py resume

# One-line status
python scripts/session_manager.py status

# Close session (merges findings to consolidated files)
python scripts/session_manager.py close

# List all sessions
python scripts/session_manager.py list
```

`new` refuses if active session exists — use `resume`, `close`, or `new --force`.
After bootstrap → begin Phase 0. Fill `analysis_plan.md` with setup outputs.

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

### What to Write and When

| Event | Write To | Content |
|-------|----------|---------|
| Phase 0 complete | `analysis_plan.md`, `phase_outputs/phase_0.md`, `state.md`, `progress.md` | Setup outputs, question pyramid, initial hypotheses |
| New observation | `observations.md` index + `observations/{topic}.md` | Probe result, pattern, data point |
| Hypothesis update | Run tracker with `--file {session-dir}/hypotheses.json` | Bayesian update |
| Hypothesis pivot | `decisions.md` | Why old H dropped, why new H adopted |
| Phase N complete | `phase_outputs/phase_N.md`, `state.md`, `progress.md` | Phase deliverable, updated state |
| Tier change | `decisions.md`, `state.md` | Rationale for escalation/de-escalation |
| Session end | `summary.md`, then `session_manager.py close` | Final report, merge to consolidated |

### Flush Frequency
- After every 2 observations → update `observations.md` index
- After every phase transition → update `state.md` + `progress.md`
- After every hypothesis update → run tracker CLI (auto-persists to JSON)

## Tracker Integration

Redirect existing tracker scripts to the session directory:

```bash
# System analysis hypotheses
python scripts/bayesian_tracker.py --file analyses/{session-dir}/hypotheses.json add "H1" --prior 0.6
python scripts/bayesian_tracker.py --file analyses/{session-dir}/hypotheses.json update H1 "Evidence" --preset strong_confirm
python scripts/bayesian_tracker.py --file analyses/{session-dir}/hypotheses.json report

# PSYCH tier traits
python scripts/belief_tracker.py --file analyses/{session-dir}/beliefs.json add "High Neuroticism" --prior 0.5
python scripts/belief_tracker.py --file analyses/{session-dir}/beliefs.json profile

# RAPID tier assessment
python scripts/rapid_checker.py --file analyses/{session-dir}/rapid_assessment.json start "Claim"
python scripts/rapid_checker.py --file analyses/{session-dir}/rapid_assessment.json verdict
```

## Mandatory Re-reads

These files are active working memory. Re-read during the conversation, not just at start.

| When | Read | Why |
|------|------|-----|
| Before starting any phase | `state.md`, `progress.md` | Confirm phase, check stop conditions |
| Before updating hypotheses | `decisions.md` | Don't repeat failed approaches |
| Before Phase 0 or tier change | `analyses/FINDINGS.md`, `analyses/DECISIONS.md` | Cross-analysis context |
| After >10 turns in same phase | `state.md`, latest `phase_outputs/` | Reorient, check for drift |
| On context loss / new conversation | All files via `session_manager.py resume` | Full state recovery |

## Recovery from Context Loss

1. Run `python scripts/session_manager.py resume` → get session directory + state summary
2. Read `state.md` → current phase, tier, hypothesis count
3. Read `analysis_plan.md` → system type, questions, fidelity target
4. Read `decisions.md` → what was tried and why
5. Read `progress.md` → which phases complete, which remaining
6. Read latest `phase_outputs/` files → completed phase deliverables
7. Read `observations.md` → accumulated data
8. Resume from current phase. **Never start over.**

## Phase Output Templates

Write one file per completed phase to `phase_outputs/`.

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

## Cross-Analysis Knowledge

On `session_manager.py close`:
- `observations.md` merges to `analyses/FINDINGS.md` (newest first)
- `decisions.md` merges to `analyses/DECISIONS.md` (newest first)
- Per-session headings demoted (## → ###)
- Observation links rewritten to include session directory

At start of new analysis:
- Read `analyses/FINDINGS.md` for patterns from prior analyses
- Read `analyses/DECISIONS.md` for approaches that worked/failed before
- Apply prior knowledge to Phase 0 hypothesis seeding
