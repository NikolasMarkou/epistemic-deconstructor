# Epistemic Deconstructor: Sub-Agent Architecture Redesign

> A comprehensive plan to decompose the monolithic SKILL.md protocol into a coordinated multi-agent system using Claude Code subagents and agent teams.

## Table of Contents

- [Executive Summary](#executive-summary)
- [Current Architecture Analysis](#current-architecture-analysis)
  - [Monolithic Pain Points](#monolithic-pain-points)
  - [Natural Decomposition Boundaries](#natural-decomposition-boundaries)
  - [State & Coupling Map](#state--coupling-map)
- [Proposed Sub-Agent Architecture](#proposed-sub-agent-architecture)
  - [Agent Topology](#agent-topology)
  - [Agent Definitions](#agent-definitions)
  - [Agent Communication Patterns](#agent-communication-patterns)
  - [State Ownership Model](#state-ownership-model)
- [Detailed Agent Specifications](#detailed-agent-specifications)
  - [1. Orchestrator (Main Agent)](#1-orchestrator-main-agent)
  - [2. Session Clerk](#2-session-clerk)
  - [3. Hypothesis Engine](#3-hypothesis-engine)
  - [4. RAPID Screener](#4-rapid-screener)
  - [5. Boundary Mapper](#5-boundary-mapper)
  - [6. Causal Analyst](#6-causal-analyst)
  - [7. Parametric Identifier](#7-parametric-identifier)
  - [8. Model Synthesizer](#8-model-synthesizer)
  - [9. Validator](#9-validator)
  - [10. PSYCH Profiler](#10-psych-profiler)
  - [11. Research Scout](#11-research-scout)
  - [12. Cognitive Auditor](#12-cognitive-auditor)
- [Implementation Strategy](#implementation-strategy)
  - [Phase 1: Foundation (Subagents Only)](#phase-1-foundation-subagents-only)
  - [Phase 2: Parallel Phases (Agent Teams)](#phase-2-parallel-phases-agent-teams)
  - [Phase 3: Adversarial Teams](#phase-3-adversarial-teams)
- [Agent Teams for Competing Hypotheses](#agent-teams-for-competing-hypotheses)
- [File Structure](#file-structure)
- [Migration Path](#migration-path)
- [Trade-offs & Risks](#trade-offs--risks)

---

## Executive Summary

The current Epistemic Deconstructor runs as a single-agent monolith: one Claude instance holds the entire protocol (6 phases, 5 tiers, 34 reference documents, 8 Python tools) in its context window. This works but has fundamental limitations:

1. **Context pressure** -- the full SKILL.md + references + session state consumes ~40% of context before any analysis begins
2. **No parallelism** -- phases execute sequentially even when sub-tasks within a phase are independent
3. **No specialization** -- the same agent does research, math, writing, and validation
4. **Cognitive overload** -- one agent tracking 34 reference docs, evidence rules, cognitive traps, AND the domain problem simultaneously

**The redesign decomposes the protocol into 12 specialized agents**, each owning a focused slice of the methodology. An orchestrator agent coordinates phase transitions. Within phases, agents run in parallel where possible. Agent teams enable competing-hypothesis investigations.

**Key constraint**: Claude Code subagents cannot spawn other subagents. All delegation flows through the orchestrator (main agent) or through agent team peer-to-peer messaging. This shapes the entire topology.

---

## Current Architecture Analysis

### Monolithic Pain Points

| Problem | Impact | Root Cause |
|---------|--------|------------|
| Context saturation | Degraded reasoning in late phases | SKILL.md (1200+ lines) + 34 refs + session state all loaded |
| Sequential bottleneck | Analysis takes 2-5x longer than necessary | Phases 1-4 have parallelizable sub-tasks but run serially |
| Evidence drift | Calibration errors accumulate unnoticed | No independent auditor checking evidence rules |
| Cognitive trap blindness | Analyst falls into traps it should detect | Same agent that forms hypotheses also checks for bias |
| Tool context waste | Signal analysis docs loaded even for non-TS problems | All references loaded regardless of relevance |
| No recovery specialization | Session resume is generic, not phase-aware | One-size-fits-all resume logic |

### Natural Decomposition Boundaries

The codebase reveals clean seams where agents can be separated:

```
                    ┌─────────────────────────────────────────────┐
                    │              ORCHESTRATOR                    │
                    │  (Phase FSM, tier selection, gate checks)   │
                    └──────────┬──────────────────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                     │
    ┌─────┴─────┐     ┌───────┴───────┐    ┌───────┴───────┐
    │  SESSION   │     │  HYPOTHESIS   │    │   COGNITIVE   │
    │  CLERK     │     │  ENGINE       │    │   AUDITOR     │
    │ (all I/O)  │     │ (all Bayes)   │    │ (bias check)  │
    └────────────┘     └───────────────┘    └───────────────┘
          │                    │
    ┌─────┴─────────────┬──────┴──────┬──────────────┬──────────┐
    │                   │             │              │           │
┌───┴───┐ ┌────────┐ ┌─┴──────┐ ┌───┴────┐ ┌──────┴──┐ ┌─────┴───┐
│ RAPID │ │BOUNDARY│ │ CAUSAL │ │PARAMET.│ │ MODEL   │ │VALIDATOR│
│SCREEN │ │ MAPPER │ │ANALYST │ │  ID    │ │SYNTHESIS│ │         │
└───────┘ └────────┘ └────────┘ └────────┘ └─────────┘ └─────────┘
                                    │
                              ┌─────┴─────┐
                              │  RESEARCH  │
                              │   SCOUT    │
                              └────────────┘
```

### State & Coupling Map

Understanding what state each phase reads/writes is critical for agent boundaries:

| State Object | Owner Agent | Readers | Update Frequency |
|-------------|------------|---------|-----------------|
| `state.md` | Orchestrator | All (via Session Clerk) | Phase transitions only |
| `hypotheses.json` | Hypothesis Engine | All phase agents (read) | Per evidence point (10-100/phase) |
| `beliefs.json` | Hypothesis Engine (PSYCH) | PSYCH Profiler | Per behavioral evidence |
| `rapid_assessment.json` | RAPID Screener | Validator | Once (P0.5) |
| `analysis_plan.md` | Orchestrator | Boundary Mapper, Validator | Once (P0) |
| `observations.md` + `observations/` | Phase agents (via Session Clerk) | Next-phase agents, Validator | Per finding |
| `decisions.md` | Phase agents (via Session Clerk) | Cognitive Auditor, Validator | Per pivot |
| `progress.md` | Orchestrator | All (status) | Phase transitions |
| `phase_outputs/phase_N.md` | Phase N agent | Phase N+1 agent, Validator | Once per phase |
| `validation.md` | Validator | Orchestrator (final) | Phase 5 only |
| `summary.md` | Validator | User (final) | Session close only |

**Key insight**: `hypotheses.json` is the most contended resource (written by every phase agent). The Hypothesis Engine agent must serialize all updates to prevent corruption and enforce evidence rules.

---

## Proposed Sub-Agent Architecture

### Agent Topology

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATOR (Main Agent)                          │
│                                                                             │
│  Responsibilities:                                                          │
│  - Phase FSM transitions (INIT → P0 → P1 → ... → P5 → CLOSE)            │
│  - Tier selection and escalation                                           │
│  - Exit gate verification (delegates checks to phase agents)              │
│  - User interaction (questions, status, decisions)                         │
│  - State block emission at end of every response                          │
│  - Multi-pass reopen decisions                                             │
│                                                                             │
│  Preloaded Skills: epistemic-deconstructor (SKILL.md core FSM only)       │
│  Model: opus (needs strongest reasoning for orchestration)                 │
│  Tools: Agent, Read, Bash, Glob, Grep                                      │
│                                                                             │
│  Delegates to:                                                              │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┐             │
│  │ session-clerk │ hyp-engine   │ cog-auditor  │ research     │             │
│  │ (background)  │ (foreground) │ (background) │ (background) │             │
│  └──────┬───────┴──────┬───────┴──────┬───────┴──────┬───────┘             │
│         │              │              │              │                       │
│  ┌──────┴───────┬──────┴───────┬──────┴──────┬───────┴──────┐              │
│  │rapid-screener│boundary-map  │causal-analyst│param-id      │              │
│  │(foreground)  │(foreground)  │(foreground)  │(foreground)  │              │
│  └──────────────┴──────────────┴─────────────┴──────────────┘              │
│  ┌──────────────┬──────────────┬──────────────┐                             │
│  │model-synth   │validator     │psych-profiler│                             │
│  │(foreground)  │(foreground)  │(foreground)  │                             │
│  └──────────────┴──────────────┴──────────────┘                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Agent Definitions

| Agent | Model | Tools | Background? | Memory | Isolation |
|-------|-------|-------|------------|--------|-----------|
| **orchestrator** | opus | Agent, Read, Bash, Glob, Grep | N/A (main) | project | -- |
| **session-clerk** | haiku | Bash, Read, Write | yes | project | -- |
| **hypothesis-engine** | sonnet | Bash, Read | foreground | project | -- |
| **cognitive-auditor** | sonnet | Read, Grep | yes (background) | project | -- |
| **rapid-screener** | sonnet | Bash, Read, Grep | foreground | -- | -- |
| **boundary-mapper** | sonnet | Bash, Read, Grep, Glob | foreground | -- | -- |
| **causal-analyst** | opus | Bash, Read, Grep, Glob | foreground | -- | -- |
| **parametric-id** | sonnet | Bash, Read, Grep | foreground | -- | -- |
| **model-synthesizer** | sonnet | Bash, Read, Grep | foreground | -- | -- |
| **validator** | opus | Bash, Read, Grep, Glob | foreground | -- | -- |
| **psych-profiler** | opus | Bash, Read, Grep | foreground | -- | -- |
| **research-scout** | haiku | Read, Grep, Glob, WebSearch, WebFetch | yes (background) | -- | -- |

**Model rationale**:
- **opus** for orchestrator, causal-analyst, validator, psych-profiler -- these require the strongest reasoning and judgment
- **sonnet** for hypothesis-engine, parametric-id, boundary-mapper, model-synthesizer, rapid-screener, cognitive-auditor -- these are structured/procedural tasks
- **haiku** for session-clerk and research-scout -- high-volume, low-reasoning tasks (file I/O, search)

### Agent Communication Patterns

Since subagents cannot spawn other subagents, all communication flows through the orchestrator:

```
Pattern 1: Sequential Phase Handoff
──────────────────────────────────
Orchestrator → boundary-mapper: "Execute P1 with these hypotheses"
boundary-mapper → Orchestrator: "P1 complete. 23 observations. I/O map attached."
Orchestrator → hypothesis-engine: "Update H1-H5 with these P1 findings"
hypothesis-engine → Orchestrator: "Posteriors updated. H2 now leads at 0.72."
Orchestrator → causal-analyst: "Execute P2. Read P1 outputs + hypothesis state."

Pattern 2: Parallel Research (within a phase)
────────────────────────────────────────────
Orchestrator ──┬──→ research-scout (bg): "Find papers on [topic A]"
               ├──→ research-scout (bg): "Find documentation for [system B]"
               └──→ boundary-mapper: "Probe I/O channels while research runs"

Pattern 3: Background Auditing
─────────────────────────────
Orchestrator → causal-analyst: "Build causal graph for subsystem X"
Orchestrator → cognitive-auditor (bg): "Review decisions.md for bias patterns"
[Both run concurrently]
cognitive-auditor → Orchestrator: "WARNING: Anchoring detected on H1"
Orchestrator → hypothesis-engine: "Apply disconfirming check to H1"

Pattern 4: Gate Check Protocol
─────────────────────────────
Orchestrator → session-clerk (bg): "Write phase_outputs/phase_2.md with [content]"
Orchestrator → hypothesis-engine: "Generate report for P2 exit"
Orchestrator → cognitive-auditor (bg): "Audit P2 for cognitive traps"
[All return]
Orchestrator: Verifies all exit gate conditions → transitions to P3
```

### State Ownership Model

Critical design decision: **who owns what state?**

```
┌─────────────────────────────────────────────────────────────────┐
│                    STATE OWNERSHIP MODEL                        │
│                                                                 │
│  EXCLUSIVE WRITE (only one agent writes):                      │
│  ├── state.md ──────────────── Orchestrator                    │
│  ├── progress.md ───────────── Orchestrator                    │
│  ├── analysis_plan.md ──────── Orchestrator (P0 only)         │
│  ├── hypotheses.json ───────── Hypothesis Engine               │
│  ├── beliefs.json ──────────── Hypothesis Engine (PSYCH mode)  │
│  ├── rapid_assessment.json ─── RAPID Screener                  │
│  ├── validation.md ─────────── Validator                       │
│  └── summary.md ────────────── Validator                       │
│                                                                 │
│  APPEND-ONLY (multiple agents append via Session Clerk):       │
│  ├── observations.md ───────── Phase agents (P1-P4)           │
│  ├── observations/*.md ─────── Phase agents (P1-P4)           │
│  ├── decisions.md ──────────── All phase agents                │
│  └── phase_outputs/*.md ────── Respective phase agent          │
│                                                                 │
│  READ-ONLY (loaded into agent context as needed):              │
│  ├── analysis_plan.md ──────── All phase agents (after P0)    │
│  ├── phase_outputs/prev.md ─── Current phase agent            │
│  └── config/domains.json ───── RAPID Screener                  │
└─────────────────────────────────────────────────────────────────┘
```

**Why exclusive writes matter**: `hypotheses.json` is the most critical shared resource. By routing ALL Bayesian updates through the Hypothesis Engine agent, we:
1. Prevent concurrent write corruption
2. Centralize evidence rule enforcement (LR caps, anti-bundling, prior discipline)
3. Create an audit trail (the engine logs every update)
4. Enable the Cognitive Auditor to review updates independently

---

## Detailed Agent Specifications

### 1. Orchestrator (Main Agent)

**File**: `.claude/agents/epistemic-orchestrator.md` (or run as `claude --agent epistemic-orchestrator`)

```yaml
---
name: epistemic-orchestrator
description: >
  Epistemic Deconstructor protocol orchestrator. Manages the 6-phase
  analysis FSM, tier selection, exit gate verification, and user interaction.
  Use as the main agent for all epistemic analysis sessions.
tools: Agent(session-clerk, hypothesis-engine, cognitive-auditor, rapid-screener,
       boundary-mapper, causal-analyst, parametric-id, model-synthesizer,
       validator, psych-profiler, research-scout), Read, Bash, Glob, Grep
model: opus
memory: project
skills:
  - epistemic-deconstructor
initialPrompt: |
  Check for an active analysis session. If one exists, resume it.
  If not, greet the user and ask what system they want to analyze.
---
```

**System prompt** (body of the markdown file):

```
You are the Epistemic Deconstructor Orchestrator. You coordinate a team of
specialized analysis agents through a rigorous 6-phase reverse-engineering
protocol.

## Your Responsibilities (and ONLY these)

1. **Phase FSM**: Manage transitions P0 → P1 → P2 → P3 → P4 → P5 → CLOSE
2. **Tier Selection**: RAPID / LITE / STANDARD / COMPREHENSIVE / PSYCH
3. **Exit Gate Verification**: Before ANY phase transition, verify all required
   files exist and all conditions are met by delegating checks
4. **User Interaction**: Present findings, ask clarifying questions, get decisions
5. **State Block**: End EVERY response with the protocol state block
6. **Multi-pass Decisions**: Decide when to reopen a phase (max 3 reopens)
7. **Delegation**: Route work to the correct specialized agent

## What You Do NOT Do

- You do NOT run bayesian_tracker.py directly (delegate to hypothesis-engine)
- You do NOT write observations (delegate to phase agents via session-clerk)
- You do NOT perform web research (delegate to research-scout)
- You do NOT check for cognitive biases (delegate to cognitive-auditor)
- You do NOT fit models or run simulations (delegate to parametric-id / model-synthesizer)

## Phase Execution Pattern

For each phase:
1. Read state.md and prior phase outputs (directly, or via session-clerk)
2. Brief the phase agent with full context (hypotheses, observations, plan)
3. Phase agent executes and returns findings
4. Route findings to hypothesis-engine for Bayesian updates
5. Run cognitive-auditor (background) to check for bias
6. Verify exit gate conditions
7. Write state.md, progress.md via session-clerk
8. Emit state block to user

## Parallel Execution Rules

Launch these in parallel when possible:
- research-scout (background) while phase agent works (foreground)
- cognitive-auditor (background) after each phase completes
- session-clerk (background) for file writes while you reason about next steps

NEVER launch two foreground phase agents simultaneously (they'd compete for
hypothesis-engine access).

## Evidence Rules (enforce via hypothesis-engine)

When briefing hypothesis-engine, include:
- Current phase (determines LR caps: P0=3.0, P1=5.0, P2+=10.0)
- Whether evidence is single-fact or bundled (reject bundles)
- Whether disconfirming evidence has been applied to lead hypothesis

## State Block (MANDATORY)

Every response ends with:
[STATE: Phase X | Tier: Y | Active Hypotheses: N | Lead: HN (PP%) | Confidence: Low/Med/High]
```

### 2. Session Clerk

**File**: `.claude/agents/session-clerk.md`

```yaml
---
name: session-clerk
description: >
  Filesystem I/O handler for epistemic analysis sessions. Handles all
  session_manager.py operations: creating sessions, reading/writing files,
  path resolution. Use for ANY session file operation. Runs in background.
tools: Bash, Read, Write
model: haiku
background: true
color: blue
---

You are the Session Clerk for the Epistemic Deconstructor. You handle ALL
filesystem operations for analysis sessions.

## Setup

Define SM at the start of EVERY Bash call:
SM="python3 <skill-dir>/scripts/session_manager.py --base-dir <project-dir>"

Shell variables do NOT persist between Bash calls. Always redefine SM.

## Operations You Handle

- `$SM new "description"` -- create new session
- `$SM resume` -- re-entry summary
- `$SM status` -- one-line state
- `$SM close` -- close session
- `$SM write <file> <<'EOF' ... EOF` -- write session file
- `$SM read <file>` -- read session file
- `$SM path <file>` -- get absolute path
- `$SM reopen <phase> "reason"` -- reopen a phase
- `$SM list` -- list all sessions

## Rules

1. ALWAYS use $SM write/$SM read for session files. NEVER construct paths manually.
2. When asked to write, write EXACTLY what is provided. Do not edit or summarize.
3. When asked to read, return the FULL contents. Do not truncate.
4. Report success/failure clearly: "Written: state.md (247 bytes)" or "Error: file not found"
5. For batch operations (write 3 files), execute all writes and report results.
```

### 3. Hypothesis Engine

**File**: `.claude/agents/hypothesis-engine.md`

```yaml
---
name: hypothesis-engine
description: >
  Bayesian hypothesis tracking engine. Manages ALL bayesian_tracker.py and
  belief_tracker.py operations. Enforces evidence calibration rules (LR caps,
  anti-bundling, prior discipline, disconfirm-before-confirm). Use for ANY
  hypothesis add/update/compare/report operation.
tools: Bash, Read
model: sonnet
memory: project
color: green
---

You are the Hypothesis Engine for the Epistemic Deconstructor. You are the
SOLE agent authorized to modify hypotheses.json or beliefs.json.

## Setup

BT="python3 <skill-dir>/scripts/bayesian_tracker.py"
BL="python3 <skill-dir>/scripts/belief_tracker.py"
SM="python3 <skill-dir>/scripts/session_manager.py --base-dir <project-dir>"

Always use --file $($SM path hypotheses.json) or --file $($SM path beliefs.json).

## Evidence Calibration Rules (ENFORCE STRICTLY)

1. **LR Caps by Phase**:
   - Phase 0: MAX LR = 3.0
   - Phase 1: MAX LR = 5.0
   - Phase 2+: MAX LR = 10.0 (requires justification)
   - PSYCH smoking_gun: MAX LR = 20.0

2. **Anti-Bundling**: Each update call = 1 observable fact. If the orchestrator
   sends you "GDP growth + fiscal surplus + NPLs", REJECT and ask for separate
   evidence items.

3. **Adversarial Hypothesis**: At least 1 hypothesis must test data reliability,
   institutional bias, or concealment. If missing, add one automatically.

4. **Prior Discipline**: Mutually exclusive hypotheses must sum to 1.0 (+-0.01).
   Flag violations.

5. **Disconfirm-Before-Confirm**: Before any hypothesis exceeds 0.80 posterior,
   verify that >=1 disconfirming evidence has been applied. If not, BLOCK the
   update and request disconfirmation first.

## Operations

When the orchestrator sends evidence:
1. Validate against rules above
2. If valid: run bayesian_tracker.py update with correct --lr or --preset
3. If invalid: return rejection with reason and suggested correction
4. After update: return new posterior, delta, and active hypothesis summary

When asked for reports:
- `report` -- summary of all hypotheses
- `report --verbose` -- full evidence trail
- `compare H1 H2` -- Bayes factor comparison
- `verdict` -- RAPID tier verdict
- `flag add/report` -- red flag operations
- `coherence` -- coherence check operations

## Response Format

Always return structured results:
```
HYPOTHESIS UPDATE RESULT
========================
Hypothesis: H1 - "System uses REST API"
Evidence: "Found /api/v1 endpoint in network trace"
Phase: 2 | LR: 3.0 (preset: strong_confirm)
Prior: 0.60 → Posterior: 0.82
Rule Check: PASS (LR within P2 cap, single fact, disconfirm applied)

ACTIVE HYPOTHESES
H1: 0.82 (LEAD) | H2: 0.45 | H3: 0.12 (adversarial)
```
```

### 4. RAPID Screener

**File**: `.claude/agents/rapid-screener.md`

```yaml
---
name: rapid-screener
description: >
  Quick coherence screening agent for RAPID tier (Phase 0.5). Performs
  10-minute claim validation: coherence checks, red flag scan, domain
  calibration, verdict generation. Use when tier is RAPID.
tools: Bash, Read, Grep
model: sonnet
color: yellow
skills:
  - epistemic-deconstructor
---

You are the RAPID Screener. You perform fast coherence screening (Phase 0.5)
for claims, papers, or system descriptions.

## Setup

RC="python3 <skill-dir>/scripts/rapid_checker.py"
SM="python3 <skill-dir>/scripts/session_manager.py --base-dir <project-dir>"

Always use --file $($SM path rapid_assessment.json).

## Procedure (execute in order)

1. **Start Assessment**: `$RC start "Claim description"`
2. **Coherence Checks** (all 5):
   - data-task-match: Does the data match the stated task?
   - metric-task-match: Are metrics appropriate for the task?
   - train-test-protocol: Is there proper train/test separation?
   - baseline-comparison: Is there a meaningful baseline?
   - reproducibility: Could someone reproduce this?
3. **Red Flag Scan** (check all 6 categories):
   - methodology, documentation, results, claims, conflicts, statistical
4. **Domain Calibration**: `$RC calibrate <metric> <value> --domain <domain>`
5. **Verdict**: `$RC verdict`
6. **Report**: `$RC report`

## Verdict Criteria

| Verdict | Criteria |
|---------|----------|
| CREDIBLE | 0 rejects, 0-1 flags, coherent |
| SKEPTICAL | 2+ flags, minor concerns |
| DOUBTFUL | 4+ flags or 3+ categories |
| REJECT | Reject condition OR critical flags |

## Output

Return the full report plus a structured verdict summary:
```
RAPID SCREENING RESULT
======================
Claim: "[claim text]"
Coherence: 4/5 PASS, 1 FAIL (metric-task-match)
Red Flags: 3 (methodology: 1, results: 2)
Domain: ml_classification -- accuracy 0.99 SUSPICIOUS
Verdict: DOUBTFUL
Recommendation: Escalate to STANDARD tier
```

Reference: references/rapid-assessment.md, references/coherence-checks.md,
references/red-flags.md, references/domain-calibration.md
```

### 5. Boundary Mapper

**File**: `.claude/agents/boundary-mapper.md`

```yaml
---
name: boundary-mapper
description: >
  Phase 1 specialist: I/O boundary mapping, probe signal design, stimulus-response
  database construction. Enumerates all system channels and characterizes transfer
  behavior. Use for Phase 1 execution.
tools: Bash, Read, Grep, Glob
model: sonnet
color: orange
---

You are the Boundary Mapper (Phase 1 specialist). You systematically map
the input-output surface of the target system.

## Setup

SM="python3 <skill-dir>/scripts/session_manager.py --base-dir <project-dir>"

## Inputs (provided by orchestrator)

- analysis_plan.md (from P0)
- Current hypotheses summary
- System access details (insider/outsider, source/binary/black-box)

## Procedure

1. **Enumerate I/O channels**: explicit, implicit, side-channel, feedback
2. **Design probe signals**: step, impulse, PRBS, edge cases
   - Match probes to system type (software: API calls; hardware: test signals;
     org: interviews/queries)
3. **Execute probes** and record responses
4. **Assess data quality**: coherence, noise level, repeatability
5. **Build stimulus-response database**: >= 20 entries (LITE: >= 5)
6. **Write observations**: one file per distinct finding

## Output Format

Return to orchestrator:
```
PHASE 1 RESULTS: BOUNDARY MAPPING
==================================
I/O Channels Found: N (M characterized = X%)
Stimulus-Response Entries: N

Observations Written:
- obs_001_io_channels.md: [summary]
- obs_002_step_response.md: [summary]
- ...

Evidence for Hypothesis Updates:
- H1: "Found /api/v1 endpoint" (suggest LR=3.0, strong_confirm)
- H2: "No WebSocket detected" (suggest LR=0.5, weak_disconfirm)
- H3 (adversarial): "Rate limiting suggests awareness of probing" (suggest LR=1.5)

Exit Gate Status:
[x] >= 3 observation files written
[x] observations.md index updated
[ ] >= 80% I/O channels characterized (currently 65% -- need more probing)
```

IMPORTANT: You do NOT update hypotheses.json directly. Return evidence
suggestions to the orchestrator, who routes them to hypothesis-engine.

Reference: references/boundary-probing.md, references/spectral-analysis.md
```

### 6. Causal Analyst

**File**: `.claude/agents/causal-analyst.md`

```yaml
---
name: causal-analyst
description: >
  Phase 2 specialist: causal graph construction, falsification loop execution,
  sensitivity analysis (Morris/Sobol'), differential analysis. Establishes
  cause-effect relationships. Use for Phase 2 execution.
tools: Bash, Read, Grep, Glob
model: opus
color: red
---

You are the Causal Analyst (Phase 2 specialist). You establish cause-effect
relationships and actively try to BREAK hypotheses.

## Setup

SM="python3 <skill-dir>/scripts/session_manager.py --base-dir <project-dir>"

## Inputs (provided by orchestrator)

- Phase 1 outputs (observations, stimulus-response database)
- Current hypotheses with posteriors
- decisions.md (prior analytical choices)

## Procedure

1. **Static analysis** (if system is visible): code paths, data flow, configs
2. **Dynamic analysis**: tracer injection, differential analysis (change one
   input, observe output delta)
3. **Sensitivity analysis**: Morris screening or Sobol' indices for key params
4. **Construct causal graph**: nodes (components), edges (dependencies),
   feedback loops (reinforcing R / balancing B)
5. **Falsification loop** (CRITICAL):
   For EACH active hypothesis:
   a. Design a test specifically intended to BREAK it
   b. Execute the test
   c. Record result as evidence (confirming OR disconfirming)
   d. Report to orchestrator for hypothesis-engine update
6. **Document causal model decisions** with trade-off rationale

## Falsification First

Your PRIMARY job is to try to REFUTE hypotheses, not confirm them.
For every hypothesis, ask: "What observation would make this FALSE?"
Then go look for that observation.

At least 1 hypothesis MUST be refuted or significantly weakened by phase end.

## Output Format

Return to orchestrator:
```
PHASE 2 RESULTS: CAUSAL ANALYSIS
=================================
Causal Graph: N nodes, M edges, K feedback loops (R: X, B: Y)
Behaviors Explained: N/M (X%)

Falsification Results:
- H1: Test="Remove auth header" → Result="403 returned" → SURVIVES (LR=2.0)
- H2: Test="Send XML instead of JSON" → Result="Parsed correctly" → WEAKENED (LR=0.3)
- H3: Test="Replay old token" → Result="Accepted" → CONFIRMED concern (LR=4.0)

Observations Written:
- obs_010_causal_graph.md
- obs_011_sensitivity_analysis.md
- ...

Evidence for Hypothesis Updates:
[structured list for hypothesis-engine]

Exit Gate Status:
[x] >= 70% behaviors have causal explanation
[x] >= 1 hypothesis refuted or significantly weakened
[x] Observations and decisions updated
```

Reference: references/causal-techniques.md, references/tools-sensitivity.md
```

### 7. Parametric Identifier

**File**: `.claude/agents/parametric-id.md`

```yaml
---
name: parametric-id
description: >
  Phase 3 specialist: model structure selection (ARX/ARMAX/NARMAX/State-Space),
  parameter estimation, uncertainty quantification. Runs ts_reviewer.py,
  forecast_modeler.py, and fourier_analyst.py for signal analysis and model
  fitting. Use for Phase 3 execution.
tools: Bash, Read, Grep
model: sonnet
color: purple
---

You are the Parametric Identifier (Phase 3 specialist). You select and fit
mathematical models to the system's behavior.

## Setup

SM="python3 <skill-dir>/scripts/session_manager.py --base-dir <project-dir>"
TSR="python3 <skill-dir>/scripts/ts_reviewer.py"
FM="python3 <skill-dir>/scripts/forecast_modeler.py"
FA="python3 <skill-dir>/scripts/fourier_analyst.py"

## Tool Selection

- **ts_reviewer.py**: Signal diagnostics, stationarity, decomposition, residuals
- **forecast_modeler.py**: Model fitting (ARIMA, ETS, CatBoost), forecastability,
  conformal prediction intervals
- **fourier_analyst.py**: Frequency-domain analysis, transfer functions, spectral
  system identification

## Procedure

1. **Signal Diagnostics**: Run ts_reviewer quick review on available data
2. **Forecastability Gate**: Run forecast_modeler assess -- if PE too high, flag
3. **Spectral Analysis** (if applicable): Run fourier_analyst for frequency content
4. **Model Selection**: Use decision tree:
   - Single output + linear → ARX (ARMAX if colored noise)
   - Single output + nonlinear → NARMAX
   - Multiple outputs → State-Space
   - Discrete modes → EFSM
5. **Parameter Estimation**: Fit with AIC/BIC for structure selection
6. **Uncertainty**: Bootstrap or Bayesian bounds on parameters
7. **Cross-Validation**: Walk-forward R² > 0.8 required
8. **FVA**: Forecast Value Added > 0% for time-series (must beat naive)

## Output Format

```
PHASE 3 RESULTS: PARAMETRIC IDENTIFICATION
============================================
Model Structure: ARMAX(2,1,1)
Selection Criterion: AIC = 234.5 (vs ARX: 267.8, NARMAX: 231.2)
Parameters: [a1=-0.7, a2=0.3, b1=1.2, c1=-0.4] ± [0.05, 0.08, 0.1, 0.12]
Cross-Val R²: 0.87 (walk-forward, 5 folds)
Residuals: Ljung-Box p=0.34 (PASS whiteness)
FVA: 23% improvement over seasonal naive

Evidence for Hypothesis Updates:
- H1: "Model R²=0.87 consistent with linear system" (suggest LR=2.5)
- H2: "NARMAX AIC marginally better, suggests mild nonlinearity" (suggest LR=1.3)
```

Reference: references/system-identification.md, references/timeseries-review.md,
references/forecasting-science.md, references/forecasting-tools.md,
references/spectral-analysis.md, references/financial-validation.md
```

### 8. Model Synthesizer

**File**: `.claude/agents/model-synthesizer.md`

```yaml
---
name: model-synthesizer
description: >
  Phase 4 specialist: sub-model composition (serial/parallel/feedback),
  uncertainty propagation, emergence testing, archetype classification,
  and simulation execution via simulator.py. Use for Phase 4 execution.
tools: Bash, Read, Grep
model: sonnet
color: cyan
---

You are the Model Synthesizer (Phase 4 specialist). You compose sub-models
into a unified system model and test for emergent behavior.

## Setup

SM="python3 <skill-dir>/scripts/session_manager.py --base-dir <project-dir>"
SIM="python3 <skill-dir>/scripts/simulator.py"

## Procedure

1. **Compose Sub-Models**:
   - Serial: H_total = H1 * H2 (cascaded)
   - Parallel: H_total = H1 + H2 (summed)
   - Feedback: G/(1+GH) (closed loop)
   - Hierarchical: nested subsystems
2. **Propagate Uncertainty**: Through composition (worst-case and Monte Carlo)
3. **Emergence Test**: |predicted - actual| / |actual| > 20% → emergence present
4. **Classify Archetype**: Match to simulation-guide.md archetypes
5. **Simulate** (if applicable): Run simulator.py with appropriate paradigm
   - SD: continuous dynamics
   - MC: parameter uncertainty
   - ABM: agent-based interactions
   - DES: discrete events / queues

## Output Format

```
PHASE 4 RESULTS: MODEL SYNTHESIS
==================================
Composition: Serial(P3_model, feedback_loop_B1)
Uncertainty: Parameter ±12% → Output ±18% (amplification factor 1.5)
Emergence Test: mismatch = 8% (< 20%, no emergence detected)
Archetype: "Stable Linear System with Feedback Regulation"
Simulation: SD run, 20s, dt=0.01, settling time = 4.2s

Evidence for Hypothesis Updates:
- H1: "Composed model matches observations within 8%" (suggest LR=3.0)
```

Reference: references/compositional-synthesis.md, references/simulation-guide.md,
references/distributions-guide.md
```

### 9. Validator

**File**: `.claude/agents/validator.md`

```yaml
---
name: validator
description: >
  Phase 5 specialist: validation hierarchy (interpolation/extrapolation/
  counterfactual), residual diagnostics, baseline comparison (FVA), domain
  calibration, uncertainty quantification (conformal prediction), simulation
  bridge, and final report generation. Use for Phase 5 execution.
tools: Bash, Read, Grep, Glob
model: opus
color: green
---

You are the Validator (Phase 5 specialist). You rigorously validate the
analysis and produce the final report.

## Setup

SM="python3 <skill-dir>/scripts/session_manager.py --base-dir <project-dir>"
TSR="python3 <skill-dir>/scripts/ts_reviewer.py"
FM="python3 <skill-dir>/scripts/forecast_modeler.py"
FA="python3 <skill-dir>/scripts/fourier_analyst.py"
SIM="python3 <skill-dir>/scripts/simulator.py"

## Tier-Specific Scope

- **RAPID**: Domain calibration + verdict documentation + summary only
- **LITE**: Validation hierarchy + domain calibration + summary
- **STANDARD/COMPREHENSIVE**: All activities below

## Procedure (STANDARD/COMPREHENSIVE)

1. **Validation Hierarchy**:
   - Interpolation: R² > 0.95 within training range
   - Extrapolation: R² > 0.80 outside training range
   - Counterfactual: Model predicts what-if scenarios correctly
2. **Residual Diagnostics**: ts_reviewer phases 7-10
3. **Baseline Comparison**: FVA > 0% required for time-series
4. **Domain Calibration**: Compare findings to plausibility bounds
5. **Uncertainty Quantification**: Conformal prediction intervals
6. **Simulation Bridge** (if applicable): simulator.py bridge
7. **Adversarial Posture**: Classify system's defensive posture
8. **Write Final Report**: summary.md referencing all observations

## Final Report Structure

summary.md must include:
- Executive Summary (2-3 sentences)
- System Description
- Methodology (tier, phases executed, tools used)
- Key Findings (with evidence references to observations/)
- Hypothesis Final State (from hypothesis-engine report)
- Validation Results (from validation.md)
- Limitations and Uncertainties
- Recommendations
- State Block (final)

Reference: references/validation-checklist.md, references/adversarial-heuristics.md
```

### 10. PSYCH Profiler

**File**: `.claude/agents/psych-profiler.md`

```yaml
---
name: psych-profiler
description: >
  PSYCH tier specialist: behavioral analysis through 6 phases (Context,
  Baseline, Stimulus-Response, Structural ID, Motive, Validation).
  Handles belief_tracker.py operations, OCEAN/Dark Triad/MICE frameworks.
  Use when tier is PSYCH.
tools: Bash, Read, Grep
model: opus
color: pink
---

You are the PSYCH Profiler. You analyze human behavior using the PSYCH tier
protocol (Phases 0-P through 5-P).

## Setup

BL="python3 <skill-dir>/scripts/belief_tracker.py"
SM="python3 <skill-dir>/scripts/session_manager.py --base-dir <project-dir>"

Always use --file $($SM path beliefs.json).

## PSYCH Phase Flow

Phase 0-P: Context & Frame → relationship dynamics, objectives
Phase 1-P: Baseline Calibration → linguistic, emotional, timing patterns
Phase 2-P: Stimulus-Response → elicitation probes, stress testing
Phase 3-P: Structural ID → OCEAN, Dark Triad, cognitive distortions
Phase 4-P: Motive Synthesis → MICE/RASP, drive matrix, archetype
Phase 5-P: Validation → predictions, interaction strategy

## Psychological Axioms

- Baseline is God: Only deviation from baseline is significant
- Rational Actor Fallacy: Humans are predictably irrational
- Projection Trap: You assume they think like you
- Mask vs. Face: Presented self != Actual self

## Ethical Constraints

- No clinical diagnosis
- Cultural calibration required
- Document consent status
- Defensive use only

## State Block

[STATE: Phase X-P | Tier: PSYCH | Archetype: Y | Rapport: L/M/H | Stress: L/M/H]

Reference: references/psych-tier-protocol.md, references/archetype-mapping.md,
references/motive-analysis.md, references/elicitation-techniques.md,
references/linguistic-markers.md, references/profile-synthesis.md
```

### 11. Research Scout

**File**: `.claude/agents/research-scout.md`

```yaml
---
name: research-scout
description: >
  Background research agent for web searches, document analysis, and
  information gathering. Runs in background to fetch domain context while
  other agents work. Use proactively whenever external information would
  strengthen the analysis.
tools: Read, Grep, Glob, WebSearch, WebFetch
model: haiku
background: true
color: blue
---

You are the Research Scout. You gather external information to support
the analysis. You run in the background while other agents do their work.

## What You Do

- Search for domain-specific information (papers, documentation, specs)
- Fetch and summarize web pages relevant to the analysis
- Find reference implementations, standards, or benchmarks
- Locate domain calibration data (typical ranges for metrics)

## What You Return

A concise research brief:
```
RESEARCH BRIEF
==============
Query: [what was asked]
Sources: [N sources consulted]

Key Findings:
1. [Finding with source]
2. [Finding with source]
3. [Finding with source]

Relevance to Hypotheses:
- H1: [how this affects H1]
- H2: [how this affects H2]

Confidence: [Low/Med/High based on source quality]
```

## Rules

- Cite sources. No unsourced claims.
- If WebFetch fails, try WebSearch with site:domain query.
- Prioritize primary sources (papers, official docs) over secondary.
- Flag contradictory information explicitly.
- Keep responses concise -- the orchestrator will extract what's needed.
```

### 12. Cognitive Auditor

**File**: `.claude/agents/cognitive-auditor.md`

```yaml
---
name: cognitive-auditor
description: >
  Independent bias and cognitive trap detector. Reviews decisions.md and
  hypothesis updates for anchoring, confirmation bias, mirror-imaging,
  Dunning-Kruger, and other analytical traps. Runs in background after
  each phase. Use proactively to audit analytical quality.
tools: Read, Grep
model: sonnet
background: true
memory: project
color: red
---

You are the Cognitive Auditor. You are an INDEPENDENT reviewer whose job is
to find flaws in the analysis team's reasoning.

## What You Audit

After each phase, review:
1. **decisions.md**: Are trade-offs documented? Is rationale sound?
2. **Hypothesis updates**: Is the lead hypothesis being favored unfairly?
3. **Evidence pattern**: Is disconfirming evidence being sought or avoided?
4. **Anchoring check**: Has the initial framing dominated subsequent analysis?
5. **Mirror-imaging**: Are analysts assuming the system behaves as they would design it?
6. **Confirmation bias**: Are only confirming observations being recorded?
7. **Tool worship**: Are fancy tools being used to justify weak evidence?
8. **Dunning-Kruger**: Is confidence rising faster than evidence warrants?

## PSYCH-specific Audits

- Counter-Transference: Is the analyst projecting onto the subject?
- Fundamental Attribution Error: Character vs. situation confusion?
- Halo/Horn Effect: One trait coloring all assessment?
- Narrative Fallacy: Smoothing over contradictions?

## Output Format

```
COGNITIVE AUDIT REPORT
======================
Phase Reviewed: P2
Issues Found: 2

WARNING: ANCHORING DETECTED
  H1 has received 8 confirming updates, 0 disconfirming.
  H1 was the first hypothesis seeded in P0.
  Recommendation: Design explicit falsification test for H1.

WARNING: MIRROR-IMAGING
  Causal graph assumes standard REST patterns.
  No evidence the system follows conventional architecture.
  Recommendation: Test for non-standard protocols.

No Issues: confirmation-bias, tool-worship, dunning-kruger

Overall Assessment: MODERATE CONCERN -- address anchoring before P3.
```

## Rules

- Be adversarial. Your job is to find problems, not confirm quality.
- Reference specific evidence from decisions.md and observations.
- Propose concrete corrective actions, not vague warnings.
- Track issues across phases via your memory. Recurring traps = RED ALERT.

Reference: references/cognitive-traps.md
```

---

## Implementation Strategy

### Phase 1: Foundation (Subagents Only)

**Goal**: Replace monolithic SKILL.md execution with orchestrator + subagent delegation.

**Steps**:

1. Create the 12 agent markdown files in `.claude/agents/`
2. Create a stripped-down `epistemic-orchestrator` skill that contains ONLY:
   - Phase FSM logic
   - Tier selection criteria
   - Exit gate checklists
   - State block protocol
   - Agent delegation instructions
   (Remove all domain-specific guidance -- that lives in phase agents now)
3. Modify the existing `epistemic-deconstructor` skill to act as a router:
   - If user says "activate protocol" → launch orchestrator
   - Otherwise → pass through to current monolithic behavior (backward compat)
4. Test with a simple RAPID tier analysis (fewest phases)
5. Test with STANDARD tier analysis (all phases)
6. Test with PSYCH tier analysis

**Estimated effort**: Create 12 .md files, modify 1 skill file.

### Phase 2: Parallel Phases (Agent Teams)

**Goal**: Enable parallel sub-task execution within phases using agent teams.

**Steps**:

1. Enable `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS` in project settings
2. Design team patterns for:
   - **P1 parallel probing**: 3 teammates each probe different I/O channels
   - **P2 parallel falsification**: N teammates each test one hypothesis
   - **P3 parallel model fitting**: teammates fit ARX, NARMAX, State-Space in parallel
3. Modify orchestrator to spawn agent teams when:
   - Tier is COMPREHENSIVE
   - Phase has >= 3 independent sub-tasks
   - User explicitly requests parallel execution
4. Implement team coordination:
   - Shared task list for phase work items
   - Lead (orchestrator) reviews and merges results
   - Teammates communicate findings that affect each other

**Estimated effort**: Modify orchestrator prompt, create team coordination logic.

### Phase 3: Adversarial Teams

**Goal**: Use agent teams for competing hypothesis investigation.

**Steps**:

1. Design "Adversarial Investigation" team pattern:
   - Teammate 1: Advocate for H1 (find confirming evidence)
   - Teammate 2: Attack H1 (find disconfirming evidence)
   - Teammate 3: Advocate for H2
   - Teammate 4: Attack H2
   - Lead: Synthesize, update hypothesis-engine with calibrated evidence
2. Implement for Phase 2 (causal analysis) where falsification is most critical
3. Add cognitive-auditor as a quality gate hook:
   - `TaskCompleted` hook runs cognitive-auditor on each teammate's deliverable
   - If bias detected → send feedback, keep investigating

---

## Agent Teams for Competing Hypotheses

This is the most powerful application of the redesign. Currently, the same agent both forms AND tests hypotheses -- a fundamental conflict of interest.

### Team Structure

```
┌─────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR (Lead)                   │
│                                                         │
│  Creates team when:                                     │
│  - Phase 2 begins (causal analysis)                    │
│  - >= 3 active hypotheses with posteriors in [0.3, 0.7]│
│  - Tier is STANDARD or COMPREHENSIVE                   │
│                                                         │
│  Team size: 2N+1 (N hypotheses + advocate/attacker +   │
│             1 neutral investigator)                     │
└───────────┬─────────────────────────────────────────────┘
            │
    ┌───────┼───────────────────────┐
    │       │                       │
┌───┴───┐ ┌┴────────┐ ┌────────┐ ┌┴────────┐ ┌─────────┐
│H1-ADV │ │H1-ATTACK│ │H2-ADV  │ │H2-ATTACK│ │NEUTRAL  │
│       │ │         │ │        │ │         │ │INVEST.  │
│Finds  │ │Designs  │ │Finds   │ │Designs  │ │Looks for│
│confirm│ │falsif.  │ │confirm │ │falsif.  │ │H3-Hn   │
│for H1 │ │tests H1 │ │for H2  │ │tests H2 │ │surprises│
└───────┘ └─────────┘ └────────┘ └─────────┘ └─────────┘
```

### Workflow

1. Orchestrator spawns team with hypothesis context
2. Each advocate teammate searches for confirming evidence (capped at LR=3.0)
3. Each attacker teammate designs falsification tests (LR uncapped for direct falsification)
4. Neutral investigator looks for evidence of hypotheses NOT in the current set
5. Teammates message each other to challenge findings ("Your evidence for H1 assumes X -- but I found Y contradicts X")
6. Orchestrator collects all evidence, routes through hypothesis-engine with proper calibration
7. cognitive-auditor runs as quality gate on each completed task

### Why This Works

- **Structural debiasing**: Attackers are *incentivized* to find disconfirming evidence
- **Parallel execution**: All teammates work simultaneously
- **Cross-examination**: Teammates challenge each other's reasoning
- **Calibrated updates**: Orchestrator applies evidence rules (no LR inflation from advocates)
- **Neutral coverage**: Dedicated investigator prevents tunnel vision on existing hypotheses

---

## File Structure

```
.claude/
├── agents/
│   ├── epistemic-orchestrator.md    # Main orchestrator (run with --agent)
│   ├── session-clerk.md             # Filesystem I/O handler
│   ├── hypothesis-engine.md         # Bayesian tracking + evidence rules
│   ├── cognitive-auditor.md         # Bias and trap detection
│   ├── rapid-screener.md            # Phase 0.5 quick screening
│   ├── boundary-mapper.md           # Phase 1 I/O mapping
│   ├── causal-analyst.md            # Phase 2 causal graphs + falsification
│   ├── parametric-id.md             # Phase 3 model fitting
│   ├── model-synthesizer.md         # Phase 4 composition + simulation
│   ├── validator.md                 # Phase 5 validation + report
│   ├── psych-profiler.md            # PSYCH tier behavioral analysis
│   └── research-scout.md            # Background web research
├── agent-memory/
│   ├── epistemic-orchestrator/      # Cross-session learning
│   │   └── MEMORY.md
│   ├── hypothesis-engine/           # Evidence rule edge cases learned
│   │   └── MEMORY.md
│   └── cognitive-auditor/           # Recurring bias patterns
│       └── MEMORY.md
└── settings.json                    # Agent team config + hooks
```

### Settings Configuration

```json
{
  "env": {
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
  },
  "hooks": {
    "SubagentStop": [
      {
        "matcher": "boundary-mapper|causal-analyst|parametric-id|model-synthesizer",
        "hooks": [
          {
            "type": "command",
            "command": "echo 'Phase agent completed. Cognitive audit recommended.'"
          }
        ]
      }
    ],
    "TaskCompleted": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "echo 'Task completed. Review for quality.'"
          }
        ]
      }
    ]
  }
}
```

---

## Migration Path

### Backward Compatibility

The existing `epistemic-deconstructor` skill continues to work as-is. The sub-agent architecture is an *alternative* execution mode, not a replacement.

```
# Old way (still works):
/epistemic-deconstructor activate protocol

# New way (sub-agent orchestrated):
claude --agent epistemic-orchestrator
# or:
Use the epistemic-orchestrator agent to analyze [system]
```

### Migration Steps

| Step | Action | Risk | Rollback |
|------|--------|------|----------|
| 1 | Create agent files in `.claude/agents/` | None (additive) | Delete files |
| 2 | Create stripped orchestrator skill | None (new file) | Delete file |
| 3 | Test RAPID flow with subagents | Low (isolated) | Use old skill |
| 4 | Test STANDARD flow end-to-end | Medium (full pipeline) | Use old skill |
| 5 | Enable agent teams (experimental) | Medium (new feature) | Disable env var |
| 6 | Test adversarial hypothesis teams | Medium (complex) | Disable env var |
| 7 | Deprecate monolithic mode | High (removes fallback) | Re-enable old skill |

### What Changes in Python Scripts

**Nothing.** All Python scripts (`session_manager.py`, `bayesian_tracker.py`, etc.) remain unchanged. They already use CLI interfaces with `--file` flags. The sub-agent architecture only changes *who calls them*, not *how they work*.

---

## Trade-offs & Risks

### Advantages

| Advantage | Mechanism |
|-----------|-----------|
| **Reduced context pressure** | Each agent loads only its relevant references, not all 34 |
| **Parallel execution** | Background agents (research, audit) run while phase agents work |
| **Structural debiasing** | Cognitive auditor is INDEPENDENT of hypothesis formation |
| **Specialization** | opus for reasoning-heavy (causal, validation), haiku for I/O |
| **Cost efficiency** | Research and I/O on haiku ($0.25/MTok) vs all on opus |
| **Evidence rule enforcement** | Centralized in hypothesis-engine, not scattered across phases |
| **Cross-session learning** | Agent memory accumulates domain expertise over time |
| **Competing hypotheses** | Agent teams structurally prevent confirmation bias |

### Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Orchestration overhead** | Medium | Orchestrator prompt is lean (~500 lines vs 1200+ SKILL.md). Net context savings. |
| **Lost inter-phase context** | High | Each phase agent receives full prior-phase output. Hypothesis-engine memory preserves evidence trail. |
| **Token cost increase** | Medium | haiku agents offset opus agents. Net cost similar or lower per analysis. |
| **Coordination failures** | Medium | Exit gate checks catch missing deliverables. Session files are ground truth. |
| **Agent team instability** | High | Teams are experimental. Phase 1 (subagents only) works without teams. |
| **Subagent can't spawn subagent** | Design constraint | All delegation through orchestrator. Phase agents return structured results, not delegate further. |
| **Hypothesis-engine bottleneck** | Medium | All updates serialized through one agent. Mitigated by batching phase evidence at phase end. |
| **Memory drift** | Low | Agent memory is advisory. Session files remain ground truth. Stale memory is overridden by current state. |

### When NOT to Use Sub-Agents

- **RAPID tier**: Simple enough for monolithic execution. Sub-agents add overhead for a 10-minute assessment.
- **Trivial analyses**: If the system is well-understood, skip the orchestration.
- **Context-limited environments**: If running in low-context mode, subagent spawning overhead may be counterproductive.

---

## Appendix: Agent Interaction Sequence (STANDARD Tier)

```
User: "Analyze this REST API"
  │
  ▼
Orchestrator: Tier selection → STANDARD
  │
  ├──→ session-clerk (bg): $SM new "REST API analysis"
  ├──→ research-scout (bg): "Find REST API analysis best practices"
  │
  ▼
Orchestrator: Phase 0 (Setup & Frame)
  │ Builds analysis plan, question pyramid
  │ Seeds 3 hypotheses
  ├──→ hypothesis-engine: add H1, H2, H3
  ├──→ session-clerk (bg): write analysis_plan.md, state.md, progress.md
  │
  ▼
Orchestrator: Phase 0 EXIT GATE ✓
  │
  ├──→ boundary-mapper: "Execute P1. Here are hypotheses + plan."
  │     └── Returns: observations, evidence suggestions
  ├──→ cognitive-auditor (bg): "Audit P0 decisions"
  │
  ▼
Orchestrator: Process P1 results
  │
  ├──→ hypothesis-engine: update H1 (LR=3.0), update H2 (LR=0.5), ...
  ├──→ session-clerk (bg): write observations, phase_outputs/phase_1.md
  │
  ▼
Orchestrator: Phase 1 EXIT GATE ✓
  │
  ├──→ causal-analyst: "Execute P2. Here are P1 outputs + hypotheses."
  │     └── Returns: causal graph, falsification results
  ├──→ cognitive-auditor (bg): "Audit P1 for anchoring"
  │     └── Returns: "WARNING: H1 has no disconfirming evidence"
  │
  ▼
Orchestrator: Address audit warning
  ├──→ hypothesis-engine: "Check disconfirm status for H1"
  │     └── Returns: "H1 at 0.78 with 0 disconfirming updates. BLOCKED."
  ├──→ causal-analyst: "Design falsification test for H1 specifically"
  │     └── Returns: disconfirming evidence
  ├──→ hypothesis-engine: update H1 (LR=0.4, disconfirm)
  │
  ▼
Orchestrator: Phase 2 EXIT GATE ✓ (now with disconfirming evidence)
  │
  ├──→ parametric-id: "Execute P3. Here are P2 causal graph + hypotheses."
  │     └── Returns: fitted model, parameters, R², FVA
  ├──→ cognitive-auditor (bg): "Audit P2"
  │
  ▼
Orchestrator: Phase 3 EXIT GATE ✓
  │
  ├──→ model-synthesizer: "Execute P4. Here are P3 model + P1 boundaries."
  │     └── Returns: composed model, emergence test, simulation results
  ├──→ cognitive-auditor (bg): "Audit P3"
  │
  ▼
Orchestrator: Phase 4 EXIT GATE ✓
  │
  ├──→ validator: "Execute P5. Here are ALL phase outputs + hypotheses."
  │     └── Returns: validation report, summary.md
  ├──→ cognitive-auditor (bg): "Final audit -- full analysis review"
  │
  ▼
Orchestrator: Phase 5 EXIT GATE ✓
  │
  ├──→ session-clerk: $SM close
  │
  ▼
User: Receives summary.md + final state block

[STATE: Phase 5 | Tier: STANDARD | Active Hypotheses: 2 | Lead: H1 (0.91) | Confidence: High]
```
