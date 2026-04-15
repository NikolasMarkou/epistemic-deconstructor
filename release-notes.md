## What's New in v7.13.0

### Agent Enforcement Layer for v7.12 Protocol

v7.12.0 added Scope Interrogation (Phase 0.7, H_S standing pair, trigger S1) to `SKILL.md` and the reference layer, but the sub-agent definitions in `src/agents/` were not updated to *enforce* those features at their own exit gates. This release wires enforcement into 8 agents so the protocol is self-enforcing.

### Changed
- **`validator.md`** — mandatory **Scope Completeness Check** (halts `summary.md` if `[H_S_prime]` > 0.40 with no prior `S1` reopen logged) and **P5.1–P5.4 trigger evaluation**.
- **`epistemic-orchestrator.md`** — replaced the 2-line exit-gate step with an explicit **7-step Gate Check Procedure** covering file completeness, hypothesis state review, H_S pair check, U1–U4 + S1 + phase-specific trigger evaluation, reopen-or-advance decision, and cognitive-auditor launch.
- **`hypothesis-engine.md`** — concrete grep-based **disconfirm-before-confirm verification** (blocks any update that would push a posterior past 0.80 without a disconfirming test) and **H_S pair seeding guard** (blocks Phase 1 updates until the standing pair is seeded).
- **`psych-profiler.md`** — added **Phase 0-P.7** (scope-auditor delegation with life-context framing), **PP.1–PP.4** trigger evaluation, and **Phase 5-P Scope Completeness Check** on `beliefs.json`.
- **`boundary-mapper.md`** — now actually invokes `ts_reviewer.py quick` + `fourier_analyst.py quick` / `analyze` for numeric I/O data (SKILL.md called for this but the agent never ran either tool). Added **P1.1/P1.2 + U1/U2** trigger evaluation.
- **`causal-analyst.md`** — **P2.1/P2.2/P2.3 + U1/U3/U4/S1** trigger evaluation.
- **`parametric-id.md`** — **P3.1/P3.2/P3.3 + S1** trigger evaluation and a mandatory **post-fit `scope_auditor.py residual-match`** step that catches omitted drivers at the earliest point they become detectable.
- **`model-synthesizer.md`** — archetype classification now cross-checks `archetype-accomplices.md` for S1 signals; `distributions-guide.md` reference added for MC/ABM/DES parameter distribution selection; **P4.1/P4.2 + U1/S1** trigger evaluation.

### Unchanged (intentionally)
`session-clerk.md`, `research-scout.md`, `rapid-screener.md`, `scope-auditor.md`, `cognitive-auditor.md` — already aligned with v7.12.

### Rationale
The protocol is only as good as the agent that executes it. An un-enforced evidence rule is indistinguishable from no evidence rule. This release brings all agents up to spec so `claude --agent epistemic-orchestrator` produces a result that matches SKILL.md's written guarantees.

No `scripts/`, `references/`, `config/`, or `tests/` changes — this is a pure agent-definition release. 8 files modified, 260 insertions, 10 deletions. Existing 466-test pytest suite unaffected.

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
