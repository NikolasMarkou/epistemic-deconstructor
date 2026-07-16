# Warning Reception — Observer-Side Community Response Dynamics

This reference covers how a third party — an institution, community, or audience — receives, discounts, reframes, or acts on a warning about an external hazard: the observer side of the warner-audience relationship. It is analysis material for scope interrogation (Phase 0.7) and PSYCH-adjacent source evaluation. It does NOT change the PSYCH tier's single-subject unit of analysis.

## Table of Contents

- [Scope & Disambiguation](#scope--disambiguation)
- [The Anti-Self-Sealing Rule (HARD)](#the-anti-self-sealing-rule-hard)
- [Warning-Chain Break-Point Taxonomy (C11.2)](#warning-chain-break-point-taxonomy-c112)
- [Observer-Side Response Patterns (C12)](#observer-side-response-patterns-c12)
- [Counterfeit-Witness Discriminators (C13)](#counterfeit-witness-discriminators-c13)
- [Cross-References](#cross-references)

---

## Scope & Disambiguation

**"Warning" in this file** means: a claim by an identifiable warner that a specific hazard threatens an audience, delivered before the outcome is known. This is NOT any of the repo's three existing senses of the word:

1. **Refusal Protocol** (`SKILL.md`) — the orchestrator's procedural refusal messages when asked to bypass the phase FSM. Procedural compliance language, not hazard claims.
2. **Saturation warnings** (`references/evidence-calibration.md:155-157`) — internal analytical alerts that a posterior is approaching a CONFIRMED/REFUTED transition boundary. Instrumentation, not hazard claims.
3. **Warning Archetypes** (`references/archetype-mapping.md:243`) — personality risk-patterns OF a profiled subject — behavioral archetypes (`references/archetype-mapping.md`, PSYCH personas). Those describe risk the *subject* poses; this file describes how a *receiving community* handles a hazard claim.

**"Archetype" tagging rule.** This file may touch both of the repo's archetype vocabularies. Every use of "archetype" here — and in any analysis artifact citing this file — must be tagged inline as either a **behavioral archetype** (`references/archetype-mapping.md`, PSYCH personas) or a **system archetype** (`config/archetypes.json`, scope-interrogation domain shapes). An untagged "archetype" is a drafting error.

**Unit of analysis.** The observed unit here is the RECEIVING community's or institution's documented response — not the warner's psychology, and not a PSYCH profile. For warner-side credibility discrimination (the Troubler-vs-Watchman question: is this warner credible?), see `references/elicitation-techniques.md`. PSYCH profiling stays subject-singular; when a PSYCH session encounters observer-side reception material, it is scored under this file's mechanisms as source-evaluation context, never as a second profiled subject.

---

## The Anti-Self-Sealing Rule (HARD)

Dismissal, mockery, or punishment of a warning is NEVER evidence that the warning was true.

Every mechanism below scores the RESPONSE — as a property of the responder — and caps or adjusts source weights. None of them confirms the warned-of hazard. A warning is confirmed only by evidence about the hazard itself. If an analysis ever finds itself reasoning "they dismissed it, so there must be something to it," that inference is void and must be struck from the evidence trail.

---

## Warning-Chain Break-Point Taxonomy (C11.2)

For each documented unheeded warning, classify the break point — the mechanism by which the warning failed to produce action. Each class carries its own documentary test.

| Class | Definition | Documentary test |
|-------|------------|------------------|
| **DISMISSED** | Rejection cites messenger attributes, not content | Rejection record contains zero content propositions |
| **DELAYED** | Acknowledged, but no action owner within N days | No named owner in the record within the institution's own response-time norm |
| **DILUTED** | Scope reduced with no stated evidence basis | Scope-reduction memo lacks evidentiary citation |
| **BURIED** | No named owner appears in the record at all | Archival absence of assignment |
| **REFRAMED** | Restated in liability/PR language | Response document's register shifts; original hazard terms absent |
| **POLITICIZED** | Response splits on faction lines | Documented positions correlate with affiliation, not with evidence access |
| **FATIGUED** | Response latency rises across repeated warnings | Measurable latency trend across the warning sequence |
| **SYMBOLIC** | Acknowledgment artifact exists, zero tracked action items | Artifact present; action-item record empty |

**Hard rule:** an unclassifiable warning is FLAGGED as unclassified — never forced into a class. Forcing a class fabricates a mechanism the record does not support.

**Validation test:** every classification must cite the documentary record satisfying that class's test (the rejection memo, the assignment log, the latency series). A classification without its documentary test is void.

---

## Observer-Side Response Patterns (C12)

Recurring patterns in how receiving communities and institutions handle warnings. Each pattern is a property of the response record; per the Anti-Self-Sealing Rule, none of them bears on whether the warned-of hazard is real.

### Temporal Mismatch

The response horizon is shorter than the hazard horizon — a quarterly-cycle response to a decade-scale warning. The institution is answering a different question than the one asked: it addresses the warning inside its own planning cadence rather than the hazard's timescale.

**Validation test:** compare the documented response horizon (budget cycle, review period, mandate length) against the warning's stated hazard horizon; the pattern fires only when both horizons are documented and the response horizon is shorter.

### Private Agreement / Public Abandonment

Actors agree with the warner privately but withhold public support. The warning's failure is then attributed to lack of support that privately existed. This pattern requires positive documentation on both sides.

**Validation test:** documented private assent (mail, meeting minutes, contemporaneous notes) paired with an absent public record from the same actor. Missing documentation on either side = UNKNOWN, never assumed.

### Early-Accuracy Punishment

The earliest accurate warner bears sanction while later repeaters of the same claim gain credit. The community's incentive gradient rewards timing (asserting after consensus shifts), not accuracy.

**Validation test:** a sequence ledger (see `references/abductive-reasoning.md` for provenance and sequence documentation) shows sanction-at-assertion for the earliest assertor followed by consensus adoption without credit reassignment.

### Vindication Without Repentance

The warned-of outcome occurs; the record shows no revision of the original dismissal and no change in the dismissers' standing. The institution absorbs the outcome without updating its warning-handling process — a predictor of repeat failure on the next warning.

**Validation test:** post-outcome record lacks any documented re-evaluation of the dismissal decision or its authors.

### Perfect-Messenger Demand

The response sets warner-conduct preconditions unrelated to claim content: the warning would be heard if only the warner were calmer, better credentialed, less affiliated. Content evaluation is deferred indefinitely behind messenger requirements. This pairs with the treatment of tone-only rebuttals as non-evidence in `references/evidence-calibration.md`.

**Validation test:** the stated rejection criteria enumerate messenger properties (tone, style, affiliation, standing) while the warning's content propositions remain unaddressed in the record.

---

## Counterfeit-Witness Discriminators (C13)

Not every warner is a genuine witness to a hazard; some produce alarm because alarm is rewarded. These discriminators separate the two WITHOUT conspiracy assumptions.

### Engagement-Economics Selection Model (C13.2)

Platforms and audiences reward alarm output along attention gradients: alarm content earns reach, reach earns revenue and status, and producers who supply alarm are selected for — no coordination required. Selection does the work that a conspiracy theory would wrongly attribute to intent. This is the same incentive-covariate reasoning as the organizational_actor and research_knowledge_producer entries in `references/archetype-accomplices.md` (system archetypes, `config/archetypes.json`): reward structures select which outputs get produced, independent of any producer's sincerity.

**Validation test:** the model predicts alarm output tracks the platform's documented reward gradient (engagement metrics, monetization thresholds), not the hazard evidence base; test by comparing the source's output volume/timing against both series where documented.

### Discriminators

Each discriminator is a checkable property of the warning and its context. Score each independently.

- **Cost borne at assertion** — did the warning cost the warner anything at the time it was made (position, income, standing, access)? *Documentary test:* dated record of a sanction, loss, or forgone benefit attributable to the assertion.
- **Patron exposure** — does the warning implicate the warner's own funders, platform, or patrons? *Documentary test:* map the warning's named culpable parties against the warner's documented funding/platform relationships; overlap = exposure.
- **Self-implication** — does the change the warner demands bind the warner's own side, audience, or interests? *Documentary test:* enumerate the warning's demanded actions; check whether any bind the issuer's side. A warning that demands zero cost from its own audience is a counterfeit-witness indicator (see the Institutional Response Red Flags in `references/red-flags.md` for the corresponding red-flag treatment).

**Validation test:** each discriminator resolves to a documented YES/NO or UNKNOWN; a discriminator scored without its documentary basis is void.

**Missing-field rule:** any discriminator that cannot be documented is recorded UNKNOWN — never inferred in either direction. An UNKNOWN neither penalizes nor credits the warner.

---

## Cross-References

- `references/red-flags.md` — Institutional Response Red Flags: the red-flag treatment of defensive institutional responses; LR-cap consequences.
- `references/archetype-accomplices.md` — organizational_actor (system archetype, `config/archetypes.json`): incentive covariates behind institutional response behavior.
- `references/evidence-calibration.md` — LR conventions, saturation warnings; scoring of content-free rebuttals.
- `references/cognitive-traps.md` — PSYCH-tier traps, incl. Trap 24 (Interiority Refusal): the receiving institution's actors also have interiors.
- `references/elicitation-techniques.md` — warner-side credibility probing (Troubler-vs-Watchman question).
- `references/abductive-reasoning.md` — provenance discipline and sequence documentation for contested claims.
- `references/scope-interrogation.md` — Phase 0.7 mechanisms; community/stakeholder reception as scope-exogeneity material.
- `references/psych-tier-protocol.md` — the single-subject unit of analysis this file deliberately does not alter.
