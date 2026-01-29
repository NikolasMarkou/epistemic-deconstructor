# Setup & Frame Definition Reference

Techniques for Phase 0 metacognitive setup and frame definition.

## Table of Contents

- [Context Assessment](#context-assessment)
- [Question Pyramid](#question-pyramid)
- [Rumsfeld Matrix](#rumsfeld-matrix)
- [Hypothesis Generation](#hypothesis-generation)
- [Tier Selection Decision Tree](#tier-selection-decision-tree)
- [Cognitive Trap Awareness](#cognitive-trap-awareness)
- [Analysis Frameworks](#analysis-frameworks)
- [Output: Analysis Plan Document](#output-analysis-plan-document)

---

## Context Assessment

### Position Matrix
| Dimension | Options | Implications |
|-----------|---------|--------------|
| Access | Insider / Outsider | Determines available channels |
| Intent | Cooperative / Adversarial | Shapes stealth requirements |
| Authority | Authorized / Unauthorized | Legal/ethical constraints |
| Knowledge | Expert / Novice | Starting hypothesis quality |

### Constraint Inventory
```
TIME: [hours/days/weeks available]
TOOLS: [available analysis tools]
ACCESS: [I/O channels accessible]
ETHICS: [prohibited actions]
LEGAL: [regulatory constraints]
RESOURCES: [compute/storage/personnel]
```

### Information-Theoretic Baseline
```
System complexity: C = log₂(|state space|) bits
Channel capacity: B = max bits extractable per experiment
Minimum experiments: N_min = C / B

Example:
- 256 possible states → C = 8 bits
- Binary yes/no questions → B = 1 bit/experiment
- Minimum 8 experiments needed for full characterization
```

## Question Pyramid

### Levels of Understanding

**L1: Behavioral (What)**
- What outputs does the system produce?
- What inputs does it accept?
- What is the I/O relationship?
- *Test*: Can predict output given input with >90% accuracy

**L2: Functional (How)**
- How is input transformed to output?
- What are the major processing stages?
- What are the control flows?
- *Test*: Can explain each processing step

**L3: Structural (Why)**
- Why was this architecture chosen?
- What are the design tradeoffs?
- What constraints drove decisions?
- *Test*: Can predict design choices in similar systems

**L4: Parametric (How much)**
- What are the numerical parameters?
- What are the tolerances?
- What is the precision of estimates?
- *Test*: <5% error on measurable quantities

**L5: Generative (Can replicate)**
- Can build a functionally equivalent system?
- Does replica pass indistinguishability tests?
- *Test*: Replica outputs statistically indistinguishable

### Question Decomposition
```
Main Question: "How does this system work?"
    ├─ L1: "What does it do?"
    │   ├─ "What inputs does it accept?"
    │   ├─ "What outputs does it produce?"
    │   └─ "What is the input-output mapping?"
    ├─ L2: "How does it do it?"
    │   ├─ "What are the processing stages?"
    │   ├─ "What algorithms are used?"
    │   └─ "What is the data flow?"
    ├─ L3: "Why this design?"
    │   ├─ "What were the constraints?"
    │   ├─ "What tradeoffs were made?"
    │   └─ "What alternatives were rejected?"
    └─ L4: "What are the parameters?"
        ├─ "What are the numerical values?"
        ├─ "What are the tolerances?"
        └─ "What is the uncertainty?"
```

## Rumsfeld Matrix

### Framework
|  | Known | Unknown |
|--|-------|---------|
| **Known** | Facts (document) | Questions (investigate) |
| **Unknown** | Blind spots (probe randomly) | Surprises (stay alert) |

### Population Procedure
1. **Known Knowns**: List all established facts about system
2. **Known Unknowns**: List all questions you know to ask
3. **Unknown Knowns**: Actively search for assumptions you haven't validated
4. **Unknown Unknowns**: Allocate time for random exploration

### Example
```
Known Knowns:
- System is a web application
- Written in Python
- Uses PostgreSQL database

Known Unknowns:
- Authentication mechanism?
- API structure?
- Caching strategy?

Unknown Knowns (discovered later):
- Assumed REST but actually GraphQL
- Assumed single-tenant but multi-tenant

Unknown Unknowns (surprises):
- Hidden admin interface
- Undocumented debug endpoint
```

## Hypothesis Generation

### Strategies

**1. Archetype Matching**
```
Observe initial behavior → Match to known archetype → Generate archetype-specific hypotheses

Example:
- Observation: Request-response pattern
- Archetype match: Client-server
- Hypotheses: H1=REST API, H2=GraphQL, H3=gRPC
```

**2. Failure Mode Analysis**
```
For each component, ask:
- What could go wrong?
- How would failure manifest?
- What would cause this failure?

Reverse into hypotheses about normal operation.
```

**3. Adversarial Thinking**
```
If I were designing this system to deceive analysts:
- What would I hide?
- How would I mislead?
- What would be too convenient?

Generate H_adversarial for each possibility.
```

### Hypothesis Quality Criteria
- **Falsifiable**: Can design test to refute
- **Specific**: Makes precise predictions
- **Relevant**: Addresses core questions
- **Parsimonious**: No unnecessary complexity
- **Distinguishable**: Different from alternatives

### Initial Hypothesis Template
```
Hypothesis ID: H[n]
Phase: P0
Statement: [Clear, falsifiable claim]
Predictions:
  - If true: [Observable consequence 1]
  - If true: [Observable consequence 2]
Prior: [0.0-1.0, with justification]
Key test: [Most discriminating experiment]
Alternatives: [What hypothesis would explain if this is wrong]
```

## Tier Selection Decision Tree

```
START
├─ Is system archetype already known?
│   ├─ Yes → Is system stable (no changes expected during analysis)?
│   │         ├─ Yes → Is single function being analyzed?
│   │         │         ├─ Yes → LITE tier
│   │         │         └─ No → STANDARD tier
│   │         └─ No → STANDARD tier (minimum)
│   └─ No → Continue below
├─ Is adversary expected?
│   ├─ Yes → COMPREHENSIVE tier
│   └─ No → Continue
├─ Are there >15 components visible?
│   ├─ Yes → COMPREHENSIVE tier
│   └─ No → Continue
├─ Is multi-domain (software + hardware + organizational)?
│   ├─ Yes → COMPREHENSIVE tier
│   └─ No → STANDARD tier
```

## Cognitive Trap Awareness

### Traps to Watch For
| Trap | Sign | Countermeasure |
|------|------|----------------|
| Mirror-imaging | "I would do X, so they did" | List 3 alternatives |
| Confirmation bias | Only finding supporting evidence | Seek disconfirmation |
| Anchoring | First hypothesis dominates | Generate parallel hypotheses |
| Availability | Recent/vivid events overweighted | Use systematic sampling |
| Dunning-Kruger | Early overconfidence | Track prediction accuracy |

### Pre-Analysis Checklist
- [ ] Have I listed my assumptions?
- [ ] Have I generated ≥3 hypotheses?
- [ ] Have I identified what would refute each hypothesis?
- [ ] Have I allocated time for random exploration?
- [ ] Have I identified my potential blind spots?

## OSINT and Reconnaissance Tools

For Phase 0 information gathering before active probing.

### Web/Network Reconnaissance

| Tool | Purpose | Use |
|------|---------|-----|
| **nmap** | Port scanning, service detection | Discover network I/O channels |
| **whois** | Domain registration info | Understand ownership, timeline |
| **dig/nslookup** | DNS queries | Map infrastructure |
| **Shodan/Censys** | Internet-wide scanning | Find exposed services |

### Code/Repository Intelligence

| Tool | Purpose | Use |
|------|---------|-----|
| **GitHub search** | Code patterns, dependencies | Find similar implementations |
| **VirusTotal** | Hash/URL reputation | Check for known malicious samples |
| **Libraries.io** | Dependency tracking | Map software supply chain |

### Social/Organization

| Tool | Purpose | Use |
|------|---------|-----|
| **LinkedIn** | Organization structure | Identify key personnel |
| **Wayback Machine** | Historical snapshots | Track changes over time |
| **SEC filings** | Corporate disclosures | Financial/organizational context |

### Automated OSINT Frameworks

| Framework | Focus |
|-----------|-------|
| **theHarvester** | Email, subdomain, host discovery |
| **Maltego** | Visual link analysis (commercial) |
| **SpiderFoot** | Automated OSINT collection |
| **Amass** | Subdomain enumeration |

---

## Analysis Frameworks

### FIRST Malware Analysis Framework
| Phase | Activities |
|-------|------------|
| 1. Case Framing | Define scope, objectives, constraints |
| 2. Preliminary | Entropy analysis, strings, PE headers, hash lookup |
| 3. Behavioral | Sandbox execution, API monitoring, network capture |
| 4. Code Analysis | Unpack → Static (Ghidra) → Dynamic (debugger) → Document |

### Chaos Engineering Framework
1. **Define steady state**: Measurable baseline (latency p99, error rate, throughput)
2. **Hypothesize**: Steady state continues under perturbation
3. **Introduce variables**: Instance failure, network partition, latency, resource exhaustion
4. **Measure divergence**: Compare stressed vs baseline
5. **Rollback**: Always restore normal operation

## Output: Analysis Plan Document

```markdown
# Analysis Plan: [System Name]

## Context
- Position: [insider/outsider, cooperative/adversarial]
- Access: [available I/O channels]
- Constraints: [time, tools, ethics, legal]
- System Type: [white/grey/black box]

## Fidelity Target
- Target Level: L[1-5]
- Justification: [why this level]
- Success Criteria: [specific metrics]

## Tier
- Selected: [LITE/STANDARD/COMPREHENSIVE]
- Rationale: [decision tree path]

## Questions
- Primary: [main question]
- Secondary: [supporting questions]

## Initial Hypotheses
| ID | Statement | Prior | Key Test |
|----|-----------|-------|----------|
| H1 | ... | 0.X | ... |
| H2 | ... | 0.X | ... |
| H3 | ... | 0.X | ... |

## Known I/O Channels
- Inputs: [list]
- Outputs: [list]
- Potential side channels: [list]

## Rumsfeld Matrix
- Known Knowns: [list]
- Known Unknowns: [list]
- Suspected Unknown Knowns: [list]

## Time Allocation
- Phase 0: [X%] - COMPLETE
- Phase 1: [X%]
- Phase 2: [X%]
- Phase 3: [X%]
- Phase 4: [X%]
- Phase 5: [X%]

## Risk Assessment
- Potential adversarial response: [low/medium/high]
- Observer effect risk: [low/medium/high]
- Time overrun risk: [low/medium/high]
```
