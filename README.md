# Epistemic Deconstructor

![License](https://img.shields.io/badge/license-GPLv3-blue.svg)
![Protocol](https://img.shields.io/badge/Protocol-v6.0-green.svg)
![Type](https://img.shields.io/badge/Claude-Skill-purple.svg)

**A systematic framework for reverse engineering unknown systems.**

The **Epistemic Deconstructor** is a specialized capability designed for Large Language Models (specifically Claude) to assist in the rigorous analysis of black-box systems. Whether analyzing software binaries, biological processes, organizational structures, or mechanical systems, this skill transforms epistemic uncertainty into predictive control through a 6-phase scientific protocol.

## Core Philosophy

Most reverse engineering attempts fail due to cognitive bias (mirror-imaging, confirmation bias) or lack of structure. This skill enforces the **Epistemic Deconstruction Protocol v6.0**, which dictates:

1.  **Falsification over Confirmation:** We do not prove hypotheses true; we fail to prove them false.
2.  **Quantified Uncertainty:** We track confidence using Bayesian updates, not gut feelings.
3.  **Model Synthesis:** We build compositional models (math, logic, or state machines) to predict system behavior.

## Repository Structure

```text
epistemic-deconstructor/
├── README.md                 # This file
├── LICENSE                   # GNU GPL v3
├── SKILL.md                  # The core system prompt/instruction set
├── references/               # Knowledge base for the AI
│   ├── boundary-probing.md   # Techniques for I/O characterization
│   ├── causal-techniques.md  # Methods for establishing causality
│   ├── cognitive-traps.md    # Countermeasures for analytical bias
│   ├── compositional-synthesis.md # Math for combining sub-models
│   ├── setup-techniques.md   # Phase 0 framing and Rumsfeld matrices
│   ├── system-identification.md # Parametric estimation algorithms
│   └── tools-sensitivity.md  # Binary tools & sensitivity analysis
└── scripts/
    └── bayesian_tracker.py   # CLI tool for tracking hypothesis confidence
```

## %Installation & Usage

### Method 1: Claude Project (Recommended)

1.  Create a new **Project** in Claude.
2.  Upload all files from the `references/` directory to the Project Knowledge.
3.  Copy the contents of `SKILL.md` into the **Custom Instructions** for the project.
4.  (Optional) Upload `scripts/bayesian_tracker.py` if you want Claude to write code to interact with it.

### Method 2: Session Context

1.  Attach `SKILL.md` and the relevant reference files (e.g., `cognitive-traps.md`, `setup-techniques.md`) to your chat.
2.  Start your prompt with: *"Activate Epistemic Deconstruction Protocol. I have a target system I need to analyze..."*

## The Protocol (v6.0)

The skill guides the user through six distinct phases. You must select a tier (**LITE**, **STANDARD**, or **COMPREHENSIVE**) before beginning.

| Phase | Name | Objective | Output |
|-------|------|-----------|--------|
| **0** | **Setup & Frame** | Define scope, adversary profile, and hypotheses. | Analysis Plan & Question Pyramid |
| **1** | **Boundary Mapping** | Enumerate inputs/outputs and stimulus response. | I/O Surface Map |
| **2** | **Causal Analysis** | Map internal dependencies and graph structure. | Causal Graph |
| **3** | **Parametric ID** | Fit mathematical models (ARX, State-Space). | Equations & Parameters |
| **4** | **Synthesis** | Combine sub-models and detect emergence. | Unified Model |
| **5** | **Validation** | Red-teaming and adversarial surface mapping. | Validation Report |

## Included Tools

### Bayesian Tracker (`scripts/bayesian_tracker.py`)

A Python CLI tool is included to formally track hypothesis confidence using Bayesian inference. It moves analysis away from "I think maybe..." to "Posterior probability is 0.85 based on likelihood ratio 10.0."

**Usage:**

```bash
# Add a new hypothesis
python scripts/bayesian_tracker.py add "System uses a State Machine architecture" --prior 0.5

# Update with evidence (Likelihood Ratio > 1 confirms, < 1 disconfirms)
python scripts/bayesian_tracker.py update H1 "Observed discrete transitions" --preset strong_confirm

# Generate report
python scripts/bayesian_tracker.py report
```

### Reference Modules

The `references/` folder contains specific technical knowledge implementation details that the AI calls upon during analysis:
*   **Cognitive Traps**: forces the AI to check for Mirror-Imaging and Dunning-Kruger effects.
*   **System Identification**: Provides Python code for N4SID, ARX, and SINDy algorithms.
*   **Boundary Probing**: Generates signal patterns (Chirp, PRBS, Step) to test system limits.

## License

This project is licensed under the **GNU General Public License v3.0**. See the [LICENSE](LICENSE) file for details.

---