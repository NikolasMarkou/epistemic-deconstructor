# Tool Catalog Reference

Tools and techniques for system deconstruction, organized by domain and analysis phase.

## Table of Contents

- [Tool Integration Summary](#tool-integration-summary)
- [Binary Reverse Engineering](#binary-reverse-engineering)
- [Dynamic Analysis](#dynamic-analysis)
- [Symbolic Execution](#symbolic-execution)
- [System Identification](#system-identification)
- [Protocol Analysis](#protocol-analysis)
- [Fuzzing](#fuzzing)
- [Sensitivity Analysis](#sensitivity-analysis)
- [Phase-Based Tool Recommendations](#phase-based-tool-recommendations)
- [Web Search Triggers](#web-search-triggers)

---

## Tool Integration Summary

| Domain | Tools | Use |
|--------|-------|-----|
| Binary RE | Ghidra | Disassembly, decompilation |
| Dynamic | Frida, Unicorn | Runtime instrumentation, emulation |
| Symbolic | angr | Path exploration, constraint solving |
| System ID | SysIdentPy, SIPPY | Parameter estimation |
| Protocol | Netzob, Wireshark, Scapy | State machine extraction |
| Fuzzing | AFL++, libFuzzer | Edge case discovery |
| Sensitivity | SALib | Sobol', Morris screening |
| Time Series | ts_reviewer.py | Signal diagnostics, forecasting validation, conformal PI |
| Simulation | simulator.py | Forward simulation (SD, MC, ABM, DES), validation bridge |
| Utility | strace/procmon, pefile | System monitoring, PE analysis |

---

## Binary Reverse Engineering

### Ghidra (Free, NSA)

**Best for**: Initial analysis, decompilation, collaboration

```bash
# Launch with project
ghidraRun /path/to/project
```

Key features:
- Decompilation to C-like pseudocode
- Cross-references and call graphs
- Scripting via Python (Jython) or Java
- Collaborative analysis support

---

## Dynamic Analysis

### Frida (Free)

**Best for**: Runtime instrumentation, hooking, tracing

```javascript
// Example: Hook function and log arguments
Interceptor.attach(Module.findExportByName(null, "open"), {
    onEnter: function(args) {
        console.log("open(" + Memory.readUtf8String(args[0]) + ")");
    }
});
```

Use cases:
- API hooking
- Argument/return value modification
- SSL pinning bypass
- Memory inspection

### Unicorn (Free)

**Best for**: CPU emulation, shellcode analysis

```python
from unicorn import *
from unicorn.x86_const import *

mu = Uc(UC_ARCH_X86, UC_MODE_64)
mu.mem_map(0x1000, 0x1000)
mu.mem_write(0x1000, shellcode)
mu.emu_start(0x1000, 0x1000 + len(shellcode))
```

---

## Symbolic Execution

### angr (Free, Python)

**Best for**: Automated vulnerability discovery, constraint solving

```python
import angr

proj = angr.Project("./binary")
state = proj.factory.entry_state()
simgr = proj.factory.simulation_manager(state)

# Find path to target address
simgr.explore(find=0x401234, avoid=0x401000)

if simgr.found:
    solution = simgr.found[0]
    print(solution.posix.dumps(0))  # stdin that reaches target
```

---

## System Identification

### SysIdentPy (Python, Free)

**Best for**: NARMAX models, Python integration

```python
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial

basis = Polynomial(degree=2)
model = FROLS(basis_function=basis)
model.fit(X_train, y_train)
```

### SIPPY (Python, Free)

**Best for**: Subspace methods, state-space models

```python
from sippy import system_identification

# N4SID estimation
sys_id = system_identification(y, u, 'N4SID', SS_fixed_order=4)
```

---

## Protocol Analysis

### Netzob (Python, Free)

**Best for**: Automated protocol reverse engineering

```python
from netzob.all import *

# Cluster messages
messages = [RawMessage(data) for data in captures]
symbol = Symbol(messages=messages)
Format.clusterByKeyField(symbol, Field())
```

### Wireshark

**Best for**: Network capture, protocol dissection

Key features:
- 3000+ protocol dissectors
- Lua scripting for custom protocols
- Flow analysis and statistics

### Scapy (Python)

**Best for**: Packet crafting, custom protocol implementation

```python
from scapy.all import *

# Craft and send custom packet
pkt = IP(dst="target")/TCP(dport=80)/Raw("payload")
send(pkt)
```

---

## Fuzzing

### AFL++ (Free)

**Best for**: Coverage-guided fuzzing

```bash
# Compile with instrumentation
afl-clang-fast -o target target.c

# Run fuzzer
afl-fuzz -i inputs/ -o findings/ -- ./target @@
```

### libFuzzer (LLVM)

**Best for**: In-process fuzzing, sanitizer integration

```cpp
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    FunctionToTest(data, size);
    return 0;
}
```

---

## Sensitivity Analysis

### SALib (Python, Free)

**Best for**: Global sensitivity analysis

```python
from SALib.sample import saltelli
from SALib.analyze import sobol

# Define problem
problem = {
    'num_vars': 3,
    'names': ['x1', 'x2', 'x3'],
    'bounds': [[0, 1], [0, 1], [0, 1]]
}

# Generate samples
param_values = saltelli.sample(problem, 1024)

# Evaluate model (user-defined)
Y = np.array([model(x) for x in param_values])

# Analyze
Si = sobol.analyze(problem, Y)
print(Si['S1'])  # First-order indices
print(Si['ST'])  # Total-order indices
```

---

## Phase-Based Tool Recommendations

### Phase 0: Setup & Frame

- Documentation review (manual)
- Reconnaissance tools (nmap, recon-ng)
- OSINT tools (Maltego, theHarvester)

### Phase 1: Boundary Mapping

| Task | Tool |
|------|------|
| Network I/O discovery | Wireshark, tcpdump |
| File I/O discovery | strace, procmon |
| API enumeration | Frida, API Monitor |
| Probe signal generation | Custom scripts, SALib |

### Phase 2: Causal Analysis

| Task | Tool |
|------|------|
| Static analysis | Ghidra |
| Dynamic tracing | Frida |
| Taint analysis | Triton |
| Symbolic execution | angr |

### Phase 3: Parametric Identification

| Task | Tool |
|------|------|
| Linear system ID | SysIdentPy |
| Nonlinear system ID | SysIdentPy (NARMAX) |
| State-space estimation | SIPPY (N4SID) |
| Frequency analysis | scipy.signal |
| Time-series diagnostics | ts_reviewer.py (stationarity, PE, baselines, FVA) |
| Model comparison | ts_reviewer.py `compare_models()` — ranks by MASE |
| Temporal cross-validation | ts_reviewer.py `walk_forward_split()` — expanding/rolling CV |
| Model selection guidance | `references/forecasting-science.md` — Naive→ETS→ARIMA→CatBoost hierarchy |
| Financial domain validation | `references/financial-validation.md` — martingale baseline, returns-only metrics |

### Phase 4: Model Synthesis

| Task | Tool |
|------|------|
| Graph visualization | Graphviz, NetworkX |
| Forward simulation | simulator.py — paradigm from archetype (see `simulation-guide.md`): SD for controllers, MC for stochastic, ABM for networks, DES for queues |
| Emergence testing | simulator.py MC/SD — compare output to Phase 1 observations via ts_reviewer.py quick |
| Sensitivity analysis | simulator.py sensitivity — Sobol/Morris/OAT on model parameters |

### Phase 5: Validation & Adversarial

| Task | Tool |
|------|------|
| Residual diagnostics | ts_reviewer.py phases 7-10 (whiteness, homoscedasticity, normality) |
| Baseline/FVA validation | ts_reviewer.py Phase 6 — FVA > 0% required |
| Uncertainty quantification | ts_reviewer.py `conformal_intervals()` or `cqr_intervals()` |
| Simulation validation | simulator.py `bridge` — feeds simulation output to Phase 5 checks |
| Edge case discovery | AFL++, libFuzzer |
| Attack surface mapping | Burp Suite, OWASP ZAP |
| Adversarial testing | Custom adversarial scripts |

---

## Web Search Triggers

Use web search when encountering:

1. **Unknown component**: "What is [component name]?"
2. **Unexpected behavior**: "[behavior] [system type]"
3. **CVE lookup**: "CVE [component] [version]"
4. **Library documentation**: "[library] API reference"
5. **Error messages**: Exact error string in quotes
6. **Protocol specifications**: "[protocol] RFC" or "[protocol] specification"
7. **Algorithm details**: "[algorithm name] implementation"

**Search strategy**:
- Use exact phrases in quotes
- Include version numbers when relevant
- Prefer official documentation sources
- Cross-reference multiple sources for critical information

---

## Cross-References

- Sensitivity analysis algorithms: `references/tools-sensitivity.md`
- System identification methods: `references/system-identification.md`
- Causal analysis techniques: `references/causal-techniques.md`
- Adversarial bypass techniques: `references/adversarial-heuristics.md`
- Simulation paradigms and validation bridge: `references/simulation-guide.md`
- Time-series diagnostics and usage guide: `references/timeseries-review.md`
- Forecasting science (model selection, metrics, conformal): `references/forecasting-science.md`
- Financial forecasting validation: `references/financial-validation.md`
