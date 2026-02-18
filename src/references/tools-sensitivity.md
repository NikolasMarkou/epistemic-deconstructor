# Tools & Sensitivity Analysis Reference

## Table of Contents

- [Analysis Tools by Domain](#analysis-tools-by-domain)
- [Frida Quick Reference](#frida-quick-reference)
- [angr Quick Reference](#angr-quick-reference)
- [Sensitivity Analysis](#sensitivity-analysis)
- [Anti-Analysis Detection](#anti-analysis-detection)
- [Chaos Engineering Principles](#chaos-engineering-principles)
- [Protocol State Machine Extraction](#protocol-state-machine-extraction)

---

## Analysis Tools by Domain

### Binary Analysis Platforms

| Tool | Type | Cost | Strength | Best For |
|------|------|------|----------|----------|
| **Ghidra** | Disassembler/Decompiler | Free | Comprehensive, P-code IR, extensible | General RE |
| **angr** | Analysis framework | Free | Symbolic execution, Python API | CTF, research |

### Dynamic Instrumentation

| Framework | Overhead | Best For |
|-----------|----------|----------|
| **Frida** | Low | Mobile, JavaScript scripting |
| **Unicorn** | N/A | CPU emulation, shellcode |

### System Identification Software

| Tool | Platform | Strength |
|------|----------|----------|
| **SysIdentPy** | Python | NARMAX with FROLS, PyTorch integration |
| **SIPPY** | Python | Linear MIMO (ARX, ARMAX, subspace) |
| **PyDMD** | Python | Dynamic Mode Decomposition variants |
| **python-control** | Python | Control analysis |

### Network Protocol Analysis

| Tool | Purpose |
|------|---------|
| **Wireshark** | Packet capture/analysis |
| **Netzob** | Protocol RE framework, L* grammar inference |
| **Scapy** | Packet crafting/manipulation |

## Frida Quick Reference

```javascript
// Attach to function by address
Interceptor.attach(ptr("0x401234"), {
    onEnter: function(args) {
        console.log("arg0: " + args[0]);
        console.log("arg1: " + args[1].readUtf8String());
    },
    onLeave: function(retval) {
        console.log("returned: " + retval);
    }
});

// Hook by symbol name
Interceptor.attach(Module.findExportByName(null, "open"), {
    onEnter: function(args) {
        console.log("open(" + args[0].readUtf8String() + ")");
    }
});

// Replace function implementation
Interceptor.replace(ptr("0x401234"), new NativeCallback(function(x) {
    return x * 2;  // Modified behavior
}, 'int', ['int']));

// Memory scanning
Memory.scan(ptr("0x400000"), 0x10000, "48 8b ?? ?? ?? ?? ??", {
    onMatch: function(address, size) {
        console.log("Found at: " + address);
    }
});
```

## angr Quick Reference

```python
import angr
import claripy

# Load binary
project = angr.Project("./target", auto_load_libs=False)

# Create symbolic input
stdin_len = 20
stdin = claripy.BVS("stdin", stdin_len * 8)

# Initial state with symbolic stdin
state = project.factory.entry_state(stdin=angr.SimFile("/dev/stdin", content=stdin))

# Explore to find path
simgr = project.factory.simgr(state)
simgr.explore(find=0x401234, avoid=0x401300)

if simgr.found:
    solution = simgr.found[0].solver.eval(stdin, cast_to=bytes)
    print(f"Input to reach target: {solution}")
```

## Sensitivity Analysis

### Sobol' Indices (Global)

Decomposes output variance across input factors:

**First-order index** (main effect):
```
Si = Var_Xi[E_X~i(Y|Xi)] / Var(Y)
```

**Total-order index** (including interactions):
```
STi = 1 - Var_X~i[E_Xi(Y|X~i)] / Var(Y)
```

**Interpretation**:
- Si ≈ STi → No interactions with other factors
- STi >> Si → Strong interaction effects
- STi ≈ 0 → Parameter has negligible influence, can be fixed

**Computational cost**: N(2k+2) model evaluations for k parameters

```python
from SALib.sample import saltelli
from SALib.analyze import sobol

problem = {
    'num_vars': 3,
    'names': ['x1', 'x2', 'x3'],
    'bounds': [[0, 1], [0, 1], [0, 1]]
}

# Generate samples
param_values = saltelli.sample(problem, 1024)

# Evaluate model
Y = np.array([model(x) for x in param_values])

# Compute indices
Si = sobol.analyze(problem, Y)
print("First-order:", Si['S1'])
print("Total-order:", Si['ST'])
```

### Morris Screening (Efficient Preliminary)

Efficient preliminary analysis at cost r(k+1) where r ≈ 10-100 trajectories.

**Elementary Effect**:
```
EEi = [Y(x1,...,xi+Δ,...,xk) - Y(x)] / Δ
```

**Statistics**:
- μ* = mean(|EEi|) → Importance (proxy for Si)
- σ = std(EEi) → Nonlinearity/interaction indicator

```python
from SALib.sample import morris as morris_sample
from SALib.analyze import morris

# Sample
param_values = morris_sample.sample(problem, N=100)

# Evaluate
Y = np.array([model(x) for x in param_values])

# Analyze
Si = morris.analyze(problem, param_values, Y)
print("mu_star:", Si['mu_star'])  # Importance
print("sigma:", Si['sigma'])       # Nonlinearity
```

**Decision rule**: Use Morris first (cheap), then Sobol' on important factors (expensive but precise).

### Local Sensitivity (Jacobian)

```python
def local_sensitivity(model, x0, delta=1e-6):
    """
    Compute Jacobian at x0 via finite differences.
    """
    n = len(x0)
    y0 = model(x0)
    m = len(y0) if hasattr(y0, '__len__') else 1
    
    J = np.zeros((m, n))
    for i in range(n):
        x_plus = x0.copy()
        x_plus[i] += delta
        J[:, i] = (model(x_plus) - y0) / delta
    
    return J
```

## Anti-Analysis Detection

### Common Techniques

| Technique | Detection Method | Bypass |
|-----------|------------------|--------|
| PEB.BeingDebugged | Check PEB flags | Patch PEB in memory |
| NtGlobalFlag | Check PEB+0x68 | Clear flag |
| Timing (RDTSC) | Measure instruction timing | Hardware breakpoints, emulation |
| IsDebuggerPresent | API call | Hook and return false |
| CheckRemoteDebuggerPresent | API call | Hook and return false |
| Hardware breakpoint detection | GetThreadContext | Don't use HW breakpoints |
| TLS callbacks | Execute before main | Analyze TLS directory |
| Self-modifying code | Runtime decryption | Dump after decryption |

### Detection Code Patterns
```c
// PEB check (x64)
mov rax, gs:[60h]      // PEB
movzx eax, byte ptr [rax+2]  // BeingDebugged
test eax, eax
jnz debugger_detected

// Timing check
rdtsc
mov esi, eax
// ... code being protected ...
rdtsc
sub eax, esi
cmp eax, 1000h
ja debugger_detected
```

### Bypass Script (Frida)
```javascript
// Bypass IsDebuggerPresent
Interceptor.attach(Module.findExportByName("kernel32.dll", "IsDebuggerPresent"), {
    onLeave: function(retval) {
        retval.replace(0);
    }
});

// Bypass timing checks
Interceptor.attach(Module.findExportByName(null, "QueryPerformanceCounter"), {
    onLeave: function(retval) {
        // Return consistent small delta
    }
});
```

## Chaos Engineering Principles

### Steady State Hypothesis
1. Define measurable baseline (latency, error rate, throughput)
2. Hypothesize it continues under perturbation
3. Introduce controlled failure
4. Measure divergence from baseline

### Perturbation Types
| Type | Example | Tool |
|------|---------|------|
| Instance failure | Kill random VM | Chaos Monkey |
| Network partition | Block traffic between services | tc, iptables |
| Latency injection | Add delay to responses | tc netem |
| Resource exhaustion | Fill disk, exhaust memory | stress-ng |
| Dependency failure | Kill database | Custom |

### Implementation
```python
def chaos_experiment(system, perturbation, duration_sec=300):
    """
    Run chaos experiment with automatic rollback.
    """
    # 1. Measure baseline
    baseline = measure_steady_state(system, duration=60)
    
    # 2. Apply perturbation
    try:
        perturbation.apply()
        
        # 3. Measure under stress
        stressed = measure_steady_state(system, duration=duration_sec)
        
        # 4. Compare to baseline
        deviation = compare_metrics(baseline, stressed)
        
    finally:
        # Always rollback
        perturbation.rollback()
    
    # 5. Assess
    return {
        'baseline': baseline,
        'stressed': stressed,
        'deviation': deviation,
        'hypothesis_held': deviation['error_rate'] < baseline['error_rate'] * 1.1
    }
```

## Protocol State Machine Extraction

### L* Algorithm Implementation
```python
class LStarLearner:
    def __init__(self, alphabet, oracle):
        self.alphabet = alphabet
        self.oracle = oracle
        self.S = {''}  # Prefixes
        self.E = {''}  # Suffixes
        self.T = {}    # Observation table
    
    def learn(self):
        while True:
            self._fill_table()
            
            if not self._is_closed():
                self._close()
                continue
            
            if not self._is_consistent():
                self._make_consistent()
                continue
            
            hypothesis = self._build_hypothesis()
            counterexample = self.oracle.equivalence_query(hypothesis)
            
            if counterexample is None:
                return hypothesis
            
            # Add all prefixes of counterexample
            for i in range(len(counterexample)):
                self.S.add(counterexample[:i+1])
    
    def _fill_table(self):
        for s in self.S | {s+a for s in self.S for a in self.alphabet}:
            for e in self.E:
                if (s, e) not in self.T:
                    self.T[(s, e)] = self.oracle.membership_query(s + e)
```

### Passive Inference from Traces
```python
def infer_fsm_from_traces(traces, merge_threshold=0.8):
    """
    Build state machine from observed message sequences.
    Uses prefix tree acceptor with state merging.
    """
    # Build prefix tree
    pta = PrefixTreeAcceptor()
    for trace in traces:
        pta.add_trace(trace)
    
    # State merging (EDSM algorithm)
    red = {pta.root}
    blue = set(pta.root.children.values())
    
    while blue:
        b = blue.pop()
        merged = False
        
        for r in red:
            if states_compatible(r, b, threshold=merge_threshold):
                merge_states(r, b)
                merged = True
                break
        
        if not merged:
            red.add(b)
        
        blue = {c for r in red for c in r.children.values() if c not in red}
    
    return build_fsm_from_pta(pta)
```
