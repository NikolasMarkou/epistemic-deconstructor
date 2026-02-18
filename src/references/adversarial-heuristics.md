# Adversarial Heuristics Reference

Techniques for identifying, assessing, and bypassing adversarial defenses in systems under analysis.

## Table of Contents

- [Adversarial Posture Classification](#adversarial-posture-classification)
- [Anti-Analysis Detection Techniques](#anti-analysis-detection-techniques)
- [Deception Indicators](#deception-indicators)
- [Bypass Strategies](#bypass-strategies)
- [Adversarial Pre-Check Protocol](#adversarial-pre-check-protocol)

---

## Adversarial Posture Classification

Classify the target's defensive posture before deep analysis:

| Posture Level | Indicators | Response |
|---------------|------------|----------|
| **L0: None** | No anti-analysis features | Standard protocol |
| **L1: Passive** | Obfuscation, encryption, packing | Deobfuscate, decrypt, unpack |
| **L2: Active Detection** | Debugger checks, VM detection, logging | Stealth mode, evasion techniques |
| **L3: Active Response** | Terminate on detection, data corruption, counter-attack | Isolated analysis environment |
| **L4: Adaptive** | Changes behavior when analyzed, patches vulnerabilities | Parallel approach, rapid iteration |

### L0: No Adversarial Features

Indicators:
- Clear code structure
- Readable strings
- Standard error handling
- No timing checks

Response: Proceed with standard analysis protocol.

### L1: Passive Defenses

Indicators:
- Packed executables (high entropy sections)
- Encrypted configuration/strings
- Obfuscated control flow
- String encryption/encoding

Response:
1. Identify packer/crypter (PEiD, Detect It Easy)
2. Unpack using appropriate tools
3. Decrypt strings dynamically
4. Deobfuscate control flow

### L2: Active Detection

Indicators:
- Debugger detection calls
- VM/sandbox detection
- Timing-based detection
- Process enumeration
- Network connectivity checks

Response:
1. Identify detection vectors
2. Apply appropriate bypasses
3. Use stealth analysis tools
4. Modify detection return values

### L3: Active Response

Indicators:
- Process termination on detection
- Data corruption/wiping
- Anti-forensics
- Counter-reconnaissance

Response:
1. Full isolation (air-gapped VM)
2. Snapshot before analysis
3. Parallel analysis instances
4. Network containment

### L4: Adaptive Defense

Indicators:
- Behavior changes between runs
- Server-side logic changes
- Rapid patching of discovered issues
- Honeytokens/canaries

Response:
1. Multiple parallel analysis streams
2. Rapid iteration before adaptation
3. Diverse analysis approaches
4. Minimize detection surface

---

## Anti-Analysis Detection Techniques

### Debugger Detection (Software)

| Technique | Description | Bypass |
|-----------|-------------|--------|
| `PEB.BeingDebugged` | Check PEB flag | Patch PEB in memory |
| `IsDebuggerPresent()` | WinAPI check | Hook API, return false |
| `NtQueryInformationProcess` | Kernel check | Hook NtDll, fake response |
| `OutputDebugString` | Side-effect timing | Ignore timing |
| `FindWindow` | Search for debugger windows | Hide window, spoof class |

### Timing-Based Detection

| Technique | Description | Bypass |
|-----------|-------------|--------|
| `RDTSC` | CPU timestamp counter | Hardware breakpoints, emulation |
| `QueryPerformanceCounter` | High-resolution timer | Hook and normalize |
| `GetTickCount` | System uptime | Hook and normalize |
| Instruction timing | Measure execution delta | Transparent debugging |

### VM/Sandbox Detection

| Technique | Description | Bypass |
|-----------|-------------|--------|
| Registry keys | VM-specific entries | Remove/rename keys |
| MAC address | VM vendor prefixes | Spoof MAC |
| Hardware strings | BIOS/DMI detection | Modify strings |
| CPUID | Hypervisor bit | Hide hypervisor |
| Resolution/clicks | Low activity detection | Simulate activity |

### Process/Environment Checks

| Technique | Description | Bypass |
|-----------|-------------|--------|
| Process enumeration | Find analysis tools | Rename/hide processes |
| Module enumeration | Detect hooks | Unhook or hide modules |
| Username/hostname | Sandbox indicators | Realistic values |
| File presence | Tool artifacts | Remove artifacts |

### TLS Callbacks

TLS (Thread Local Storage) callbacks execute before main():

Detection: Analyze TLS directory in PE header
Bypass: Set breakpoint on TLS callback, analyze separately

### Self-Modifying Code

Indicators:
- VirtualProtect/mprotect calls
- Execute from heap/stack
- Code writes to code sections

Analysis:
1. Set memory breakpoints on code writes
2. Dump after modification
3. Trace execution dynamically

---

## Deception Indicators

Signs that the system may be intentionally deceiving the analyst:

### Data Deception

| Indicator | Interpretation |
|-----------|----------------|
| **Too-clean data** | Possible honeypot or synthetic environment |
| **Perfect patterns** | Artificially generated, not real-world |
| **Convenient errors** | Errors that reveal "just enough" |
| **Breadcrumb trails** | Obvious path to "discovery" |

### Structural Deception

| Indicator | Interpretation |
|-----------|----------------|
| **Inconsistent complexity** | Hidden subsystem, decoy components |
| **Orphan code** | Dead code that appears functional |
| **Perfect documentation** | Synthetic environment or trap |
| **Too-easy access** | Honeypot or intentional exposure |

### Behavioral Deception

| Indicator | Interpretation |
|-----------|----------------|
| **Predictable randomness** | Seeded or fake randomization |
| **Consistent errors** | Scripted error responses |
| **Response without processing** | Cached/fake responses |
| **Artificial delays** | Simulating work that isn't done |

### Honeypot Detection

Questions to ask:
1. Is this system too easy to access?
2. Does the "sensitive" data look templated?
3. Are there monitoring artifacts?
4. Does behavior change after certain actions?

---

## Bypass Strategies

### General Principles

1. **Identify before bypass**: Know what you're evading
2. **Minimal footprint**: Change only what's necessary
3. **Validate bypass**: Confirm detection was evaded
4. **Document techniques**: For reproducibility

### Debugger Evasion

```python
# Frida script to bypass IsDebuggerPresent
Interceptor.attach(Module.findExportByName("kernel32.dll", "IsDebuggerPresent"), {
    onLeave: function(retval) {
        retval.replace(0);
    }
});
```

### PEB Patching

```python
# Patch BeingDebugged flag in PEB
import ctypes
import struct

# Get PEB address and patch BeingDebugged byte
```

### Timing Normalization

```python
# Hook timing functions to return normalized values
original_time = time.time
start = original_time()

def normalized_time():
    elapsed = original_time() - start
    return start + elapsed * 0.001  # Slow down apparent time
```

### VM Artifact Removal

Checklist:
- [ ] Remove VM tools/additions
- [ ] Change MAC address prefix
- [ ] Modify BIOS strings
- [ ] Rename VM-specific registry keys
- [ ] Spoof hardware identifiers

---

## Adversarial Pre-Check Protocol

Before probing any system, assess adversarial risk:

### Quick Assessment (2 minutes)

```
[ ] Is this a known protected system type?
[ ] High entropy sections present? (packed/encrypted)
[ ] Known anti-debug patterns in strings?
[ ] Unusual imports (timing, VM detection APIs)?
[ ] Server-side or cloud-connected?
```

### Risk Classification

| Risk Level | Indicators | Action |
|------------|------------|--------|
| **Low** | None of the above | Standard protocol |
| **Medium** | 1-2 indicators | Enhanced isolation, slower pace |
| **High** | 3+ indicators | Full isolation, escalate to COMPREHENSIVE tier |
| **Critical** | Active defense observed | Air-gap, parallel analysis, expert consultation |

### Isolation Levels

| Level | Description | When to Use |
|-------|-------------|-------------|
| **Standard** | Normal workstation | L0 adversarial posture |
| **VM Isolated** | Dedicated VM, snapshots | L1-L2 adversarial posture |
| **Air-Gapped** | No network, clean environment | L3 adversarial posture |
| **Hardware Isolated** | Dedicated hardware, no shared components | L4 or critical systems |

---

## Cross-References

- Tool catalog: `references/tool-catalog.md`
- Cognitive traps (avoiding misdirection): `references/cognitive-traps.md`
- Causal analysis: `references/causal-techniques.md`
