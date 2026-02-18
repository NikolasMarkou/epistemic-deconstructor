# Boundary Mapping & Probing Reference

Techniques for characterizing system I/O behavior.

## Table of Contents

- [Probe Signal Generation](#probe-signal-generation)
- [I/O Channel Discovery](#io-channel-discovery)
- [Edge Case Testing](#edge-case-testing)
- [Response Analysis](#response-analysis)
- [Stimulus-Response Database Schema](#stimulus-response-database-schema)

---

## Probe Signal Generation

### Step Input
```python
import numpy as np

def step_signal(duration, sample_rate, amplitude=1.0, delay=0.1):
    """Generate step input signal."""
    t = np.linspace(0, duration, int(duration * sample_rate))
    signal = np.where(t >= delay, amplitude, 0)
    return t, signal
```
**Reveals**: Time constants, settling time, overshoot, steady-state gain

### Impulse
```python
def impulse_signal(duration, sample_rate, amplitude=1.0, width=0.001):
    """Generate impulse (brief pulse)."""
    t = np.linspace(0, duration, int(duration * sample_rate))
    signal = np.where((t >= 0.1) & (t < 0.1 + width), amplitude, 0)
    return t, signal
```
**Reveals**: Impulse response → frequency response via FFT

### PRBS (Pseudo-Random Binary Sequence)
```python
def prbs_signal(n_bits, sample_rate, duration, amplitude=1.0):
    """Generate maximum-length PRBS."""
    from scipy.signal import max_len_seq
    seq, _ = max_len_seq(n_bits)
    seq = 2 * seq - 1  # Convert to ±1
    seq = amplitude * seq
    
    # Repeat to fill duration
    n_samples = int(duration * sample_rate)
    repeats = n_samples // len(seq) + 1
    signal = np.tile(seq, repeats)[:n_samples]
    t = np.linspace(0, duration, n_samples)
    return t, signal
```
**Reveals**: Broadband linear response, good for system identification

### Chirp (Frequency Sweep)
```python
from scipy.signal import chirp

def chirp_signal(duration, sample_rate, f_start, f_stop):
    """Generate frequency sweep."""
    t = np.linspace(0, duration, int(duration * sample_rate))
    signal = chirp(t, f_start, duration, f_stop, method='linear')
    return t, signal
```
**Reveals**: Frequency response, resonances, bandwidth

### Multisine
```python
def multisine_signal(duration, sample_rate, frequencies, amplitudes=None):
    """Generate sum of sinusoids at specified frequencies."""
    t = np.linspace(0, duration, int(duration * sample_rate))
    if amplitudes is None:
        amplitudes = np.ones(len(frequencies))
    
    signal = np.zeros_like(t)
    for f, a in zip(frequencies, amplitudes):
        phase = np.random.uniform(0, 2*np.pi)  # Random phase
        signal += a * np.sin(2 * np.pi * f * t + phase)
    return t, signal
```
**Reveals**: Precise frequency domain characterization

## I/O Channel Discovery

### Explicit Channels
| Type | Discovery Method |
|------|------------------|
| Network ports | Port scanning (nmap) |
| File I/O | strace, procmon |
| API endpoints | Documentation, fuzzing |
| User interface | Manual enumeration |
| Configuration | Config file discovery |

### Implicit Channels
| Type | Discovery Method |
|------|------------------|
| Environment variables | env dump, isolation |
| Time/clock | Time variation experiments |
| System load | Resource monitoring |
| Random seed | Repeated trials |

### Side Channels
| Type | Measurement |
|------|-------------|
| Timing | High-resolution timers |
| Power | Current probe, EM probe |
| Electromagnetic | Near-field probe, SDR |
| Acoustic | Microphone, spectrum analyzer |
| Cache | Prime+Probe, Flush+Reload |

## Edge Case Testing

### Boundary Values
```python
def boundary_test_cases(param_spec):
    """Generate boundary test cases."""
    cases = []
    
    if param_spec['type'] == 'int':
        mn, mx = param_spec['min'], param_spec['max']
        cases = [mn-1, mn, mn+1, (mn+mx)//2, mx-1, mx, mx+1]
    
    elif param_spec['type'] == 'string':
        cases = [
            "",                    # Empty
            "a",                   # Single char
            "a" * 256,             # Long
            "a" * 65536,           # Very long
            "\x00",                # Null byte
            "\n\r\t",              # Whitespace
            "'; DROP TABLE x;--", # Injection
        ]
    
    elif param_spec['type'] == 'float':
        cases = [
            float('-inf'), -1e308, -1.0, -1e-308,
            0.0, 1e-308, 1.0, 1e308, float('inf'),
            float('nan')
        ]
    
    return cases
```

### Stress Testing
```python
def stress_test(system, duration_sec=60, rate_multiplier=10):
    """Stress test at elevated load."""
    baseline_rate = measure_normal_rate(system)
    test_rate = baseline_rate * rate_multiplier
    
    results = {
        'errors': [],
        'latencies': [],
        'resource_usage': []
    }
    
    start = time.time()
    while time.time() - start < duration_sec:
        try:
            t0 = time.time()
            response = system.probe()
            results['latencies'].append(time.time() - t0)
        except Exception as e:
            results['errors'].append((time.time() - start, str(e)))
        
        time.sleep(1.0 / test_rate)
    
    return results
```

## Response Analysis

### Time Domain Metrics
```python
def analyze_step_response(t, y, setpoint=1.0):
    """Extract step response characteristics."""
    y_final = y[-1]
    y_steady = np.mean(y[-len(y)//10:])  # Last 10%
    
    # Rise time (10% to 90%)
    y10 = 0.1 * y_steady
    y90 = 0.9 * y_steady
    t10 = t[np.argmax(y >= y10)]
    t90 = t[np.argmax(y >= y90)]
    rise_time = t90 - t10
    
    # Overshoot
    peak = np.max(y)
    overshoot = (peak - y_steady) / y_steady * 100
    
    # Settling time (±2%)
    settled = np.abs(y - y_steady) < 0.02 * y_steady
    settling_idx = np.where(settled)[0]
    if len(settling_idx) > 0:
        # Find last non-settled point
        settling_time = t[settling_idx[0]]
    else:
        settling_time = float('inf')
    
    return {
        'rise_time': rise_time,
        'overshoot_pct': overshoot,
        'settling_time': settling_time,
        'steady_state': y_steady,
        'steady_state_error': abs(y_steady - setpoint)
    }
```

### Frequency Domain Analysis
```python
def frequency_response(input_signal, output_signal, sample_rate):
    """Compute frequency response from I/O signals."""
    # FFT
    U = np.fft.fft(input_signal)
    Y = np.fft.fft(output_signal)
    freqs = np.fft.fftfreq(len(input_signal), 1/sample_rate)
    
    # Transfer function estimate
    H = Y / (U + 1e-10)  # Avoid division by zero
    
    # Keep positive frequencies
    pos_idx = freqs > 0
    freqs = freqs[pos_idx]
    H = H[pos_idx]
    
    magnitude_db = 20 * np.log10(np.abs(H) + 1e-10)
    phase_deg = np.angle(H) * 180 / np.pi
    
    return freqs, magnitude_db, phase_deg
```

## Stimulus-Response Database Schema

```python
@dataclass
class ProbeRecord:
    probe_id: str
    timestamp: str
    input_type: str          # step, impulse, prbs, chirp, etc.
    input_params: dict       # Signal parameters
    input_data: np.ndarray   # Raw input signal
    output_data: np.ndarray  # Raw output signal
    sample_rate: float
    latency_ms: float        # Response latency
    anomalies: List[str]     # Unexpected behaviors
    metrics: dict            # Extracted metrics
    notes: str
```

### Database Operations
```python
class ProbeDatabase:
    def __init__(self, filepath="probes.json"):
        self.filepath = filepath
        self.probes = []
        self.load()
    
    def add(self, probe: ProbeRecord):
        self.probes.append(probe)
        self.save()
    
    def query(self, input_type=None, min_latency=None, has_anomaly=False):
        results = self.probes
        if input_type:
            results = [p for p in results if p.input_type == input_type]
        if min_latency:
            results = [p for p in results if p.latency_ms >= min_latency]
        if has_anomaly:
            results = [p for p in results if p.anomalies]
        return results
    
    def coverage_report(self):
        """Report on I/O coverage."""
        types = set(p.input_type for p in self.probes)
        anomaly_count = sum(1 for p in self.probes if p.anomalies)
        return {
            'total_probes': len(self.probes),
            'input_types_covered': list(types),
            'probes_with_anomalies': anomaly_count
        }
```
