# Spectral Analysis Reference

Frequency-domain analysis guide for `fourier_analyst.py`. Complements the time-domain `ts_reviewer.py` — use both for complete signal characterization.

## Table of Contents

- [When to Use](#when-to-use)
- [Phase Mapping](#phase-mapping)
- [CLI Quick Reference](#cli-quick-reference)
- [Phases Overview](#phases-overview)
- [Verdict Interpretation](#verdict-interpretation)
- [Utility Functions](#utility-functions)
- [Domain-Specific Guidance](#domain-specific-guidance)
- [Cross-References](#cross-references)

---

## When to Use

| Protocol Phase | When | What fourier_analyst Does |
|---|---|---|
| Phase 1: Boundary Mapping | Signal has periodic/oscillatory components | Spectral profile, dominant frequencies, noise classification |
| Phase 3: Parametric ID | Need frequency-domain model (transfer function) | System identification via coherence + FRF |
| Phase 4: Model Synthesis | Validate simulated output spectrally | Compare simulated vs observed spectra |
| Phase 5: Validation | Verify spectral predictions, detect anomalies | Spectral anomaly detection, baseline comparison |
| RAPID tier | Quick signal characterization | `quick` mode (phases 1-5, no model needed) |

**Use fourier_analyst when:**
- System produces oscillatory/periodic signals (vibration, electrical, acoustic)
- Frequency content matters more than time-domain shape
- Transfer function estimation needed (I/O pair available)
- Baseline spectral comparison required (anomaly detection)
- Machinery health monitoring (bearing faults, shaft imbalance)

**Use ts_reviewer instead when:**
- Forecasting / temporal prediction is the goal
- Stationarity testing needed
- Time-series cross-validation (walk-forward)
- Trend/seasonality decomposition

**Use both when:**
- Complete signal characterization (time + frequency)
- Phase 1 boundary mapping of unknown signals
- Phase 5 validation of dynamic system models

---

## Phase Mapping

| fourier_analyst Phase | Protocol Phase | Purpose |
|---|---|---|
| 1. Spectral Profile | Phase 1 | FFT, PSD, dominant frequencies, spectral shape |
| 2. Harmonic Analysis | Phase 1, 3 | Fundamental, harmonics, THD, sidebands |
| 3. Windowing Quality | Phase 1 | Leakage detection, window recommendations |
| 4. Noise Floor | Phase 1 | SNR, dynamic range, noise color |
| 5. Bandwidth Analysis | Phase 1 | Spectral rolloff, centroid, energy distribution |
| 6. System Identification | Phase 3 | Transfer function, coherence, FRF (needs I/O pair) |
| 7. Spectral Anomaly | Phase 5 | Compare against baseline spectrum |
| 8. Time-Frequency | Phase 1, 5 | STFT, stationarity, transient detection |
| 9. System Health | Phase 5 | Vibration diagnostics, bearing fault detection |

**Quick mode** (phases 1-5): Use for rapid spectral characterization without input/output pair or reference signal.

---

## CLI Quick Reference

```bash
# Full 9-phase analysis from CSV
python3 fourier_analyst.py analyze data.csv --column signal --fs 1000

# With shaft speed and JSON output
python3 fourier_analyst.py analyze vibration.csv --column accel_z --fs 10000 \
    --shaft-rpm 3600 --output analysis.json

# Quick review (phases 1-5 only)
python3 fourier_analyst.py quick data.csv --column voltage --fs 44100

# Compare multiple signals
python3 fourier_analyst.py compare sensors.csv --columns ch1,ch2,ch3 --fs 1000

# Built-in demo
python3 fourier_analyst.py demo
```

### CLI Flags

| Flag | Command | Description |
|---|---|---|
| `--column` | analyze, quick | Column name in CSV to analyze |
| `--fs` | all | Sample rate in Hz (default: 1.0) |
| `--fundamental` | analyze | Known fundamental frequency for harmonic analysis |
| `--shaft-rpm` | analyze | Shaft speed for vibration diagnostics |
| `--output` | analyze, quick, compare | Save JSON report to file |
| `--verbose` | analyze | Show all findings (default: true) |
| `--columns` | compare | Comma-separated column names |

---

## Phases Overview

### Phase 1: Spectral Profile
- **FFT configuration**: N, Fs, Nyquist, frequency resolution
- **Dominant frequencies**: Top peaks by amplitude (scipy `find_peaks` or local-max fallback)
- **DC offset check**: Flags if DC dominates signal (ratio > 10x RMS)
- **Spectral flatness**: Wiener entropy — 0 = tonal, 1 = white noise
- **Spectral centroid**: Center of mass of spectrum

### Phase 2: Harmonic Analysis
- **Fundamental detection**: Strongest non-DC peak (or user-specified)
- **Harmonic search**: 2f, 3f, ..., Nf with local prominence validation
- **THD**: Total Harmonic Distortion (% of fundamental)
  - < 1%: excellent | 1-10%: low | 10-50%: moderate (WARN) | > 50%: high (WARN)
- **Sideband detection**: Peaks near fundamental indicating modulation

### Phase 3: Windowing Quality
- **Window comparison**: Boxcar, Hann, Hamming, Blackman-Harris, Flat-top
- **Leakage assessment**: Dynamic range improvement from windowing
- **Nyquist proximity**: Flags dominant frequencies within 10% of Nyquist

### Phase 4: Noise Floor
- **Noise floor**: Median PSD (robust to peaks)
- **SNR estimate**: Signal power vs noise power in dB
  - < 3 dB: buried in noise | 3-10 dB: low | 10-20 dB: acceptable | > 20 dB: good
- **Noise color**: PSD slope in log-log space
  - ~0: white | -0.5 to -1.5: pink (1/f) | < -1.5: brown (1/f^2)

### Phase 5: Bandwidth Analysis
- **Spectral rolloff**: Frequencies containing 85%, 95%, 99% of energy
- **Oversampling detection**: Flags if 99% energy is < 20% of Nyquist
- **Band energy distribution**: Energy % across frequency bands
- **Spectral spread**: Standard deviation around centroid

### Phase 6: System Identification (requires I/O pair)
- **Coherence**: Input-output correlation by frequency
  - < 0.3: unreliable (FAIL) | 0.3-0.6: moderate (WARN) | > 0.6: good (PASS)
- **Transfer function H1**: Gain (dB) and phase (deg) vs frequency
- **Resonance detection**: Sharp peaks in gain (prominence > 6 dB)

### Phase 7: Spectral Anomaly Detection (requires reference signal)
- **Log-spectral distance**: RMS difference in dB
  - < 10 dB: similar | 10-20 dB: noticeable (WARN) | > 20 dB: major (FAIL)
- **Band energy shifts**: Per-band dB change from reference
- **New peak detection**: Frequencies present now but not in reference

### Phase 8: Time-Frequency Analysis (requires scipy)
- **STFT**: Short-Time Fourier Transform spectrogram
- **Spectral stationarity**: Power CV across time windows
  - CV < 0.2: stationary | 0.2-0.5: mildly non-stationary | > 0.5: non-stationary (WARN)
- **Centroid evolution**: Tracks frequency content shift over time
- **Transient detection**: Sudden power spikes (> 3 sigma)

### Phase 9: System Health Diagnostics
- **Crest factor**: Peak/RMS — > 6 indicates impulsive events (bearing defect, gear fault)
- **Kurtosis**: > 7 indicates strong impulsive events
- **Shaft harmonics**: 1x-5x shaft frequency amplitudes (requires `--shaft-rpm`)
- **Bearing fault frequencies**: BPFO, BPFI, BSF, FTF monitoring
- **Spectral fingerprint**: Top 8 dominant frequencies as signature

---

## Verdict Interpretation

| Verdict | Meaning |
|---|---|
| PASS | Check passed, no issues |
| WARN | Potential issue, investigate further |
| FAIL | Significant problem detected |
| REJECT | Signal unsuitable for analysis (e.g., too short) |
| SKIP | Phase skipped (missing dependency or insufficient data) |

| Severity | Meaning |
|---|---|
| INFO | Informational metric |
| LOW | Minor observation |
| MEDIUM | Worth investigating |
| HIGH | Likely problem |
| CRITICAL | Analysis-blocking issue |

---

## Utility Functions

For programmatic use (Python API, not CLI):

| Function | Protocol Phase | Usage |
|---|---|---|
| `quick_spectrum(signal, fs, name)` | Phase 1 | One-liner full analysis |
| `compare_spectra(signals_dict, fs)` | Phase 1, 5 | Multi-signal PSD comparison |
| `transfer_function(inp, out, fs)` | Phase 3 | H1 estimate + coherence |
| `spectral_distance(sig_a, sig_b, fs)` | Phase 5 | Log-spectral distance (dB) |
| `band_energy_profile(signal, fs, n_bands)` | Phase 1, 5 | Energy distribution by band |

---

## Domain-Specific Guidance

### Vibration / Rotating Machinery
- Use `--shaft-rpm` to enable shaft harmonic analysis
- Supply bearing fault frequencies (BPFO, BPFI, BSF, FTF) programmatically
- Phase 9 provides crest factor, kurtosis, and fault frequency monitoring
- Sideband detection (Phase 2) indicates modulation from bearing/gear faults

### Electrical / Power Systems
- Set `--fundamental` to line frequency (50 or 60 Hz) for THD analysis
- THD > 5% indicates power quality issues (IEEE 519 limit)
- Spectral flatness helps distinguish clean power from noisy sources

### Acoustic / Audio
- Use high `--fs` (44100+ Hz) for audio signals
- Spectral centroid and rolloff characterize tonal quality
- Time-frequency analysis (Phase 8) reveals temporal evolution

### Digital Systems / Communications
- Spectral flatness near 1.0 suggests encrypted/random data
- Band energy distribution reveals channel allocation
- Anomaly detection compares against known-good baseline

---

## Cross-References

- Time-domain signal diagnostics: `references/timeseries-review.md`
- Frequency domain system ID methods: `references/system-identification.md`
- I/O characterization probes (including frequency sweeps): `references/boundary-probing.md`
- Simulation output validation: `references/simulation-guide.md`
- Tool selection by phase: `references/tool-catalog.md`
