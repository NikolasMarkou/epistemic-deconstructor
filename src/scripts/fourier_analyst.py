#!/usr/bin/env python3
"""
fourier_analyst.py — Fourier-Based System Analysis Framework
==============================================================
Domain-specific spectral analysis tool for evaluating unknown time series
produced by physical, mechanical, or digital systems.

Synthesizes:
  - Classical Fourier / spectral analysis: FFT, PSD (Welch), periodogram
  - System identification: transfer function estimation, coherence, FRF
  - Harmonic analysis: dominant frequencies, harmonic ratios, sidebands
  - Spectral diagnostics: leakage detection, windowing quality, noise floor
  - Anomaly detection: spectral distance, band energy drift, new-peak detection
  - Time-frequency: STFT / spectrogram, spectral evolution tracking
  - System health: vibration signatures, spectral fingerprinting, bearing faults

Usage:
    from fourier_analyst import FourierAnalyst

    fa = FourierAnalyst(signal, fs=1000.0, name="pump_vibration")
    report = fa.full_analysis()
    report.print_report()

    # Individual phases
    fa.phase_spectral_profile()          # FFT + PSD + dominant frequencies
    fa.phase_harmonic_analysis()         # harmonic structure, THD, sidebands
    fa.phase_windowing_quality()         # leakage, window recommendations
    fa.phase_noise_floor()               # noise floor estimation, SNR
    fa.phase_bandwidth_analysis()        # spectral rolloff, bandwidth
    fa.phase_system_identification(inp, out)  # transfer function, coherence
    fa.phase_spectral_anomaly(reference) # compare against reference spectrum
    fa.phase_time_frequency()            # STFT, spectral evolution
    fa.phase_system_health()             # vibration diagnostics, fault freqs

Dependencies (graceful degradation):
    Required : numpy  (FFT is inherently numerical)
    Optional : scipy  (Welch PSD, windows, signal processing)
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Required: numpy (FFT without numpy is impractical)
# ---------------------------------------------------------------------------
try:
    import numpy as np
    from numpy.fft import fft, ifft, fftfreq, rfft, rfftfreq

    _HAS_NUMPY = True
except ImportError:
    raise ImportError(
        "fourier_analyst requires numpy. Install with: pip install numpy"
    )

# ---------------------------------------------------------------------------
# Optional: scipy (Welch PSD, windows, advanced signal processing)
# ---------------------------------------------------------------------------
try:
    from scipy.signal import welch, coherence as sp_coherence, csd
    from scipy.signal import stft as sp_stft
    from scipy.signal import get_window
    from scipy.signal import find_peaks
    from scipy import stats as sp_stats

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ===========================================================================
# 1. ENUMS & DATA STRUCTURES (same pattern as ts_reviewer)
# ===========================================================================


class Verdict(Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    REJECT = "REJECT"
    SKIP = "SKIP"


class Severity(Enum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class Finding:
    phase: str
    check: str
    verdict: Verdict
    severity: Severity
    detail: str
    value: Any = None

    def __str__(self) -> str:
        icons = {
            "PASS": "[OK]", "WARN": "[!!]", "FAIL": "[XX]",
            "REJECT": "[RJ]", "SKIP": "[--]",
        }
        return f"  {icons[self.verdict.value]} [{self.severity.value:8s}] {self.check}: {self.detail}"


@dataclass
class PhaseResult:
    name: str
    findings: List[Finding] = field(default_factory=list)

    @property
    def verdict(self) -> Verdict:
        for v in (Verdict.REJECT, Verdict.FAIL, Verdict.WARN):
            if any(f.verdict == v for f in self.findings):
                return v
        if all(f.verdict == Verdict.SKIP for f in self.findings):
            return Verdict.SKIP
        return Verdict.PASS

    def add(self, check: str, verdict: Verdict, severity: Severity,
            detail: str, value: Any = None) -> Finding:
        f = Finding(self.name, check, verdict, severity, detail, value)
        self.findings.append(f)
        return f


@dataclass
class AnalysisReport:
    signal_name: str
    n_samples: int
    sample_rate: float
    phases: List[PhaseResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    spectra: Dict[str, Any] = field(default_factory=dict)

    @property
    def overall_verdict(self) -> Verdict:
        for v in (Verdict.REJECT, Verdict.FAIL, Verdict.WARN):
            if any(p.verdict == v for p in self.phases):
                return v
        return Verdict.PASS

    @property
    def all_findings(self) -> List[Finding]:
        return [f for p in self.phases for f in p.findings]

    @property
    def red_flags(self) -> List[Finding]:
        return [f for f in self.all_findings if f.verdict in (Verdict.FAIL, Verdict.REJECT)]

    @property
    def warnings(self) -> List[Finding]:
        return [f for f in self.all_findings if f.verdict == Verdict.WARN]

    def print_report(self, *, verbose: bool = True) -> None:
        w = 72
        nyq = self.sample_rate / 2
        dur = self.n_samples / self.sample_rate if self.sample_rate > 0 else 0
        print(f"\n{'=' * w}")
        print(f"  FOURIER ANALYSIS: {self.signal_name}")
        print(f"  Samples: {self.n_samples}  |  Fs: {self.sample_rate} Hz  |  "
              f"Nyquist: {nyq} Hz  |  Duration: {dur:.4f}s")
        print(f"  Overall: {self.overall_verdict.value}  |  "
              f"Red flags: {len(self.red_flags)}  |  Warnings: {len(self.warnings)}")
        print(f"{'=' * w}")

        for phase in self.phases:
            print(f"\n--- {phase.name} [{phase.verdict.value}] ---")
            items = phase.findings if verbose else [
                f for f in phase.findings if f.verdict != Verdict.PASS
            ]
            for f in items:
                print(str(f))

        flags = self.red_flags
        if flags:
            print(f"\n{'!' * w}")
            print(f"  RED FLAGS ({len(flags)}):")
            for f in flags:
                print(f"    [{f.phase}] {f.check}: {f.detail}")
            print(f"{'!' * w}")
        print()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_name": self.signal_name,
            "n_samples": self.n_samples,
            "sample_rate": self.sample_rate,
            "overall_verdict": self.overall_verdict.value,
            "red_flag_count": len(self.red_flags),
            "warning_count": len(self.warnings),
            "metadata": _safe(self.metadata),
            "phases": [
                {
                    "name": p.name,
                    "verdict": p.verdict.value,
                    "findings": [
                        {
                            "check": f.check,
                            "verdict": f.verdict.value,
                            "severity": f.severity.value,
                            "detail": f.detail,
                            "value": _safe(f.value),
                        }
                        for f in p.findings
                    ],
                }
                for p in self.phases
            ],
        }


def _safe(v: Any) -> Any:
    if v is None or isinstance(v, (int, float, str, bool)):
        return v
    if isinstance(v, (list, tuple)):
        return [_safe(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _safe(val) for k, val in v.items()}
    if isinstance(v, np.ndarray):
        return v.tolist()
    try:
        f = float(v)
        return f if math.isfinite(f) else str(v)
    except Exception:
        return str(v)


# ===========================================================================
# 2. SPECTRAL HELPERS
# ===========================================================================


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _apply_window(x: np.ndarray, window: str = "hann") -> Tuple[np.ndarray, float]:
    """Apply window and return (windowed_signal, coherent_gain)."""
    n = len(x)
    if _HAS_SCIPY:
        w = get_window(window, n)
    else:
        # fallback: Hann window
        w = 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / (n - 1)))
    cg = np.mean(w)  # coherent gain for amplitude correction
    return x * w, cg


def _compute_psd(x: np.ndarray, fs: float, nperseg: int = 0,
                 window: str = "hann") -> Tuple[np.ndarray, np.ndarray]:
    """Compute PSD via Welch (scipy) or periodogram fallback."""
    if nperseg == 0:
        nperseg = min(len(x), max(256, _next_pow2(len(x) // 8)))

    if _HAS_SCIPY:
        f, pxx = welch(x, fs=fs, window=window, nperseg=nperseg,
                       noverlap=nperseg // 2, scaling="density")
    else:
        # manual periodogram
        xw, cg = _apply_window(x, window)
        nfft = _next_pow2(len(xw))
        X = rfft(xw, n=nfft)
        f = rfftfreq(nfft, d=1.0 / fs)
        pxx = (np.abs(X) ** 2) / (fs * len(xw) * (cg ** 2))
        pxx[1:-1] *= 2  # single-sided
    return f, pxx


def _amplitude_spectrum(x: np.ndarray, fs: float,
                        window: str = "hann") -> Tuple[np.ndarray, np.ndarray]:
    """Single-sided amplitude spectrum (magnitude, corrected for window)."""
    n = len(x)
    xw, cg = _apply_window(x, window)
    nfft = _next_pow2(n)
    X = rfft(xw, n=nfft)
    freqs = rfftfreq(nfft, d=1.0 / fs)
    mag = np.abs(X) * 2.0 / (n * cg)
    mag[0] /= 2.0  # DC component
    return freqs, mag


def _find_dominant_freqs(freqs: np.ndarray, mag: np.ndarray,
                         n_peaks: int = 10,
                         min_prominence: float = 0.0) -> List[Tuple[float, float]]:
    """Return list of (frequency, amplitude) for dominant peaks."""
    if _HAS_SCIPY:
        prom = min_prominence if min_prominence > 0 else np.max(mag) * 0.02
        peaks, props = find_peaks(mag, prominence=prom, distance=3)
        # sort by magnitude descending
        order = np.argsort(mag[peaks])[::-1]
        peaks = peaks[order[:n_peaks]]
        return [(float(freqs[p]), float(mag[p])) for p in sorted(peaks)]
    else:
        # simple local-max finder
        results = []
        for i in range(1, len(mag) - 1):
            if mag[i] > mag[i - 1] and mag[i] > mag[i + 1]:
                results.append((float(freqs[i]), float(mag[i])))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:n_peaks]


def _spectral_centroid(freqs: np.ndarray, mag: np.ndarray) -> float:
    total = np.sum(mag)
    if total == 0:
        return 0.0
    return float(np.sum(freqs * mag) / total)


def _spectral_rolloff(freqs: np.ndarray, mag: np.ndarray,
                      threshold: float = 0.85) -> float:
    """Frequency below which `threshold` fraction of spectral energy lies."""
    cumulative = np.cumsum(mag ** 2)
    total = cumulative[-1]
    if total == 0:
        return 0.0
    idx = np.searchsorted(cumulative, threshold * total)
    idx = min(idx, len(freqs) - 1)
    return float(freqs[idx])


def _spectral_flatness(pxx: np.ndarray) -> float:
    """Wiener entropy: geometric mean / arithmetic mean of PSD. 1.0 = white noise."""
    pxx_pos = pxx[pxx > 0]
    if len(pxx_pos) == 0:
        return 0.0
    log_mean = np.mean(np.log(pxx_pos))
    arith_mean = np.mean(pxx_pos)
    if arith_mean == 0:
        return 0
    return float(np.exp(log_mean) / arith_mean)


def _band_energy(freqs: np.ndarray, pxx: np.ndarray,
                 f_low: float, f_high: float) -> float:
    """Total energy in frequency band [f_low, f_high]."""
    mask = (freqs >= f_low) & (freqs <= f_high)
    if not np.any(mask):
        return 0.0
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    return float(np.sum(pxx[mask]) * df)


# ===========================================================================
# 3. THE ANALYST
# ===========================================================================


class FourierAnalyst:
    """
    Fourier-based system analysis for unknown time series.

    Phases:
        1. Spectral Profile       — FFT, PSD, dominant frequencies, spectral shape
        2. Harmonic Analysis       — fundamental, harmonics, THD, sidebands
        3. Windowing Quality       — leakage assessment, window recommendations
        4. Noise Floor             — noise floor estimation, SNR, dynamic range
        5. Bandwidth Analysis      — spectral rolloff, centroid, flatness, bandwidth
        6. System Identification   — transfer function, coherence, FRF (needs I/O)
        7. Spectral Anomaly        — compare against reference spectrum
        8. Time-Frequency          — STFT, spectral evolution, stationarity
        9. System Health           — vibration diagnostics, fault frequencies

    Parameters
    ----------
    signal : array-like
        Time-domain signal (1D).
    fs : float
        Sample rate in Hz.
    name : str
        Signal label for reports.
    """

    def __init__(
        self,
        signal: Sequence,
        *,
        fs: float = 1.0,
        name: str = "unknown_signal",
    ):
        self._x = np.asarray(signal, dtype=np.float64).ravel()
        self.fs = fs
        self.name = name
        self.n = len(self._x)
        self._report = AnalysisReport(
            signal_name=name, n_samples=self.n, sample_rate=fs
        )

        # cached spectra
        self._freqs: Optional[np.ndarray] = None
        self._mag: Optional[np.ndarray] = None
        self._psd_f: Optional[np.ndarray] = None
        self._psd: Optional[np.ndarray] = None
        self._dominant: Optional[List[Tuple[float, float]]] = None

    # -- cached computations -----------------------------------------------

    def _ensure_spectrum(self):
        if self._freqs is None:
            self._freqs, self._mag = _amplitude_spectrum(self._x, self.fs)

    def _ensure_psd(self):
        if self._psd_f is None:
            self._psd_f, self._psd = _compute_psd(self._x, self.fs)

    def _ensure_peaks(self):
        if self._dominant is None:
            self._ensure_spectrum()
            self._dominant = _find_dominant_freqs(self._freqs, self._mag, n_peaks=20)

    # ======================================================================
    # PHASE 1: SPECTRAL PROFILE
    # ======================================================================

    def phase_spectral_profile(self) -> PhaseResult:
        """FFT, PSD, dominant frequencies, spectral shape classification."""
        ph = PhaseResult("Phase 1: Spectral Profile")

        if self.n < 8:
            ph.add("length", Verdict.REJECT, Severity.CRITICAL,
                    f"Only {self.n} samples — cannot perform FFT")
            self._report.phases.append(ph)
            return ph

        self._ensure_spectrum()
        self._ensure_psd()
        self._ensure_peaks()

        nyquist = self.fs / 2
        freq_res = self.fs / self.n

        ph.add("fft_config", Verdict.PASS, Severity.INFO,
                f"N={self.n}, Fs={self.fs} Hz, Nyquist={nyquist} Hz, "
                f"freq resolution={freq_res:.4f} Hz",
                {"n": self.n, "fs": self.fs, "nyquist": nyquist,
                 "freq_resolution": round(freq_res, 6)})

        # dominant frequencies
        if self._dominant:
            top5 = self._dominant[:5]
            lines = [f"{f:.2f} Hz (amp={a:.4f})" for f, a in top5]
            ph.add("dominant_frequencies", Verdict.PASS, Severity.INFO,
                    f"Top {len(top5)} peaks: " + ", ".join(lines),
                    [{"freq": round(f, 4), "amplitude": round(a, 6)} for f, a in top5])
        else:
            ph.add("dominant_frequencies", Verdict.WARN, Severity.MEDIUM,
                    "No clear spectral peaks detected — signal may be noise-like")

        # DC offset check
        dc_amp = float(self._mag[0])
        rms = float(np.sqrt(np.mean(self._x ** 2)))
        dc_ratio = dc_amp / rms if rms > 0 else 0
        if dc_ratio > 10.0:
            ph.add("dc_offset", Verdict.WARN, Severity.MEDIUM,
                    f"DC component ({dc_amp:.4f}) dominates signal (ratio={dc_ratio:.1f}x RMS) — "
                    "consider detrending before spectral analysis",
                    {"dc": round(dc_amp, 6), "rms": round(rms, 6)})
        else:
            ph.add("dc_offset", Verdict.PASS, Severity.INFO,
                    f"DC={dc_amp:.4f}, RMS={rms:.4f}",
                    {"dc": round(dc_amp, 6), "rms": round(rms, 6)})

        # spectral flatness (tonality vs noise)
        flatness = _spectral_flatness(self._psd)
        if flatness > 0.8:
            shape = "noise-like (white/pink noise)"
        elif flatness > 0.3:
            shape = "mixed (tonal + noise)"
        else:
            shape = "tonal (discrete frequency components)"

        ph.add("spectral_flatness", Verdict.PASS, Severity.INFO,
                f"Spectral flatness: {flatness:.4f} — {shape}",
                round(flatness, 6))

        # spectral centroid
        centroid = _spectral_centroid(self._freqs, self._mag)
        ph.add("spectral_centroid", Verdict.PASS, Severity.INFO,
                f"Spectral centroid: {centroid:.2f} Hz", round(centroid, 4))

        # store for later phases
        self._report.spectra["frequencies"] = self._freqs
        self._report.spectra["magnitude"] = self._mag
        self._report.spectra["psd_freqs"] = self._psd_f
        self._report.spectra["psd"] = self._psd

        self._report.phases.append(ph)
        return ph

    # ======================================================================
    # PHASE 2: HARMONIC ANALYSIS
    # ======================================================================

    def phase_harmonic_analysis(self, fundamental: Optional[float] = None,
                                n_harmonics: int = 10) -> PhaseResult:
        """
        Detect fundamental frequency, harmonics, THD, sidebands.
        If fundamental not provided, infer from strongest spectral peak.
        """
        ph = PhaseResult("Phase 2: Harmonic Analysis")
        self._ensure_spectrum()
        self._ensure_peaks()

        if not self._dominant:
            ph.add("harmonics", Verdict.SKIP, Severity.INFO,
                    "No spectral peaks to analyze for harmonics")
            self._report.phases.append(ph)
            return ph

        # determine fundamental
        if fundamental is None:
            # strongest non-DC peak
            candidates = [(f, a) for f, a in self._dominant if f > 0]
            if not candidates:
                ph.add("fundamental", Verdict.SKIP, Severity.INFO, "No non-DC peaks")
                self._report.phases.append(ph)
                return ph
            fundamental = candidates[0][0]
            fund_amp = candidates[0][1]
        else:
            # find amplitude at specified fundamental
            idx = np.argmin(np.abs(self._freqs - fundamental))
            fund_amp = float(self._mag[idx])

        ph.add("fundamental", Verdict.PASS, Severity.INFO,
                f"Fundamental: {fundamental:.4f} Hz (amplitude={fund_amp:.6f})",
                {"frequency": round(fundamental, 4), "amplitude": round(fund_amp, 6)})

        # search for harmonics (2f, 3f, 4f, ...)
        freq_res = self.fs / self.n
        tolerance = max(freq_res * 2, fundamental * 0.02)
        harmonic_amps = []
        harmonics_found = []

        for h in range(2, n_harmonics + 1):
            target = fundamental * h
            if target > self.fs / 2:
                break
            idx = np.argmin(np.abs(self._freqs - target))
            actual_freq = float(self._freqs[idx])
            amp = float(self._mag[idx])

            # check if this is a real peak (above local neighborhood)
            lo = max(0, idx - 5)
            hi = min(len(self._mag), idx + 6)
            local_median = float(np.median(self._mag[lo:hi]))

            if amp > local_median * 2.0 and abs(actual_freq - target) < tolerance:
                harmonics_found.append({
                    "order": h,
                    "expected_freq": round(target, 4),
                    "actual_freq": round(actual_freq, 4),
                    "amplitude": round(amp, 6),
                    "ratio_to_fundamental": round(amp / fund_amp, 4) if fund_amp > 0 else 0,
                })
                harmonic_amps.append(amp)

        if harmonics_found:
            ph.add("harmonics_detected", Verdict.PASS, Severity.INFO,
                    f"{len(harmonics_found)} harmonics detected (orders: "
                    f"{[h['order'] for h in harmonics_found]})",
                    harmonics_found)
        else:
            ph.add("harmonics_detected", Verdict.PASS, Severity.INFO,
                    "No significant harmonics detected — signal appears sinusoidal or broadband")

        # THD (Total Harmonic Distortion)
        if harmonic_amps and fund_amp > 0:
            thd = math.sqrt(sum(a ** 2 for a in harmonic_amps)) / fund_amp * 100
            if thd > 50:
                ph.add("thd", Verdict.WARN, Severity.HIGH,
                        f"THD = {thd:.2f}% — very high harmonic distortion. "
                        "System may be heavily nonlinear or clipping.",
                        round(thd, 2))
            elif thd > 10:
                ph.add("thd", Verdict.WARN, Severity.MEDIUM,
                        f"THD = {thd:.2f}% — moderate distortion. "
                        "Check for nonlinear behavior.", round(thd, 2))
            elif thd > 1:
                ph.add("thd", Verdict.PASS, Severity.INFO,
                        f"THD = {thd:.2f}% — low distortion", round(thd, 2))
            else:
                ph.add("thd", Verdict.PASS, Severity.INFO,
                        f"THD = {thd:.4f}% — excellent", round(thd, 4))
        else:
            ph.add("thd", Verdict.PASS, Severity.INFO,
                    "THD not computable (no harmonics or zero fundamental)")

        # sideband detection (peaks near fundamental +/- small offset)
        sideband_range = fundamental * 0.15
        sidebands = []
        for f, a in self._dominant:
            offset = abs(f - fundamental)
            if 0 < offset < sideband_range and a > fund_amp * 0.05:
                sidebands.append({"freq": round(f, 4), "offset": round(offset, 4),
                                  "amplitude": round(a, 6)})

        if sidebands:
            ph.add("sidebands", Verdict.WARN, Severity.MEDIUM,
                    f"{len(sidebands)} sideband(s) near fundamental — "
                    "possible modulation, bearing fault, or mechanical looseness",
                    sidebands)
        else:
            ph.add("sidebands", Verdict.PASS, Severity.INFO,
                    "No significant sidebands detected near fundamental")

        self._report.phases.append(ph)
        return ph

    # ======================================================================
    # PHASE 3: WINDOWING QUALITY
    # ======================================================================

    def phase_windowing_quality(self) -> PhaseResult:
        """
        Assess spectral leakage, compare window functions, recommend best window.
        Key insight: spectral leakage occurs when signal frequencies don't align
        with FFT bin centers. Windowing trades resolution for leakage suppression.
        """
        ph = PhaseResult("Phase 3: Windowing Quality")

        if self.n < 64:
            ph.add("windowing", Verdict.SKIP, Severity.INFO, "Signal too short for window analysis")
            self._report.phases.append(ph)
            return ph

        # compare spectra with different windows
        windows = ["boxcar", "hann", "hamming", "blackmanharris", "flattop"] if _HAS_SCIPY else ["boxcar", "hann"]
        window_results = {}

        for wname in windows:
            try:
                freqs, mag = _amplitude_spectrum(self._x, self.fs, window=wname)
                # spectral leakage metric: ratio of energy in sidelobes vs main lobe
                # approximate via standard deviation of log-magnitude
                log_mag = np.log10(mag[mag > 0] + 1e-30)
                window_results[wname] = {
                    "peak_amp": float(np.max(mag[1:])),
                    "noise_floor": float(np.median(mag[1:])),
                    "dynamic_range_db": round(
                        20 * np.log10(np.max(mag[1:]) / max(np.median(mag[1:]), 1e-30)), 2
                    ),
                    "log_mag_std": round(float(np.std(log_mag)), 4),
                }
            except Exception:
                continue

        if len(window_results) >= 2:
            best_dr = max(window_results.items(), key=lambda x: x[1]["dynamic_range_db"])
            ph.add("window_comparison", Verdict.PASS, Severity.INFO,
                    f"Best dynamic range: {best_dr[0]} ({best_dr[1]['dynamic_range_db']:.1f} dB). "
                    f"Tested: {list(window_results.keys())}",
                    {k: v["dynamic_range_db"] for k, v in window_results.items()})

            # check if rectangular window leaks badly
            if "boxcar" in window_results and "hann" in window_results:
                rect_dr = window_results["boxcar"]["dynamic_range_db"]
                hann_dr = window_results["hann"]["dynamic_range_db"]
                improvement = hann_dr - rect_dr
                if improvement > 10:
                    ph.add("leakage_detected", Verdict.WARN, Severity.MEDIUM,
                            f"Hann window improves dynamic range by {improvement:.1f} dB over rectangular — "
                            "significant spectral leakage present. Use Hann or Blackman-Harris.",
                            round(improvement, 2))
                else:
                    ph.add("leakage_detected", Verdict.PASS, Severity.INFO,
                            f"Window choice improves dynamic range by {improvement:.1f} dB — "
                            "leakage is manageable", round(improvement, 2))

        # resolution vs leakage trade-off guidance
        freq_res = self.fs / self.n
        ph.add("freq_resolution", Verdict.PASS, Severity.INFO,
                f"Frequency resolution: {freq_res:.6f} Hz "
                f"(bin width). Hann window effective resolution: ~{1.5 * freq_res:.6f} Hz",
                {"bin_width": round(freq_res, 6),
                 "hann_resolution": round(1.5 * freq_res, 6)})

        # Nyquist check
        self._ensure_peaks()
        if self._dominant:
            max_freq = max(f for f, _ in self._dominant)
            nyq = self.fs / 2
            if max_freq > nyq * 0.9:
                ph.add("nyquist_proximity", Verdict.WARN, Severity.HIGH,
                        f"Dominant frequency {max_freq:.2f} Hz is within 10% of Nyquist ({nyq} Hz) — "
                        "risk of aliasing. Consider increasing sample rate.",
                        {"max_freq": round(max_freq, 4), "nyquist": nyq})
            else:
                ph.add("nyquist_proximity", Verdict.PASS, Severity.INFO,
                        f"Max dominant frequency {max_freq:.2f} Hz is well below Nyquist ({nyq} Hz)")

        self._report.phases.append(ph)
        return ph

    # ======================================================================
    # PHASE 4: NOISE FLOOR
    # ======================================================================

    def phase_noise_floor(self) -> PhaseResult:
        """Noise floor estimation, SNR, dynamic range."""
        ph = PhaseResult("Phase 4: Noise Floor")
        self._ensure_psd()
        self._ensure_peaks()

        if self._psd is None or len(self._psd) < 4:
            ph.add("noise_floor", Verdict.SKIP, Severity.INFO, "Insufficient spectral data")
            self._report.phases.append(ph)
            return ph

        # noise floor = median of PSD (robust to peaks)
        psd_pos = self._psd[self._psd > 0]
        if len(psd_pos) == 0:
            ph.add("noise_floor", Verdict.SKIP, Severity.INFO, "Zero PSD")
            self._report.phases.append(ph)
            return ph

        noise_floor_psd = float(np.median(psd_pos))
        noise_floor_db = 10 * np.log10(noise_floor_psd) if noise_floor_psd > 0 else -np.inf
        peak_psd = float(np.max(psd_pos))
        peak_db = 10 * np.log10(peak_psd) if peak_psd > 0 else -np.inf
        dynamic_range = peak_db - noise_floor_db

        ph.add("noise_floor_level", Verdict.PASS, Severity.INFO,
                f"Noise floor: {noise_floor_db:.1f} dB/Hz, "
                f"Peak: {peak_db:.1f} dB/Hz, Dynamic range: {dynamic_range:.1f} dB",
                {"noise_floor_db": round(noise_floor_db, 2),
                 "peak_db": round(peak_db, 2),
                 "dynamic_range_db": round(dynamic_range, 2)})

        # SNR estimate (signal power / noise power)
        if self._dominant and len(self._dominant) > 0:
            # signal power: sum of peak amplitudes squared
            signal_power = sum(a ** 2 for _, a in self._dominant[:5])
            total_power = float(np.mean(self._x ** 2))
            noise_power = max(total_power - signal_power, 0)
            snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float("inf")

            if snr < 3:
                ph.add("snr", Verdict.WARN, Severity.HIGH,
                        f"SNR = {snr:.1f} dB — signal buried in noise", round(snr, 2))
            elif snr < 10:
                ph.add("snr", Verdict.WARN, Severity.MEDIUM,
                        f"SNR = {snr:.1f} dB — low signal quality", round(snr, 2))
            elif snr < 20:
                ph.add("snr", Verdict.PASS, Severity.INFO,
                        f"SNR = {snr:.1f} dB — acceptable", round(snr, 2))
            else:
                ph.add("snr", Verdict.PASS, Severity.INFO,
                        f"SNR = {snr:.1f} dB — good signal quality", round(snr, 2))

        # spectral noise color (slope of PSD in log-log)
        if len(self._psd_f) > 10:
            f_pos = self._psd_f[self._psd_f > 0]
            p_pos = self._psd[self._psd_f > 0]
            if len(f_pos) > 10 and np.all(p_pos > 0):
                if _HAS_SCIPY:
                    log_f = np.log10(f_pos)
                    log_p = np.log10(p_pos)
                    slope, _, r_value, _, _ = sp_stats.linregress(log_f, log_p)

                    if abs(slope) < 0.3:
                        color = "white noise (~flat)"
                    elif -1.5 < slope < -0.5:
                        color = f"pink/flicker noise (1/f, slope={slope:.2f})"
                    elif slope < -1.5:
                        color = f"brown/red noise (1/f^2, slope={slope:.2f})"
                    else:
                        color = f"slope={slope:.2f}"

                    ph.add("noise_color", Verdict.PASS, Severity.INFO,
                            f"PSD slope: {slope:.3f} (R2={r_value**2:.3f}) — {color}",
                            {"slope": round(slope, 4), "r2": round(r_value ** 2, 4)})

        self._report.phases.append(ph)
        return ph

    # ======================================================================
    # PHASE 5: BANDWIDTH ANALYSIS
    # ======================================================================

    def phase_bandwidth_analysis(self) -> PhaseResult:
        """Spectral rolloff, centroid, bandwidth, energy distribution."""
        ph = PhaseResult("Phase 5: Bandwidth Analysis")
        self._ensure_spectrum()
        self._ensure_psd()

        nyq = self.fs / 2

        # spectral rolloff
        rolloff_85 = _spectral_rolloff(self._freqs, self._mag, 0.85)
        rolloff_95 = _spectral_rolloff(self._freqs, self._mag, 0.95)
        rolloff_99 = _spectral_rolloff(self._freqs, self._mag, 0.99)

        ph.add("spectral_rolloff", Verdict.PASS, Severity.INFO,
                f"85% energy below {rolloff_85:.2f} Hz, "
                f"95% below {rolloff_95:.2f} Hz, "
                f"99% below {rolloff_99:.2f} Hz",
                {"rolloff_85": round(rolloff_85, 4),
                 "rolloff_95": round(rolloff_95, 4),
                 "rolloff_99": round(rolloff_99, 4)})

        # oversampling assessment
        if rolloff_99 < nyq * 0.2:
            ph.add("oversampling", Verdict.WARN, Severity.LOW,
                    f"99% of energy below {rolloff_99:.2f} Hz but Nyquist is {nyq} Hz — "
                    f"signal is oversampled by ~{nyq / rolloff_99:.1f}x. "
                    "Could downsample to save bandwidth.",
                    round(nyq / rolloff_99, 2))

        # band energy distribution
        bands = []
        n_bands = 5
        band_width = nyq / n_bands
        total_energy = _band_energy(self._psd_f, self._psd, 0, nyq)

        for i in range(n_bands):
            f_lo = i * band_width
            f_hi = (i + 1) * band_width
            e = _band_energy(self._psd_f, self._psd, f_lo, f_hi)
            pct = e / total_energy * 100 if total_energy > 0 else 0
            bands.append({
                "band": f"{f_lo:.0f}-{f_hi:.0f} Hz",
                "energy_pct": round(pct, 2)
            })

        ph.add("band_energy_distribution", Verdict.PASS, Severity.INFO,
                f"Energy distribution across {n_bands} bands (% of total)",
                bands)

        # effective bandwidth (ENBW)
        centroid = _spectral_centroid(self._freqs, self._mag)
        spread = _spectral_centroid(self._freqs, (self._freqs - centroid) ** 2 * self._mag)
        spread = math.sqrt(spread) if spread > 0 else 0

        ph.add("spectral_spread", Verdict.PASS, Severity.INFO,
                f"Centroid: {centroid:.2f} Hz, Spread (std): {spread:.2f} Hz",
                {"centroid": round(centroid, 4), "spread": round(spread, 4)})

        self._report.phases.append(ph)
        return ph

    # ======================================================================
    # PHASE 6: SYSTEM IDENTIFICATION (needs input/output pair)
    # ======================================================================

    def phase_system_identification(
        self,
        input_signal: np.ndarray,
        output_signal: np.ndarray,
    ) -> PhaseResult:
        """
        Transfer function (H1/H2), coherence, frequency response.
        Uses cross-spectral density method:
            H1(f) = Sxy(f) / Sxx(f)    (noise on output)
            H2(f) = Syy(f) / Syx(f)    (noise on input)
            Coherence = |Sxy|^2 / (Sxx * Syy)
        """
        ph = PhaseResult("Phase 6: System Identification")

        if not _HAS_SCIPY:
            ph.add("sys_id", Verdict.SKIP, Severity.INFO,
                    "System identification requires scipy")
            self._report.phases.append(ph)
            return ph

        inp = np.asarray(input_signal, dtype=np.float64).ravel()
        out = np.asarray(output_signal, dtype=np.float64).ravel()
        n_min = min(len(inp), len(out))

        if n_min < 64:
            ph.add("sys_id", Verdict.SKIP, Severity.INFO,
                    "Insufficient samples for system identification")
            self._report.phases.append(ph)
            return ph

        inp, out = inp[:n_min], out[:n_min]
        nperseg = min(n_min, max(256, _next_pow2(n_min // 8)))

        # coherence
        f_coh, coh = sp_coherence(inp, out, fs=self.fs, nperseg=nperseg)

        mean_coh = float(np.mean(coh))
        min_coh = float(np.min(coh))
        high_coh_pct = float(np.mean(coh > 0.8)) * 100

        if mean_coh < 0.3:
            ph.add("coherence", Verdict.FAIL, Severity.HIGH,
                    f"Mean coherence={mean_coh:.3f} — input/output are barely correlated. "
                    "System identification unreliable. Check: wrong input signal, "
                    "significant noise, or nonlinear system.",
                    {"mean": round(mean_coh, 4), "min": round(min_coh, 4)})
        elif mean_coh < 0.6:
            ph.add("coherence", Verdict.WARN, Severity.MEDIUM,
                    f"Mean coherence={mean_coh:.3f} — moderate I/O correlation. "
                    f"{high_coh_pct:.0f}% of bins above 0.8",
                    {"mean": round(mean_coh, 4), "high_coh_pct": round(high_coh_pct, 1)})
        else:
            ph.add("coherence", Verdict.PASS, Severity.INFO,
                    f"Mean coherence={mean_coh:.3f}. "
                    f"{high_coh_pct:.0f}% of bins above 0.8 — good I/O relationship.",
                    {"mean": round(mean_coh, 4), "high_coh_pct": round(high_coh_pct, 1)})

        # transfer function H1
        f_csd, pxy = csd(inp, out, fs=self.fs, nperseg=nperseg)
        _, pxx = welch(inp, fs=self.fs, nperseg=nperseg)
        H1 = pxy / (pxx + 1e-30)
        H1_mag = np.abs(H1)
        H1_phase = np.angle(H1, deg=True)

        # gain characteristics
        gain_db = 20 * np.log10(H1_mag + 1e-30)
        peak_gain = float(np.max(gain_db))
        peak_gain_freq = float(f_csd[np.argmax(gain_db)])

        ph.add("transfer_function", Verdict.PASS, Severity.INFO,
                f"Peak gain: {peak_gain:.1f} dB at {peak_gain_freq:.2f} Hz",
                {"peak_gain_db": round(peak_gain, 2),
                 "peak_freq": round(peak_gain_freq, 4)})

        # resonance detection (sharp peaks in gain)
        if _HAS_SCIPY and len(gain_db) > 10:
            try:
                res_peaks, _ = find_peaks(gain_db, prominence=6, distance=5)
                if len(res_peaks) > 0:
                    resonances = [{"freq": round(float(f_csd[p]), 4),
                                   "gain_db": round(float(gain_db[p]), 2)}
                                  for p in res_peaks[:5]]
                    ph.add("resonances", Verdict.WARN, Severity.MEDIUM,
                            f"{len(res_peaks)} resonance(s) detected: "
                            f"{[r['freq'] for r in resonances[:3]]} Hz",
                            resonances)
                else:
                    ph.add("resonances", Verdict.PASS, Severity.INFO,
                            "No sharp resonances detected in transfer function")
            except Exception:
                pass

        # phase characteristics
        phase_at_peak = float(H1_phase[np.argmax(gain_db)])
        ph.add("phase_at_peak", Verdict.PASS, Severity.INFO,
                f"Phase at peak gain frequency: {phase_at_peak:.1f} deg",
                round(phase_at_peak, 2))

        # store FRF data
        self._report.spectra["coherence_freqs"] = f_coh
        self._report.spectra["coherence"] = coh
        self._report.spectra["tf_freqs"] = f_csd
        self._report.spectra["tf_gain_db"] = gain_db
        self._report.spectra["tf_phase_deg"] = H1_phase

        self._report.phases.append(ph)
        return ph

    # ======================================================================
    # PHASE 7: SPECTRAL ANOMALY DETECTION
    # ======================================================================

    def phase_spectral_anomaly(
        self,
        reference_signal: np.ndarray,
        *,
        ref_fs: Optional[float] = None,
    ) -> PhaseResult:
        """
        Compare current spectrum against a reference (baseline) spectrum.
        Detects: new peaks, missing peaks, energy shifts, spectral distance.
        """
        ph = PhaseResult("Phase 7: Spectral Anomaly Detection")
        self._ensure_psd()

        ref = np.asarray(reference_signal, dtype=np.float64).ravel()
        rfs = ref_fs or self.fs
        ref_f, ref_psd = _compute_psd(ref, rfs)

        # interpolate to common frequency grid if needed
        if len(ref_f) != len(self._psd_f) or not np.allclose(ref_f, self._psd_f):
            ref_psd_interp = np.interp(self._psd_f, ref_f, ref_psd)
        else:
            ref_psd_interp = ref_psd

        # spectral distance (log-spectral distance)
        eps = 1e-30
        with np.errstate(divide="ignore", invalid="ignore"):
            lsd = np.sqrt(np.mean(
                (10 * np.log10(self._psd + eps) - 10 * np.log10(ref_psd_interp + eps)) ** 2
            ))

        if lsd > 20:
            ph.add("spectral_distance", Verdict.FAIL, Severity.HIGH,
                    f"Log-spectral distance: {lsd:.2f} dB — major spectral change from reference",
                    round(lsd, 2))
        elif lsd > 10:
            ph.add("spectral_distance", Verdict.WARN, Severity.MEDIUM,
                    f"Log-spectral distance: {lsd:.2f} dB — noticeable spectral change",
                    round(lsd, 2))
        else:
            ph.add("spectral_distance", Verdict.PASS, Severity.INFO,
                    f"Log-spectral distance: {lsd:.2f} dB — spectra are similar",
                    round(lsd, 2))

        # band-by-band energy comparison
        n_bands = 8
        nyq = self.fs / 2
        bw = nyq / n_bands
        band_diffs = []

        for i in range(n_bands):
            f_lo, f_hi = i * bw, (i + 1) * bw
            e_cur = _band_energy(self._psd_f, self._psd, f_lo, f_hi)
            e_ref = _band_energy(self._psd_f, ref_psd_interp, f_lo, f_hi)
            if e_ref > 0:
                ratio_db = 10 * np.log10((e_cur + eps) / (e_ref + eps))
            else:
                ratio_db = 0
            band_diffs.append({
                "band": f"{f_lo:.0f}-{f_hi:.0f} Hz",
                "change_db": round(float(ratio_db), 2)
            })

        alarming_bands = [b for b in band_diffs if abs(b["change_db"]) > 6]
        if alarming_bands:
            ph.add("band_energy_shift", Verdict.WARN, Severity.HIGH,
                    f"{len(alarming_bands)} band(s) changed >6 dB: "
                    f"{[b['band'] for b in alarming_bands]}",
                    alarming_bands)
        else:
            ph.add("band_energy_shift", Verdict.PASS, Severity.INFO,
                    "All frequency bands within 6 dB of reference", band_diffs)

        # new peak detection
        cur_peaks = _find_dominant_freqs(*_amplitude_spectrum(self._x, self.fs), n_peaks=15)
        ref_peaks = _find_dominant_freqs(*_amplitude_spectrum(ref, rfs), n_peaks=15)
        ref_freqs = {round(f, 1) for f, _ in ref_peaks}
        freq_tol = self.fs / self.n * 3

        new_peaks = []
        for f, a in cur_peaks:
            is_new = all(abs(f - rf) > freq_tol for rf in ref_freqs)
            if is_new and a > 0:
                new_peaks.append({"freq": round(f, 4), "amplitude": round(a, 6)})

        if new_peaks:
            ph.add("new_peaks", Verdict.WARN, Severity.MEDIUM,
                    f"{len(new_peaks)} new spectral peak(s) not in reference: "
                    f"{[p['freq'] for p in new_peaks[:5]]} Hz",
                    new_peaks[:10])
        else:
            ph.add("new_peaks", Verdict.PASS, Severity.INFO,
                    "No new spectral peaks relative to reference")

        self._report.phases.append(ph)
        return ph

    # ======================================================================
    # PHASE 8: TIME-FREQUENCY ANALYSIS
    # ======================================================================

    def phase_time_frequency(self, nperseg: int = 0) -> PhaseResult:
        """STFT / spectrogram analysis — spectral evolution over time."""
        ph = PhaseResult("Phase 8: Time-Frequency Analysis")

        if not _HAS_SCIPY:
            ph.add("stft", Verdict.SKIP, Severity.INFO,
                    "Time-frequency analysis requires scipy")
            self._report.phases.append(ph)
            return ph

        if self.n < 128:
            ph.add("stft", Verdict.SKIP, Severity.INFO,
                    "Signal too short for meaningful STFT")
            self._report.phases.append(ph)
            return ph

        if nperseg == 0:
            nperseg = min(self.n // 4, max(128, _next_pow2(self.n // 16)))

        try:
            f_stft, t_stft, Zxx = sp_stft(self._x, fs=self.fs, nperseg=nperseg,
                                           noverlap=nperseg * 3 // 4)
            power = np.abs(Zxx) ** 2

            n_time_bins = power.shape[1]
            n_freq_bins = power.shape[0]

            ph.add("stft_config", Verdict.PASS, Severity.INFO,
                    f"STFT: {n_time_bins} time bins x {n_freq_bins} freq bins, "
                    f"segment={nperseg} samples",
                    {"time_bins": n_time_bins, "freq_bins": n_freq_bins})

            # spectral stationarity: compare PSD across time windows
            if n_time_bins >= 4:
                band_powers = []
                for t_idx in range(n_time_bins):
                    total_p = float(np.sum(power[:, t_idx]))
                    band_powers.append(total_p)

                bp = np.array(band_powers)
                bp_cv = float(np.std(bp) / np.mean(bp)) if np.mean(bp) > 0 else 0

                if bp_cv > 0.5:
                    ph.add("spectral_stationarity", Verdict.WARN, Severity.HIGH,
                            f"Total power CV={bp_cv:.3f} across time — "
                            "signal is spectrally non-stationary. "
                            "Spectral features change significantly over time.",
                            round(bp_cv, 4))
                elif bp_cv > 0.2:
                    ph.add("spectral_stationarity", Verdict.WARN, Severity.LOW,
                            f"Total power CV={bp_cv:.3f} — mildly non-stationary",
                            round(bp_cv, 4))
                else:
                    ph.add("spectral_stationarity", Verdict.PASS, Severity.INFO,
                            f"Total power CV={bp_cv:.3f} — spectrally stationary",
                            round(bp_cv, 4))

                # track spectral centroid evolution
                centroids = []
                for t_idx in range(n_time_bins):
                    col = power[:, t_idx]
                    total = np.sum(col)
                    if total > 0:
                        centroids.append(float(np.sum(f_stft * col) / total))
                    else:
                        centroids.append(0.0)

                c_arr = np.array(centroids)
                centroid_drift = float(np.max(c_arr) - np.min(c_arr))
                centroid_cv = float(np.std(c_arr) / np.mean(c_arr)) if np.mean(c_arr) > 0 else 0

                if centroid_cv > 0.15:
                    ph.add("centroid_evolution", Verdict.WARN, Severity.MEDIUM,
                            f"Spectral centroid drifts (CV={centroid_cv:.3f}, "
                            f"range={centroid_drift:.2f} Hz) — "
                            "frequency content shifts over time",
                            {"cv": round(centroid_cv, 4),
                             "range_hz": round(centroid_drift, 4)})
                else:
                    ph.add("centroid_evolution", Verdict.PASS, Severity.INFO,
                            f"Spectral centroid stable (CV={centroid_cv:.3f})",
                            round(centroid_cv, 4))

                # transient detection: sudden power spikes
                if len(band_powers) > 4:
                    bp_diff = np.diff(bp)
                    bp_diff_z = (bp_diff - np.mean(bp_diff)) / max(np.std(bp_diff), 1e-30)
                    transients = np.where(np.abs(bp_diff_z) > 3)[0]
                    if len(transients) > 0:
                        t_times = [round(float(t_stft[min(t, len(t_stft) - 1)]), 4)
                                   for t in transients[:5]]
                        ph.add("transients", Verdict.WARN, Severity.MEDIUM,
                                f"{len(transients)} spectral transient(s) detected at t={t_times}s",
                                t_times)
                    else:
                        ph.add("transients", Verdict.PASS, Severity.INFO,
                                "No spectral transients detected")

        except Exception as e:
            ph.add("stft_error", Verdict.SKIP, Severity.INFO, f"STFT failed: {e}")

        self._report.phases.append(ph)
        return ph

    # ======================================================================
    # PHASE 9: SYSTEM HEALTH (vibration / machinery diagnostics)
    # ======================================================================

    def phase_system_health(
        self,
        *,
        shaft_rpm: Optional[float] = None,
        bearing_bpfo: Optional[float] = None,
        bearing_bpfi: Optional[float] = None,
        bearing_bsf: Optional[float] = None,
        bearing_ftf: Optional[float] = None,
    ) -> PhaseResult:
        """
        Vibration-oriented diagnostics: crest factor, kurtosis,
        bearing fault frequency monitoring, spectral fingerprinting.

        Parameters
        ----------
        shaft_rpm : float, optional
            Shaft speed in RPM. Enables shaft-frequency analysis.
        bearing_bpfo/bpfi/bsf/ftf : float, optional
            Bearing characteristic fault frequencies (in Hz).
            BPFO = Ball Pass Freq Outer, BPFI = Inner,
            BSF = Ball Spin Freq, FTF = Fundamental Train Freq.
        """
        ph = PhaseResult("Phase 9: System Health Diagnostics")
        self._ensure_spectrum()
        self._ensure_psd()

        x = self._x

        # --- time-domain health indicators --------------------------------

        # RMS level
        rms = float(np.sqrt(np.mean(x ** 2)))
        ph.add("rms_level", Verdict.PASS, Severity.INFO,
                f"RMS level: {rms:.6f}", round(rms, 6))

        # Crest factor (peak / RMS) — elevated indicates impulsive events
        peak = float(np.max(np.abs(x)))
        crest = peak / rms if rms > 0 else 0
        if crest > 6.0:
            ph.add("crest_factor", Verdict.WARN, Severity.HIGH,
                    f"Crest factor={crest:.2f} — very impulsive signal. "
                    "Possible bearing defect, gear tooth fault, or impact events.",
                    round(crest, 3))
        elif crest > 4.0:
            ph.add("crest_factor", Verdict.WARN, Severity.MEDIUM,
                    f"Crest factor={crest:.2f} — mildly impulsive", round(crest, 3))
        else:
            ph.add("crest_factor", Verdict.PASS, Severity.INFO,
                    f"Crest factor={crest:.2f} — normal range", round(crest, 3))

        # Kurtosis (>3 indicates heavy tails / impulsiveness)
        kurt = float(np.mean((x - np.mean(x)) ** 4) / (np.std(x) ** 4)) if np.std(x) > 0 else 3
        if kurt > 7:
            ph.add("kurtosis", Verdict.WARN, Severity.HIGH,
                    f"Kurtosis={kurt:.2f} — super-Gaussian, strong impulsive events",
                    round(kurt, 3))
        elif kurt > 4:
            ph.add("kurtosis", Verdict.WARN, Severity.MEDIUM,
                    f"Kurtosis={kurt:.2f} — leptokurtic, some impulsiveness", round(kurt, 3))
        else:
            ph.add("kurtosis", Verdict.PASS, Severity.INFO,
                    f"Kurtosis={kurt:.2f} — near-Gaussian", round(kurt, 3))

        # --- shaft frequency analysis ------------------------------------
        shaft_freq = None
        if shaft_rpm is not None:
            shaft_freq = shaft_rpm / 60.0
            # check for 1x, 2x, 3x shaft frequency
            freq_res = self.fs / self.n
            tolerance = max(freq_res * 2, shaft_freq * 0.03)

            shaft_harmonics = []
            for order in range(1, 6):
                target = shaft_freq * order
                if target > self.fs / 2:
                    break
                idx = np.argmin(np.abs(self._freqs - target))
                amp = float(self._mag[idx])
                shaft_harmonics.append({
                    "order": f"{order}x",
                    "expected_hz": round(target, 4),
                    "amplitude": round(amp, 6),
                })

            ph.add("shaft_harmonics", Verdict.PASS, Severity.INFO,
                    f"Shaft freq={shaft_freq:.2f} Hz ({shaft_rpm} RPM). "
                    f"1x amp={shaft_harmonics[0]['amplitude']:.6f}",
                    shaft_harmonics)

            # unbalance indicator (1x dominant)
            if len(shaft_harmonics) >= 2:
                ratio_2x_1x = shaft_harmonics[1]["amplitude"] / max(shaft_harmonics[0]["amplitude"], 1e-30)
                if ratio_2x_1x > 0.5:
                    ph.add("misalignment_indicator", Verdict.WARN, Severity.MEDIUM,
                            f"2x/1x amplitude ratio={ratio_2x_1x:.2f} — "
                            "possible misalignment (2x should be << 1x for balanced system)",
                            round(ratio_2x_1x, 3))

        # --- bearing fault frequency monitoring --------------------------
        fault_freqs = {
            "BPFO": bearing_bpfo,
            "BPFI": bearing_bpfi,
            "BSF": bearing_bsf,
            "FTF": bearing_ftf,
        }
        active_faults = {k: v for k, v in fault_freqs.items() if v is not None}

        if active_faults:
            freq_res = self.fs / self.n
            for fault_name, fault_hz in active_faults.items():
                if fault_hz > self.fs / 2:
                    continue
                idx = np.argmin(np.abs(self._freqs - fault_hz))
                amp = float(self._mag[idx])
                # compare to local median
                lo = max(0, idx - 20)
                hi = min(len(self._mag), idx + 21)
                local_median = float(np.median(self._mag[lo:hi]))
                prominence = amp / max(local_median, 1e-30)

                if prominence > 5.0:
                    ph.add(f"bearing_{fault_name}", Verdict.WARN, Severity.HIGH,
                            f"{fault_name} at {fault_hz:.2f} Hz: amplitude={amp:.6f}, "
                            f"prominence={prominence:.1f}x local median — "
                            "POSSIBLE BEARING FAULT",
                            {"freq": fault_hz, "amp": round(amp, 6),
                             "prominence": round(prominence, 2)})
                elif prominence > 3.0:
                    ph.add(f"bearing_{fault_name}", Verdict.WARN, Severity.MEDIUM,
                            f"{fault_name} at {fault_hz:.2f} Hz: prominence={prominence:.1f}x — "
                            "elevated, monitor closely",
                            round(prominence, 2))
                else:
                    ph.add(f"bearing_{fault_name}", Verdict.PASS, Severity.INFO,
                            f"{fault_name} at {fault_hz:.2f} Hz: prominence={prominence:.1f}x — normal",
                            round(prominence, 2))

        # --- spectral fingerprint summary ---------------------------------
        self._ensure_peaks()
        if self._dominant:
            fingerprint = [{"freq": round(f, 4), "amp": round(a, 6)}
                           for f, a in self._dominant[:8]]
            ph.add("spectral_fingerprint", Verdict.PASS, Severity.INFO,
                    f"Spectral fingerprint: {len(fingerprint)} dominant frequencies",
                    fingerprint)

        self._report.phases.append(ph)
        return ph

    # ======================================================================
    # FULL ANALYSIS (orchestrator)
    # ======================================================================

    def full_analysis(
        self,
        *,
        fundamental: Optional[float] = None,
        reference_signal: Optional[np.ndarray] = None,
        input_signal: Optional[np.ndarray] = None,
        output_signal: Optional[np.ndarray] = None,
        shaft_rpm: Optional[float] = None,
        bearing_bpfo: Optional[float] = None,
        bearing_bpfi: Optional[float] = None,
        bearing_bsf: Optional[float] = None,
        bearing_ftf: Optional[float] = None,
    ) -> AnalysisReport:
        """Run all applicable phases."""
        self._report = AnalysisReport(
            signal_name=self.name, n_samples=self.n, sample_rate=self.fs
        )

        # always-run phases
        self.phase_spectral_profile()
        if self._report.phases[-1].verdict == Verdict.REJECT:
            return self._report

        self.phase_harmonic_analysis(fundamental=fundamental)
        self.phase_windowing_quality()
        self.phase_noise_floor()
        self.phase_bandwidth_analysis()

        # conditional phases
        if input_signal is not None and output_signal is not None:
            self.phase_system_identification(input_signal, output_signal)

        if reference_signal is not None:
            self.phase_spectral_anomaly(reference_signal)

        self.phase_time_frequency()

        self.phase_system_health(
            shaft_rpm=shaft_rpm,
            bearing_bpfo=bearing_bpfo,
            bearing_bpfi=bearing_bpfi,
            bearing_bsf=bearing_bsf,
            bearing_ftf=bearing_ftf,
        )

        # metadata
        self._report.metadata.update({
            "duration_s": round(self.n / self.fs, 6),
            "nyquist_hz": self.fs / 2,
            "freq_resolution_hz": round(self.fs / self.n, 6),
            "rms": round(float(np.sqrt(np.mean(self._x ** 2))), 6),
            "peak": round(float(np.max(np.abs(self._x))), 6),
            "has_scipy": _HAS_SCIPY,
        })

        return self._report


# ===========================================================================
# 4. CONVENIENCE FUNCTIONS
# ===========================================================================


def quick_spectrum(signal, fs: float = 1.0, name: str = "signal") -> AnalysisReport:
    """One-liner full analysis."""
    return FourierAnalyst(signal, fs=fs, name=name).full_analysis()


def compare_spectra(
    signals: Dict[str, np.ndarray],
    fs: float = 1.0,
) -> Dict[str, Dict[str, float]]:
    """Compare PSD characteristics across multiple signals."""
    results = {}
    for name, sig in signals.items():
        x = np.asarray(sig, dtype=np.float64).ravel()
        f, psd = _compute_psd(x, fs)
        freqs, mag = _amplitude_spectrum(x, fs)
        results[name] = {
            "rms": round(float(np.sqrt(np.mean(x ** 2))), 6),
            "spectral_centroid_hz": round(_spectral_centroid(freqs, mag), 4),
            "spectral_rolloff_95_hz": round(_spectral_rolloff(freqs, mag, 0.95), 4),
            "spectral_flatness": round(_spectral_flatness(psd), 6),
            "peak_freq_hz": round(float(freqs[np.argmax(mag[1:]) + 1]), 4),
            "peak_amplitude": round(float(np.max(mag[1:])), 6),
        }
    return results


def transfer_function(
    input_signal: np.ndarray,
    output_signal: np.ndarray,
    fs: float = 1.0,
    nperseg: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Compute H1 transfer function estimate and coherence.
    Returns dict with keys: freqs, gain_db, phase_deg, coherence.
    """
    if not _HAS_SCIPY:
        raise ImportError("transfer_function requires scipy")

    inp = np.asarray(input_signal, dtype=np.float64).ravel()
    out = np.asarray(output_signal, dtype=np.float64).ravel()
    n = min(len(inp), len(out))
    inp, out = inp[:n], out[:n]

    if nperseg == 0:
        nperseg = min(n, max(256, _next_pow2(n // 8)))

    f_csd, pxy = csd(inp, out, fs=fs, nperseg=nperseg)
    _, pxx = welch(inp, fs=fs, nperseg=nperseg)
    f_coh, coh = sp_coherence(inp, out, fs=fs, nperseg=nperseg)

    H1 = pxy / (pxx + 1e-30)

    return {
        "freqs": f_csd,
        "gain_db": 20 * np.log10(np.abs(H1) + 1e-30),
        "phase_deg": np.angle(H1, deg=True),
        "coherence": coh,
    }


def spectral_distance(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    fs: float = 1.0,
) -> float:
    """Log-spectral distance (dB) between two signals."""
    _, psd_a = _compute_psd(np.asarray(sig_a, dtype=np.float64).ravel(), fs)
    _, psd_b = _compute_psd(np.asarray(sig_b, dtype=np.float64).ravel(), fs)
    n = min(len(psd_a), len(psd_b))
    eps = 1e-30
    return float(np.sqrt(np.mean(
        (10 * np.log10(psd_a[:n] + eps) - 10 * np.log10(psd_b[:n] + eps)) ** 2
    )))


def band_energy_profile(
    signal: np.ndarray,
    fs: float = 1.0,
    n_bands: int = 8,
) -> List[Dict[str, Any]]:
    """Compute energy in n_bands equally spaced frequency bands."""
    x = np.asarray(signal, dtype=np.float64).ravel()
    f, psd = _compute_psd(x, fs)
    nyq = fs / 2
    bw = nyq / n_bands
    total = _band_energy(f, psd, 0, nyq)
    bands = []
    for i in range(n_bands):
        f_lo, f_hi = i * bw, (i + 1) * bw
        e = _band_energy(f, psd, f_lo, f_hi)
        bands.append({
            "band": f"{f_lo:.0f}-{f_hi:.0f} Hz",
            "energy": round(e, 8),
            "pct": round(e / total * 100, 2) if total > 0 else 0,
        })
    return bands


# ===========================================================================
# 5. CLI DEMO
# ===========================================================================


def _demo():
    """Self-contained demo with synthetic signals."""
    np.random.seed(42)
    fs = 1000.0
    t = np.arange(0, 2.0, 1.0 / fs)

    # synthetic: 50Hz fundamental + harmonics + noise + bearing fault tone
    x = (1.0 * np.sin(2 * np.pi * 50 * t)           # fundamental
         + 0.3 * np.sin(2 * np.pi * 100 * t)         # 2nd harmonic
         + 0.1 * np.sin(2 * np.pi * 150 * t)         # 3rd harmonic
         + 0.05 * np.sin(2 * np.pi * 237 * t)        # bearing fault freq
         + 0.2 * np.random.randn(len(t)))             # noise

    print("=" * 72)
    print("  fourier_analyst.py — DEMO")
    print("=" * 72)

    fa = FourierAnalyst(x, fs=fs, name="synthetic_vibration")

    # full analysis with shaft speed and bearing info
    report = fa.full_analysis(
        shaft_rpm=3000,       # 50 Hz shaft frequency
        bearing_bpfo=237.0,   # outer race fault freq
    )
    report.print_report()

    # reference comparison demo
    x_ref = (1.0 * np.sin(2 * np.pi * 50 * t)
             + 0.15 * np.sin(2 * np.pi * 100 * t)
             + 0.2 * np.random.randn(len(t)))

    fa2 = FourierAnalyst(x, fs=fs, name="current_vs_baseline")
    fa2.phase_spectral_profile()
    fa2.phase_spectral_anomaly(x_ref)
    for p in fa2._report.phases:
        print(f"\n--- {p.name} [{p.verdict.value}] ---")
        for f in p.findings:
            print(str(f))

    # convenience functions demo
    print("\n--- Spectral Distance ---")
    dist = spectral_distance(x, x_ref, fs=fs)
    print(f"  Distance: {dist:.2f} dB")

    print("\n--- Band Energy Profile ---")
    bands = band_energy_profile(x, fs=fs, n_bands=5)
    for b in bands:
        print(f"  {b['band']:>15s}: {b['pct']:.1f}%")

    print("\n--- Multi-Signal Comparison ---")
    comparison = compare_spectra({"current": x, "baseline": x_ref}, fs=fs)
    for name, metrics in comparison.items():
        print(f"  {name}: centroid={metrics['spectral_centroid_hz']:.1f} Hz, "
              f"flatness={metrics['spectral_flatness']:.4f}")

    print()


# ===========================================================================
# 6. CLI INTERFACE
# ===========================================================================


def main():
    import argparse
    import csv
    import json
    import sys

    try:
        from common import save_json as _common_save_json
    except ImportError:
        _common_save_json = None

    def _save_output(data: dict, path: str):
        """Save JSON output using common.py if available, else plain json."""
        if _common_save_json is not None:
            _common_save_json(path, data)
        else:
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        print(f"[fourier_analyst] Saved -> {path}")

    def _read_csv_column(filepath: str, column: str) -> List[Optional[float]]:
        """Read a column from CSV, converting to floats."""
        try:
            with open(filepath, newline="") as f:
                reader = csv.DictReader(f)
                if column not in reader.fieldnames:
                    print(f"Error: column '{column}' not found. "
                          f"Available: {', '.join(reader.fieldnames)}")
                    sys.exit(1)
                raw = [row[column] for row in reader]
        except FileNotFoundError:
            print(f"Error: file not found: {filepath}")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading CSV: {e}")
            sys.exit(1)

        data = []
        for v in raw:
            try:
                data.append(float(v))
            except (TypeError, ValueError):
                data.append(None)
        return data

    parser = argparse.ArgumentParser(
        description="Fourier Analyst — frequency-domain system analysis"
    )
    subparsers = parser.add_subparsers(dest="cmd", help="Commands")

    # --- analyze: full 9-phase analysis from CSV ---
    analyze_p = subparsers.add_parser("analyze", help="Full 9-phase spectral analysis from CSV")
    analyze_p.add_argument("file", help="CSV file path")
    analyze_p.add_argument("--column", required=True, help="Column name to analyze")
    analyze_p.add_argument("--fs", type=float, default=1.0, help="Sample rate in Hz (default: 1.0)")
    analyze_p.add_argument("--fundamental", type=float, default=None,
                           help="Known fundamental frequency (Hz) for harmonic analysis")
    analyze_p.add_argument("--shaft-rpm", type=float, default=None,
                           help="Shaft speed in RPM for vibration diagnostics")
    analyze_p.add_argument("--output", default=None, help="Save JSON report to file")
    analyze_p.add_argument("--verbose", action="store_true", help="Show all findings (default)")

    # --- quick: phases 1-5 only ---
    quick_p = subparsers.add_parser("quick", help="Quick review (phases 1-5 only)")
    quick_p.add_argument("file", help="CSV file path")
    quick_p.add_argument("--column", required=True, help="Column name to analyze")
    quick_p.add_argument("--fs", type=float, default=1.0, help="Sample rate in Hz (default: 1.0)")
    quick_p.add_argument("--output", default=None, help="Save JSON report to file")

    # --- compare: multi-signal comparison ---
    compare_p = subparsers.add_parser("compare", help="Compare multiple signal columns")
    compare_p.add_argument("file", help="CSV file path")
    compare_p.add_argument("--columns", required=True,
                           help="Comma-separated column names to compare")
    compare_p.add_argument("--fs", type=float, default=1.0, help="Sample rate in Hz (default: 1.0)")
    compare_p.add_argument("--output", default=None, help="Save JSON comparison to file")

    # --- demo ---
    subparsers.add_parser("demo", help="Run built-in synthetic data demo")

    args = parser.parse_args()

    if args.cmd == "demo":
        _demo()

    elif args.cmd == "analyze":
        raw_data = _read_csv_column(args.file, args.column)
        # filter out None values for numpy
        data = [v for v in raw_data if v is not None]
        if len(data) < 8:
            print(f"Error: only {len(data)} valid numeric values — need at least 8")
            sys.exit(1)

        fa = FourierAnalyst(data, fs=args.fs, name=args.column)
        report = fa.full_analysis(
            fundamental=args.fundamental,
            shaft_rpm=args.shaft_rpm,
        )
        report.print_report(verbose=True)

        if args.output:
            _save_output(report.to_dict(), args.output)

    elif args.cmd == "quick":
        raw_data = _read_csv_column(args.file, args.column)
        data = [v for v in raw_data if v is not None]
        if len(data) < 8:
            print(f"Error: only {len(data)} valid numeric values — need at least 8")
            sys.exit(1)

        fa = FourierAnalyst(data, fs=args.fs, name=args.column)

        # Run phases 1-5 only
        fa.phase_spectral_profile()
        if fa._report.phases[-1].verdict != Verdict.REJECT:
            fa.phase_harmonic_analysis()
            fa.phase_windowing_quality()
            fa.phase_noise_floor()
            fa.phase_bandwidth_analysis()

        fa._report.metadata.update({
            "duration_s": round(fa.n / fa.fs, 6),
            "nyquist_hz": fa.fs / 2,
            "freq_resolution_hz": round(fa.fs / fa.n, 6),
            "has_scipy": _HAS_SCIPY,
            "mode": "quick",
        })

        fa._report.print_report(verbose=True)

        if args.output:
            _save_output(fa._report.to_dict(), args.output)

    elif args.cmd == "compare":
        columns = [c.strip() for c in args.columns.split(",")]
        signals = {}
        for col in columns:
            raw_data = _read_csv_column(args.file, col)
            data = [v for v in raw_data if v is not None]
            if len(data) < 8:
                print(f"Warning: column '{col}' has only {len(data)} valid values — skipping")
                continue
            signals[col] = np.array(data, dtype=np.float64)

        if not signals:
            print("Error: no valid columns to compare")
            sys.exit(1)

        comparison = compare_spectra(signals, fs=args.fs)

        w = 72
        print(f"\n{'=' * w}")
        print(f"  SPECTRAL COMPARISON ({len(signals)} signals, Fs={args.fs} Hz)")
        print(f"{'=' * w}")
        header = f"{'Signal':>20s} | {'Centroid Hz':>12s} | {'Rolloff95 Hz':>12s} | {'Flatness':>10s} | {'Peak Hz':>10s} | {'Peak Amp':>10s}"
        print(header)
        print("-" * len(header))
        for name, m in comparison.items():
            print(f"{name:>20s} | {m['spectral_centroid_hz']:>12.2f} | "
                  f"{m['spectral_rolloff_95_hz']:>12.2f} | {m['spectral_flatness']:>10.4f} | "
                  f"{m['peak_freq_hz']:>10.2f} | {m['peak_amplitude']:>10.6f}")
        print()

        if args.output:
            _save_output(comparison, args.output)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
