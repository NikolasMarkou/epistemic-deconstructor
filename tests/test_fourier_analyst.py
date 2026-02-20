#!/usr/bin/env python3
"""Tests for src/scripts/fourier_analyst.py"""

import math
import os
import sys
import unittest

# Allow importing from src/scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'scripts'))

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

if HAS_NUMPY:
    from fourier_analyst import (
        FourierAnalyst,
        AnalysisReport,
        PhaseResult,
        Finding,
        Verdict,
        Severity,
        _safe,
        _next_pow2,
        _apply_window,
        _compute_psd,
        _amplitude_spectrum,
        _find_dominant_freqs,
        _spectral_centroid,
        _spectral_rolloff,
        _spectral_flatness,
        _band_energy,
        quick_spectrum,
        compare_spectra,
        spectral_distance,
        band_energy_profile,
    )


# ---------------------------------------------------------------------------
# Helper: generate test signals
# ---------------------------------------------------------------------------

def _sine(freq, fs=1000.0, duration=1.0, amplitude=1.0):
    """Generate a pure sine wave."""
    t = np.arange(0, duration, 1.0 / fs)
    return amplitude * np.sin(2 * np.pi * freq * t)


def _sine_with_harmonics(fundamental, fs=1000.0, duration=1.0):
    """Generate fundamental + 2nd and 3rd harmonics."""
    t = np.arange(0, duration, 1.0 / fs)
    return (1.0 * np.sin(2 * np.pi * fundamental * t)
            + 0.3 * np.sin(2 * np.pi * 2 * fundamental * t)
            + 0.1 * np.sin(2 * np.pi * 3 * fundamental * t))


# ===========================================================================
# HELPER FUNCTION TESTS
# ===========================================================================


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class TestNextPow2(unittest.TestCase):

    def test_power_of_2(self):
        self.assertEqual(_next_pow2(256), 256)

    def test_non_power(self):
        self.assertEqual(_next_pow2(200), 256)

    def test_one(self):
        self.assertEqual(_next_pow2(1), 1)

    def test_small(self):
        self.assertEqual(_next_pow2(3), 4)


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class TestApplyWindow(unittest.TestCase):

    def test_returns_same_length(self):
        x = np.ones(100)
        windowed, cg = _apply_window(x, "hann")
        self.assertEqual(len(windowed), 100)

    def test_coherent_gain_positive(self):
        x = np.ones(100)
        _, cg = _apply_window(x, "hann")
        self.assertGreater(cg, 0)

    def test_hann_tapers_edges(self):
        x = np.ones(100)
        windowed, _ = _apply_window(x, "hann")
        self.assertAlmostEqual(windowed[0], 0.0, places=5)
        # last sample of symmetric Hann window is near-zero but not exactly zero
        self.assertLess(abs(windowed[-1]), 0.01)


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class TestComputePSD(unittest.TestCase):

    def test_returns_frequencies_and_psd(self):
        x = _sine(50, fs=1000, duration=0.5)
        f, psd = _compute_psd(x, fs=1000)
        self.assertEqual(len(f), len(psd))
        self.assertGreater(len(f), 0)

    def test_psd_nonnegative(self):
        x = _sine(100, fs=1000, duration=0.5)
        _, psd = _compute_psd(x, fs=1000)
        self.assertTrue(np.all(psd >= 0))

    def test_peak_near_signal_frequency(self):
        x = _sine(100, fs=1000, duration=1.0)
        f, psd = _compute_psd(x, fs=1000)
        peak_freq = f[np.argmax(psd)]
        self.assertAlmostEqual(peak_freq, 100.0, delta=5.0)


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class TestAmplitudeSpectrum(unittest.TestCase):

    def test_returns_correct_shape(self):
        x = _sine(50, fs=1000, duration=0.5)
        freqs, mag = _amplitude_spectrum(x, fs=1000)
        self.assertEqual(len(freqs), len(mag))

    def test_peak_at_signal_frequency(self):
        x = _sine(100, fs=1000, duration=1.0)
        freqs, mag = _amplitude_spectrum(x, fs=1000)
        peak_freq = freqs[np.argmax(mag[1:]) + 1]  # skip DC
        self.assertAlmostEqual(peak_freq, 100.0, delta=2.0)

    def test_amplitude_approximately_correct(self):
        x = _sine(50, fs=1000, duration=1.0, amplitude=2.0)
        freqs, mag = _amplitude_spectrum(x, fs=1000)
        peak_amp = np.max(mag[1:])
        self.assertAlmostEqual(peak_amp, 2.0, delta=0.3)


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class TestFindDominantFreqs(unittest.TestCase):

    def test_finds_single_tone(self):
        x = _sine(200, fs=1000, duration=1.0)
        freqs, mag = _amplitude_spectrum(x, fs=1000)
        peaks = _find_dominant_freqs(freqs, mag, n_peaks=5)
        self.assertGreater(len(peaks), 0)
        # strongest peak near 200 Hz
        self.assertAlmostEqual(peaks[0][0], 200.0, delta=3.0)

    def test_finds_multiple_tones(self):
        x = _sine_with_harmonics(50, fs=1000, duration=1.0)
        freqs, mag = _amplitude_spectrum(x, fs=1000)
        peaks = _find_dominant_freqs(freqs, mag, n_peaks=10)
        peak_freqs = [f for f, _ in peaks]
        # should find near 50, 100, 150
        found_50 = any(abs(f - 50) < 5 for f in peak_freqs)
        found_100 = any(abs(f - 100) < 5 for f in peak_freqs)
        self.assertTrue(found_50)
        self.assertTrue(found_100)

    def test_respects_n_peaks_limit(self):
        x = _sine_with_harmonics(50, fs=1000, duration=1.0)
        freqs, mag = _amplitude_spectrum(x, fs=1000)
        peaks = _find_dominant_freqs(freqs, mag, n_peaks=2)
        self.assertLessEqual(len(peaks), 2)


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class TestSpectralCentroid(unittest.TestCase):

    def test_single_tone(self):
        # centroid of a single tone should be near that frequency
        x = _sine(100, fs=1000, duration=1.0)
        freqs, mag = _amplitude_spectrum(x, fs=1000)
        centroid = _spectral_centroid(freqs, mag)
        self.assertAlmostEqual(centroid, 100.0, delta=20.0)

    def test_zero_magnitude(self):
        freqs = np.array([0, 1, 2, 3])
        mag = np.zeros(4)
        self.assertEqual(_spectral_centroid(freqs, mag), 0.0)


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class TestSpectralRolloff(unittest.TestCase):

    def test_low_freq_signal(self):
        x = _sine(20, fs=1000, duration=1.0)
        freqs, mag = _amplitude_spectrum(x, fs=1000)
        rolloff = _spectral_rolloff(freqs, mag, threshold=0.95)
        # most energy near 20 Hz, rolloff should be low
        self.assertLess(rolloff, 100.0)

    def test_zero_magnitude(self):
        freqs = np.array([0, 1, 2, 3])
        mag = np.zeros(4)
        self.assertEqual(_spectral_rolloff(freqs, mag), 0.0)


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class TestSpectralFlatness(unittest.TestCase):

    def test_white_noise_near_one(self):
        np.random.seed(42)
        x = np.random.randn(10000)
        _, psd = _compute_psd(x, fs=1000)
        flatness = _spectral_flatness(psd)
        self.assertGreater(flatness, 0.5)

    def test_pure_tone_near_zero(self):
        x = _sine(100, fs=1000, duration=1.0)
        _, psd = _compute_psd(x, fs=1000)
        flatness = _spectral_flatness(psd)
        self.assertLess(flatness, 0.3)

    def test_empty_returns_zero(self):
        self.assertEqual(_spectral_flatness(np.array([])), 0.0)

    def test_all_zeros(self):
        self.assertEqual(_spectral_flatness(np.zeros(10)), 0.0)


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class TestBandEnergy(unittest.TestCase):

    def test_total_energy_positive(self):
        x = _sine(100, fs=1000, duration=1.0)
        f, psd = _compute_psd(x, fs=1000)
        e = _band_energy(f, psd, 0, 500)
        self.assertGreater(e, 0)

    def test_energy_concentrated_near_signal(self):
        x = _sine(100, fs=1000, duration=1.0)
        f, psd = _compute_psd(x, fs=1000)
        e_signal = _band_energy(f, psd, 80, 120)
        e_away = _band_energy(f, psd, 200, 300)
        self.assertGreater(e_signal, e_away)

    def test_empty_band_returns_zero(self):
        f = np.array([0, 10, 20, 30])
        psd = np.array([1, 1, 1, 1])
        self.assertEqual(_band_energy(f, psd, 100, 200), 0.0)


# ===========================================================================
# SAFE SERIALIZER TESTS
# ===========================================================================


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class TestSafe(unittest.TestCase):

    def test_primitives(self):
        self.assertEqual(_safe(42), 42)
        self.assertEqual(_safe(3.14), 3.14)
        self.assertEqual(_safe("hello"), "hello")
        self.assertIsNone(_safe(None))
        self.assertTrue(_safe(True))

    def test_numpy_array(self):
        result = _safe(np.array([1, 2, 3]))
        self.assertEqual(result, [1, 2, 3])

    def test_numpy_scalar(self):
        result = _safe(np.float64(3.14))
        self.assertAlmostEqual(result, 3.14, places=5)

    def test_nested_dict(self):
        result = _safe({"a": np.array([1, 2]), "b": np.float64(3)})
        self.assertEqual(result, {"a": [1, 2], "b": 3.0})

    def test_inf_passthrough(self):
        # float("inf") is a valid float, _safe passes it through
        result = _safe(float("inf"))
        self.assertEqual(result, float("inf"))


# ===========================================================================
# DATA STRUCTURE TESTS
# ===========================================================================


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class TestFinding(unittest.TestCase):

    def test_str_format(self):
        f = Finding("Phase 1", "check", Verdict.PASS, Severity.INFO, "detail")
        s = str(f)
        self.assertIn("[OK]", s)
        self.assertIn("INFO", s)
        self.assertIn("detail", s)

    def test_warn_icon(self):
        f = Finding("Phase 1", "check", Verdict.WARN, Severity.HIGH, "detail")
        self.assertIn("[!!]", str(f))


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class TestPhaseResult(unittest.TestCase):

    def test_verdict_pass(self):
        ph = PhaseResult("test")
        ph.add("c1", Verdict.PASS, Severity.INFO, "ok")
        self.assertEqual(ph.verdict, Verdict.PASS)

    def test_verdict_warn(self):
        ph = PhaseResult("test")
        ph.add("c1", Verdict.PASS, Severity.INFO, "ok")
        ph.add("c2", Verdict.WARN, Severity.MEDIUM, "warning")
        self.assertEqual(ph.verdict, Verdict.WARN)

    def test_verdict_reject_overrides(self):
        ph = PhaseResult("test")
        ph.add("c1", Verdict.WARN, Severity.HIGH, "warn")
        ph.add("c2", Verdict.REJECT, Severity.CRITICAL, "reject")
        self.assertEqual(ph.verdict, Verdict.REJECT)

    def test_verdict_all_skip(self):
        ph = PhaseResult("test")
        ph.add("c1", Verdict.SKIP, Severity.INFO, "skip")
        self.assertEqual(ph.verdict, Verdict.SKIP)

    def test_empty_is_skip(self):
        # no findings -> all(skip) branch -> SKIP
        ph = PhaseResult("test")
        self.assertEqual(ph.verdict, Verdict.SKIP)


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class TestAnalysisReport(unittest.TestCase):

    def test_overall_verdict_pass(self):
        r = AnalysisReport("sig", 100, 1000.0)
        ph = PhaseResult("p1")
        ph.add("c1", Verdict.PASS, Severity.INFO, "ok")
        r.phases.append(ph)
        self.assertEqual(r.overall_verdict, Verdict.PASS)

    def test_red_flags(self):
        r = AnalysisReport("sig", 100, 1000.0)
        ph = PhaseResult("p1")
        ph.add("c1", Verdict.FAIL, Severity.HIGH, "bad")
        ph.add("c2", Verdict.PASS, Severity.INFO, "ok")
        r.phases.append(ph)
        self.assertEqual(len(r.red_flags), 1)
        self.assertEqual(len(r.warnings), 0)

    def test_to_dict_structure(self):
        r = AnalysisReport("sig", 100, 1000.0)
        ph = PhaseResult("p1")
        ph.add("c1", Verdict.PASS, Severity.INFO, "ok", 42)
        r.phases.append(ph)
        d = r.to_dict()
        self.assertEqual(d["signal_name"], "sig")
        self.assertEqual(d["n_samples"], 100)
        self.assertEqual(d["sample_rate"], 1000.0)
        self.assertEqual(d["overall_verdict"], "PASS")
        self.assertEqual(len(d["phases"]), 1)
        self.assertEqual(d["phases"][0]["findings"][0]["value"], 42)


# ===========================================================================
# FOURIER ANALYST PHASE TESTS
# ===========================================================================


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class TestConstructor(unittest.TestCase):

    def test_basic(self):
        fa = FourierAnalyst([1, 2, 3, 4] * 10, fs=100.0, name="test")
        self.assertEqual(fa.n, 40)
        self.assertEqual(fa.fs, 100.0)
        self.assertEqual(fa.name, "test")

    def test_defaults(self):
        fa = FourierAnalyst([1, 2, 3, 4] * 10)
        self.assertEqual(fa.fs, 1.0)
        self.assertEqual(fa.name, "unknown_signal")

    def test_numpy_input(self):
        fa = FourierAnalyst(np.zeros(100), fs=44100.0)
        self.assertEqual(fa.n, 100)


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class TestPhaseSpectralProfile(unittest.TestCase):

    def test_rejects_short_signal(self):
        fa = FourierAnalyst([1, 2, 3], fs=100.0)
        ph = fa.phase_spectral_profile()
        self.assertEqual(ph.verdict, Verdict.REJECT)

    def test_pass_on_clean_signal(self):
        x = _sine(50, fs=1000, duration=0.5)
        fa = FourierAnalyst(x, fs=1000.0)
        ph = fa.phase_spectral_profile()
        self.assertEqual(ph.verdict, Verdict.PASS)

    def test_finds_dominant_frequency(self):
        x = _sine(100, fs=1000, duration=1.0)
        fa = FourierAnalyst(x, fs=1000.0)
        ph = fa.phase_spectral_profile()
        # find the dominant_frequencies finding
        dom_f = [f for f in ph.findings if f.check == "dominant_frequencies"]
        self.assertEqual(len(dom_f), 1)
        self.assertEqual(dom_f[0].verdict, Verdict.PASS)

    def test_spectral_flatness_reported(self):
        x = _sine(50, fs=1000, duration=0.5)
        fa = FourierAnalyst(x, fs=1000.0)
        ph = fa.phase_spectral_profile()
        flat_f = [f for f in ph.findings if f.check == "spectral_flatness"]
        self.assertEqual(len(flat_f), 1)

    def test_dc_offset_detection(self):
        # signal with massive DC offset relative to AC content
        x = np.ones(1000) * 100 + 0.001 * np.sin(2 * np.pi * 50 * np.arange(1000) / 1000)
        fa = FourierAnalyst(x, fs=1000.0)
        ph = fa.phase_spectral_profile()
        dc_f = [f for f in ph.findings if f.check == "dc_offset"]
        self.assertEqual(len(dc_f), 1)


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class TestPhaseHarmonicAnalysis(unittest.TestCase):

    def test_finds_harmonics(self):
        x = _sine_with_harmonics(50, fs=1000, duration=1.0)
        fa = FourierAnalyst(x, fs=1000.0)
        fa.phase_spectral_profile()
        ph = fa.phase_harmonic_analysis()
        harmonics_f = [f for f in ph.findings if f.check == "harmonics_detected"]
        self.assertEqual(len(harmonics_f), 1)

    def test_thd_computed(self):
        x = _sine_with_harmonics(50, fs=1000, duration=1.0)
        fa = FourierAnalyst(x, fs=1000.0)
        fa.phase_spectral_profile()
        ph = fa.phase_harmonic_analysis()
        thd_f = [f for f in ph.findings if f.check == "thd"]
        self.assertEqual(len(thd_f), 1)

    def test_explicit_fundamental(self):
        x = _sine_with_harmonics(50, fs=1000, duration=1.0)
        fa = FourierAnalyst(x, fs=1000.0)
        fa.phase_spectral_profile()
        ph = fa.phase_harmonic_analysis(fundamental=50.0)
        fund_f = [f for f in ph.findings if f.check == "fundamental"]
        self.assertEqual(len(fund_f), 1)
        self.assertIn("50.0", fund_f[0].detail)

    def test_pure_sine_no_harmonics(self):
        x = _sine(100, fs=1000, duration=1.0)
        fa = FourierAnalyst(x, fs=1000.0)
        fa.phase_spectral_profile()
        ph = fa.phase_harmonic_analysis()
        # should report no significant harmonics or low THD
        self.assertIn(ph.verdict, (Verdict.PASS, Verdict.SKIP))

    def test_skip_on_no_peaks(self):
        # constant signal -> no spectral peaks
        fa = FourierAnalyst(np.zeros(256), fs=1000.0)
        fa.phase_spectral_profile()
        ph = fa.phase_harmonic_analysis()
        self.assertEqual(ph.verdict, Verdict.SKIP)


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class TestPhaseWindowingQuality(unittest.TestCase):

    def test_runs_on_normal_signal(self):
        x = _sine(50, fs=1000, duration=0.5)
        fa = FourierAnalyst(x, fs=1000.0)
        fa.phase_spectral_profile()
        ph = fa.phase_windowing_quality()
        self.assertIn(ph.verdict, (Verdict.PASS, Verdict.WARN))

    def test_skip_short_signal(self):
        fa = FourierAnalyst([1, 2, 3] * 5, fs=100.0)
        ph = fa.phase_windowing_quality()
        self.assertEqual(ph.verdict, Verdict.SKIP)

    def test_freq_resolution_reported(self):
        x = _sine(50, fs=1000, duration=1.0)
        fa = FourierAnalyst(x, fs=1000.0)
        fa.phase_spectral_profile()
        ph = fa.phase_windowing_quality()
        res_f = [f for f in ph.findings if f.check == "freq_resolution"]
        self.assertEqual(len(res_f), 1)

    def test_nyquist_warning_near_limit(self):
        # 490 Hz signal with 1000 Hz sampling -> near Nyquist
        x = _sine(490, fs=1000, duration=0.5)
        fa = FourierAnalyst(x, fs=1000.0)
        fa.phase_spectral_profile()
        ph = fa.phase_windowing_quality()
        nyq_f = [f for f in ph.findings if f.check == "nyquist_proximity"]
        self.assertEqual(len(nyq_f), 1)
        self.assertEqual(nyq_f[0].verdict, Verdict.WARN)


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class TestPhaseNoiseFloor(unittest.TestCase):

    def test_runs_on_normal_signal(self):
        x = _sine(50, fs=1000, duration=0.5)
        fa = FourierAnalyst(x, fs=1000.0)
        fa.phase_spectral_profile()
        ph = fa.phase_noise_floor()
        self.assertIn(ph.verdict, (Verdict.PASS, Verdict.WARN))

    def test_snr_reported(self):
        x = _sine(100, fs=1000, duration=1.0)
        fa = FourierAnalyst(x, fs=1000.0)
        fa.phase_spectral_profile()
        ph = fa.phase_noise_floor()
        snr_f = [f for f in ph.findings if f.check == "snr"]
        self.assertEqual(len(snr_f), 1)

    def test_noise_floor_reported(self):
        x = _sine(50, fs=1000, duration=0.5)
        fa = FourierAnalyst(x, fs=1000.0)
        fa.phase_spectral_profile()
        ph = fa.phase_noise_floor()
        nf = [f for f in ph.findings if f.check == "noise_floor_level"]
        self.assertEqual(len(nf), 1)


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class TestPhaseBandwidthAnalysis(unittest.TestCase):

    def test_runs_on_normal_signal(self):
        x = _sine(50, fs=1000, duration=0.5)
        fa = FourierAnalyst(x, fs=1000.0)
        fa.phase_spectral_profile()
        ph = fa.phase_bandwidth_analysis()
        # may WARN about oversampling (50 Hz in 500 Hz Nyquist)
        self.assertIn(ph.verdict, (Verdict.PASS, Verdict.WARN))

    def test_rolloff_reported(self):
        x = _sine(50, fs=1000, duration=0.5)
        fa = FourierAnalyst(x, fs=1000.0)
        fa.phase_spectral_profile()
        ph = fa.phase_bandwidth_analysis()
        ro_f = [f for f in ph.findings if f.check == "spectral_rolloff"]
        self.assertEqual(len(ro_f), 1)

    def test_band_energy_reported(self):
        x = _sine(50, fs=1000, duration=0.5)
        fa = FourierAnalyst(x, fs=1000.0)
        fa.phase_spectral_profile()
        ph = fa.phase_bandwidth_analysis()
        be_f = [f for f in ph.findings if f.check == "band_energy_distribution"]
        self.assertEqual(len(be_f), 1)

    def test_oversampling_warning(self):
        # very low freq signal with high sample rate
        x = _sine(5, fs=10000, duration=1.0)
        fa = FourierAnalyst(x, fs=10000.0)
        fa.phase_spectral_profile()
        ph = fa.phase_bandwidth_analysis()
        os_f = [f for f in ph.findings if f.check == "oversampling"]
        self.assertEqual(len(os_f), 1)
        self.assertEqual(os_f[0].verdict, Verdict.WARN)


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class TestPhaseSpectralAnomaly(unittest.TestCase):

    def test_identical_signals_pass(self):
        x = _sine(100, fs=1000, duration=1.0)
        fa = FourierAnalyst(x, fs=1000.0)
        fa.phase_spectral_profile()
        ph = fa.phase_spectral_anomaly(x)
        dist_f = [f for f in ph.findings if f.check == "spectral_distance"]
        self.assertEqual(dist_f[0].verdict, Verdict.PASS)

    def test_very_different_signals_fail(self):
        x = _sine(50, fs=1000, duration=1.0)
        ref = _sine(400, fs=1000, duration=1.0)
        fa = FourierAnalyst(x, fs=1000.0)
        fa.phase_spectral_profile()
        ph = fa.phase_spectral_anomaly(ref)
        dist_f = [f for f in ph.findings if f.check == "spectral_distance"]
        self.assertIn(dist_f[0].verdict, (Verdict.WARN, Verdict.FAIL))

    def test_new_peaks_detected(self):
        np.random.seed(42)
        ref = _sine(100, fs=1000, duration=1.0) + 0.01 * np.random.randn(1000)
        # current has an extra tone
        cur = ref + 0.5 * _sine(300, fs=1000, duration=1.0)
        fa = FourierAnalyst(cur, fs=1000.0)
        fa.phase_spectral_profile()
        ph = fa.phase_spectral_anomaly(ref)
        new_f = [f for f in ph.findings if f.check == "new_peaks"]
        self.assertEqual(len(new_f), 1)


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class TestPhaseSystemHealth(unittest.TestCase):

    def test_basic_health_metrics(self):
        x = _sine(50, fs=1000, duration=1.0)
        fa = FourierAnalyst(x, fs=1000.0)
        fa.phase_spectral_profile()
        ph = fa.phase_system_health()
        checks = {f.check for f in ph.findings}
        self.assertIn("rms_level", checks)
        self.assertIn("crest_factor", checks)
        self.assertIn("kurtosis", checks)

    def test_shaft_harmonics(self):
        x = _sine(50, fs=1000, duration=1.0)
        fa = FourierAnalyst(x, fs=1000.0)
        fa.phase_spectral_profile()
        ph = fa.phase_system_health(shaft_rpm=3000)
        shaft_f = [f for f in ph.findings if f.check == "shaft_harmonics"]
        self.assertEqual(len(shaft_f), 1)
        self.assertIn("3000", shaft_f[0].detail)

    def test_bearing_fault_monitoring(self):
        # signal with a component at the fault frequency
        np.random.seed(42)
        x = (_sine(50, fs=1000, duration=1.0)
             + 0.5 * _sine(237, fs=1000, duration=1.0)
             + 0.01 * np.random.randn(1000))
        fa = FourierAnalyst(x, fs=1000.0)
        fa.phase_spectral_profile()
        ph = fa.phase_system_health(bearing_bpfo=237.0)
        bpfo_f = [f for f in ph.findings if f.check == "bearing_BPFO"]
        self.assertEqual(len(bpfo_f), 1)

    def test_impulsive_signal_high_crest(self):
        np.random.seed(42)
        x = np.random.randn(1000) * 0.1
        x[500] = 10.0  # single spike
        fa = FourierAnalyst(x, fs=1000.0)
        fa.phase_spectral_profile()
        ph = fa.phase_system_health()
        crest_f = [f for f in ph.findings if f.check == "crest_factor"]
        self.assertEqual(len(crest_f), 1)
        self.assertEqual(crest_f[0].verdict, Verdict.WARN)

    def test_spectral_fingerprint(self):
        x = _sine_with_harmonics(50, fs=1000, duration=1.0)
        fa = FourierAnalyst(x, fs=1000.0)
        fa.phase_spectral_profile()
        ph = fa.phase_system_health()
        fp_f = [f for f in ph.findings if f.check == "spectral_fingerprint"]
        self.assertEqual(len(fp_f), 1)


# ===========================================================================
# FULL ANALYSIS & CONVENIENCE FUNCTION TESTS
# ===========================================================================


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class TestFullAnalysis(unittest.TestCase):

    def test_returns_report(self):
        x = _sine(50, fs=1000, duration=0.5)
        fa = FourierAnalyst(x, fs=1000.0, name="test_sig")
        report = fa.full_analysis()
        self.assertIsInstance(report, AnalysisReport)
        self.assertEqual(report.signal_name, "test_sig")
        self.assertGreater(len(report.phases), 0)

    def test_short_circuit_on_reject(self):
        fa = FourierAnalyst([1, 2, 3], fs=100.0)
        report = fa.full_analysis()
        self.assertEqual(report.overall_verdict, Verdict.REJECT)
        self.assertEqual(len(report.phases), 1)

    def test_metadata_populated(self):
        x = _sine(50, fs=1000, duration=0.5)
        fa = FourierAnalyst(x, fs=1000.0)
        report = fa.full_analysis()
        self.assertIn("duration_s", report.metadata)
        self.assertIn("nyquist_hz", report.metadata)
        self.assertIn("rms", report.metadata)

    def test_with_reference_signal(self):
        x = _sine(50, fs=1000, duration=0.5)
        ref = _sine(50, fs=1000, duration=0.5)
        fa = FourierAnalyst(x, fs=1000.0)
        report = fa.full_analysis(reference_signal=ref)
        phase_names = [p.name for p in report.phases]
        self.assertTrue(any("Anomaly" in n for n in phase_names))

    def test_with_shaft_rpm(self):
        x = _sine(50, fs=1000, duration=0.5)
        fa = FourierAnalyst(x, fs=1000.0)
        report = fa.full_analysis(shaft_rpm=3000)
        phase_names = [p.name for p in report.phases]
        self.assertTrue(any("Health" in n for n in phase_names))

    def test_to_dict_serializable(self):
        """Verify full report can be JSON-serialized."""
        import json
        x = _sine_with_harmonics(50, fs=1000, duration=0.5)
        fa = FourierAnalyst(x, fs=1000.0)
        report = fa.full_analysis()
        d = report.to_dict()
        # should not raise
        json.dumps(d)


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class TestQuickSpectrum(unittest.TestCase):

    def test_returns_report(self):
        x = _sine(100, fs=1000, duration=0.5)
        report = quick_spectrum(x, fs=1000.0, name="quick_test")
        self.assertIsInstance(report, AnalysisReport)
        self.assertEqual(report.signal_name, "quick_test")


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class TestCompareSpectra(unittest.TestCase):

    def test_returns_dict_with_metrics(self):
        x1 = _sine(50, fs=1000, duration=0.5)
        x2 = _sine(100, fs=1000, duration=0.5)
        result = compare_spectra({"sig1": x1, "sig2": x2}, fs=1000.0)
        self.assertIn("sig1", result)
        self.assertIn("sig2", result)
        for key in ("rms", "spectral_centroid_hz", "spectral_rolloff_95_hz",
                     "spectral_flatness", "peak_freq_hz", "peak_amplitude"):
            self.assertIn(key, result["sig1"])

    def test_different_centroids(self):
        x1 = _sine(50, fs=1000, duration=0.5)
        x2 = _sine(200, fs=1000, duration=0.5)
        result = compare_spectra({"low": x1, "high": x2}, fs=1000.0)
        self.assertLess(result["low"]["spectral_centroid_hz"],
                        result["high"]["spectral_centroid_hz"])


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class TestSpectralDistance(unittest.TestCase):

    def test_identical_is_zero(self):
        x = _sine(100, fs=1000, duration=0.5)
        d = spectral_distance(x, x, fs=1000.0)
        self.assertAlmostEqual(d, 0.0, places=1)

    def test_different_signals_positive(self):
        x1 = _sine(50, fs=1000, duration=0.5)
        x2 = _sine(300, fs=1000, duration=0.5)
        d = spectral_distance(x1, x2, fs=1000.0)
        self.assertGreater(d, 0)

    def test_symmetric(self):
        x1 = _sine(50, fs=1000, duration=0.5)
        x2 = _sine(100, fs=1000, duration=0.5)
        self.assertAlmostEqual(spectral_distance(x1, x2, fs=1000.0),
                               spectral_distance(x2, x1, fs=1000.0), places=3)


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class TestBandEnergyProfile(unittest.TestCase):

    def test_returns_bands(self):
        x = _sine(100, fs=1000, duration=0.5)
        bands = band_energy_profile(x, fs=1000.0, n_bands=5)
        self.assertEqual(len(bands), 5)
        for b in bands:
            self.assertIn("band", b)
            self.assertIn("energy", b)
            self.assertIn("pct", b)

    def test_percentages_sum_to_100(self):
        x = _sine(100, fs=1000, duration=0.5)
        bands = band_energy_profile(x, fs=1000.0, n_bands=5)
        total_pct = sum(b["pct"] for b in bands)
        self.assertAlmostEqual(total_pct, 100.0, delta=1.0)

    def test_energy_concentrated_in_correct_band(self):
        x = _sine(50, fs=1000, duration=0.5)
        bands = band_energy_profile(x, fs=1000.0, n_bands=5)
        # 50 Hz is in the first band (0-100 Hz)
        self.assertGreater(bands[0]["pct"], 50.0)


# ===========================================================================
# CLI PARSER TEST
# ===========================================================================


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class TestCLIParser(unittest.TestCase):

    def test_parser_builds(self):
        from fourier_analyst import main
        # main() exists and is callable
        self.assertTrue(callable(main))

    def test_no_cmd_prints_help(self):
        """Calling with no args should not crash."""
        import io
        from contextlib import redirect_stdout, redirect_stderr
        import fourier_analyst
        import sys
        old_argv = sys.argv
        sys.argv = ["fourier_analyst.py"]
        try:
            out = io.StringIO()
            err = io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                fourier_analyst.main()
        except SystemExit:
            pass
        finally:
            sys.argv = old_argv


if __name__ == "__main__":
    unittest.main()
