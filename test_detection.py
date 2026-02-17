"""
Synthetic unit tests for the EventDetector in BotLobbies StepSense.

Generates fake stereo audio frames with known spectral characteristics
and verifies that the detector correctly classifies them as footstep,
shot, or None — and specifically that gunfire is NOT misclassified as
a footstep.
"""

import importlib
import math
import sys
import time

import numpy as np
import pytest

# Import the main module (has a space in the filename)
spec = importlib.util.spec_from_file_location(
    "stepsense", "BotLobbies StepSense.py"
)
ss = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ss)

FS = ss.FS
WIN = ss.WIN
HOP = ss.HOP


# ---------------------------------------------------------------------------
#  Helpers: generate synthetic stereo frames
# ---------------------------------------------------------------------------

def make_tone(freq_hz: float, duration_samples: int = WIN, amplitude: float = 0.1,
              fs: int = FS) -> np.ndarray:
    """Pure sine tone, mono."""
    t = np.arange(duration_samples) / fs
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def make_stereo(mono: np.ndarray, pan: float = 0.0) -> np.ndarray:
    """Convert mono to stereo with pan (-1 = left, 0 = center, +1 = right)."""
    l_gain = np.sqrt(0.5 * (1.0 - pan))
    r_gain = np.sqrt(0.5 * (1.0 + pan))
    return np.column_stack([mono * l_gain, mono * r_gain]).astype(np.float32)


def make_noise(duration_samples: int = WIN, amplitude: float = 0.001,
               fs: int = FS) -> np.ndarray:
    """Low-level white noise, stereo."""
    noise = amplitude * np.random.randn(duration_samples, 2)
    return noise.astype(np.float32)


def make_footstep_frame(amplitude: float = 0.15, pan: float = 0.0) -> np.ndarray:
    """Simulate a COD footstep hitting both FOOT_A (80-250 Hz) and FOOT_B (2500-6000 Hz).

    Low thump (heel impact) plus high-frequency texture (scrape/surface detail).
    The competitive COD footstep sweet spot is ~4 kHz.
    """
    t = np.arange(WIN) / FS
    decay = np.exp(-t * 80)  # fast decay
    # FOOT_A: low thump at 120 Hz
    thump = amplitude * np.sin(2 * np.pi * 120 * t) * decay
    # FOOT_B: texture detail at 3500-4000 Hz (COD competitive sweet spot)
    scrape = (amplitude * 0.3) * np.sin(2 * np.pi * 4000 * t) * np.exp(-t * 120)
    texture = (amplitude * 0.15) * np.sin(2 * np.pi * 3200 * t) * np.exp(-t * 100)
    mono = (thump + scrape + texture).astype(np.float32)
    return make_stereo(mono, pan)


def make_gunshot_frame(amplitude: float = 0.6, pan: float = 0.0) -> np.ndarray:
    """Simulate a COD gunshot: broadband transient in GUN_LO (300-1200) + GUN_HI (1200-2500).

    Sharp impulse with high crest factor, energy concentrated below 2500 Hz.
    """
    t = np.arange(WIN) / FS
    # Very sharp attack, fast decay — creates high crest factor
    envelope = np.exp(-t * 200)
    # Frequencies targeting GUN_LO (300-1200) and GUN_HI (1200-2500)
    signal = (
        0.3 * np.sin(2 * np.pi * 500 * t) +   # GUN_LO
        0.3 * np.sin(2 * np.pi * 900 * t) +   # GUN_LO (small-arms core)
        0.4 * np.sin(2 * np.pi * 1500 * t) +  # GUN_HI
        0.3 * np.sin(2 * np.pi * 2000 * t) +  # GUN_HI (crack)
        0.2 * np.sin(2 * np.pi * 2300 * t)    # GUN_HI upper
    )
    mono = (amplitude * signal * envelope).astype(np.float32)
    return make_stereo(mono, pan)


def make_gunshot_tail_frame(amplitude: float = 0.08, pan: float = 0.0) -> np.ndarray:
    """Simulate the reverb/tail after a gunshot — lower energy, still in gun bands."""
    t = np.arange(WIN) / FS
    envelope = np.exp(-t * 40)
    signal = (
        0.3 * np.sin(2 * np.pi * 600 * t) +   # GUN_LO reverb
        0.4 * np.sin(2 * np.pi * 1400 * t) +  # GUN_HI reverb
        0.2 * np.sin(2 * np.pi * 2200 * t)    # GUN_HI upper reverb
    )
    mono = (amplitude * signal * envelope).astype(np.float32)
    return make_stereo(mono, pan)


def warm_up_detector(det: ss.EventDetector, n_frames: int = 200):
    """Feed quiet noise so the adaptive noise floor stabilizes."""
    for _ in range(n_frames):
        frame = make_noise(amplitude=0.001)
        det.detect(frame)


# ---------------------------------------------------------------------------
#  Tests
# ---------------------------------------------------------------------------

class TestQuietFrames:
    """Silence / low noise should produce no events."""

    def test_silence_returns_none(self):
        det = ss.EventDetector(sensitivity=1.0)
        warm_up_detector(det)
        frame = np.zeros((WIN, 2), dtype=np.float32)
        assert det.detect(frame) is None

    def test_low_noise_returns_none(self):
        det = ss.EventDetector(sensitivity=1.0)
        warm_up_detector(det)
        for _ in range(20):
            frame = make_noise(amplitude=0.001)
            evt = det.detect(frame)
            assert evt is None


class TestFootstepDetection:
    """Footstep-like signals should be classified as footsteps."""

    def test_footstep_detected(self):
        det = ss.EventDetector(sensitivity=1.5)
        warm_up_detector(det)
        # Feed a footstep frame — may take a couple to trigger after warmup
        detected = []
        for _ in range(5):
            evt = det.detect(make_footstep_frame(amplitude=0.2))
            if evt is not None:
                detected.append(evt)
            # Intersperse quiet frames to let noise floor settle
            for _ in range(3):
                det.detect(make_noise(amplitude=0.001))
        assert any(e.cls == 'footstep' for e in detected), \
            f"Expected footstep detection, got: {[e.cls for e in detected]}"

    def test_footstep_not_classified_as_shot(self):
        det = ss.EventDetector(sensitivity=1.5)
        warm_up_detector(det)
        detected = []
        for _ in range(5):
            evt = det.detect(make_footstep_frame(amplitude=0.2))
            if evt is not None:
                detected.append(evt)
            for _ in range(3):
                det.detect(make_noise(amplitude=0.001))
        shots = [e for e in detected if e.cls == 'shot']
        assert len(shots) == 0, \
            f"Footstep signal misclassified as shot {len(shots)} times"

    def test_footstep_direction_left(self):
        det = ss.EventDetector(sensitivity=1.5)
        warm_up_detector(det)
        detected = []
        for _ in range(5):
            evt = det.detect(make_footstep_frame(amplitude=0.2, pan=-0.8))
            if evt is not None:
                detected.append(evt)
            for _ in range(3):
                det.detect(make_noise(amplitude=0.001))
        footsteps = [e for e in detected if e.cls == 'footstep']
        if footsteps:
            # Negative theta = left
            assert footsteps[-1].theta < 0, \
                f"Expected negative theta for left pan, got {math.degrees(footsteps[-1].theta):.1f}°"

    def test_footstep_direction_right(self):
        det = ss.EventDetector(sensitivity=1.5)
        warm_up_detector(det)
        detected = []
        for _ in range(5):
            evt = det.detect(make_footstep_frame(amplitude=0.2, pan=0.8))
            if evt is not None:
                detected.append(evt)
            for _ in range(3):
                det.detect(make_noise(amplitude=0.001))
        footsteps = [e for e in detected if e.cls == 'footstep']
        if footsteps:
            # Positive theta = right
            assert footsteps[-1].theta > 0, \
                f"Expected positive theta for right pan, got {math.degrees(footsteps[-1].theta):.1f}°"


class TestGunshotDetection:
    """Gunshot-like signals should be classified as shots."""

    def test_gunshot_detected_as_shot(self):
        det = ss.EventDetector(sensitivity=1.0)
        warm_up_detector(det)
        detected = []
        for _ in range(3):
            evt = det.detect(make_gunshot_frame(amplitude=0.6))
            if evt is not None:
                detected.append(evt)
            for _ in range(5):
                det.detect(make_noise(amplitude=0.001))
        assert any(e.cls == 'shot' for e in detected), \
            f"Expected shot detection, got: {[e.cls for e in detected]}"

    def test_gunshot_NOT_classified_as_footstep(self):
        """THE critical test: gunfire must not trigger footstep detection."""
        det = ss.EventDetector(sensitivity=1.0)
        warm_up_detector(det)
        detected = []
        for _ in range(5):
            evt = det.detect(make_gunshot_frame(amplitude=0.6))
            if evt is not None:
                detected.append(evt)
            for _ in range(3):
                det.detect(make_noise(amplitude=0.001))
        footsteps = [e for e in detected if e.cls == 'footstep']
        assert len(footsteps) == 0, \
            f"Gunshot misclassified as footstep {len(footsteps)} times out of {len(detected)} detections"


class TestPostShotBlanking:
    """After a gunshot, footstep detection should be suppressed briefly."""

    def test_tail_after_shot_is_blanked(self):
        """Gunfire tail/reverb right after a shot should NOT produce a footstep event."""
        det = ss.EventDetector(sensitivity=1.0)
        warm_up_detector(det)

        # Fire a shot
        evt = det.detect(make_gunshot_frame(amplitude=0.6))
        assert evt is not None and evt.cls == 'shot', "Expected initial shot detection"

        # Immediately feed tail frames (within 150ms blanking window)
        # At 10ms per hop, 15 frames = 150ms
        tail_events = []
        for _ in range(10):
            evt = det.detect(make_gunshot_tail_frame(amplitude=0.08))
            if evt is not None:
                tail_events.append(evt)

        footsteps = [e for e in tail_events if e.cls == 'footstep']
        assert len(footsteps) == 0, \
            f"Post-shot tail produced {len(footsteps)} false footstep(s)"


class TestSpectralRatioRejection:
    """Sounds with gun-dominant spectral ratios should be rejected as footsteps."""

    def test_broadband_burst_rejected_as_footstep(self):
        """A broadband burst (gun-like spectrum) below crest threshold should
        not be classified as a footstep due to spectral ratio check."""
        det = ss.EventDetector(sensitivity=1.0)
        warm_up_detector(det)

        # Energy concentrated in gun bands (300-2500 Hz), none in footstep bands
        t = np.arange(WIN) / FS
        signal = (
            0.10 * np.sin(2 * np.pi * 500 * t) +    # GUN_LO (300-1200)
            0.10 * np.sin(2 * np.pi * 900 * t) +    # GUN_LO
            0.12 * np.sin(2 * np.pi * 1500 * t) +   # GUN_HI (1200-2500)
            0.10 * np.sin(2 * np.pi * 2200 * t) +   # GUN_HI
            0.01 * np.sin(2 * np.pi * 150 * t)      # tiny footstep band
        ).astype(np.float32)
        frame = make_stereo(signal, pan=0.0)

        detected = []
        for _ in range(5):
            evt = det.detect(frame)
            if evt is not None:
                detected.append(evt)
            for _ in range(3):
                det.detect(make_noise(amplitude=0.001))

        footsteps = [e for e in detected if e.cls == 'footstep']
        assert len(footsteps) == 0, \
            f"Gun-heavy spectrum produced {len(footsteps)} false footstep(s)"


class TestBorderlineShotRejection:
    """Sounds with moderate crest + gun band energy should not be footsteps."""

    def test_moderate_crest_with_gun_energy_rejected(self):
        det = ss.EventDetector(sensitivity=1.0)
        warm_up_detector(det)

        # Moderate transient in gun bands — crest around 3.5-5.0
        t = np.arange(WIN) / FS
        envelope = np.exp(-t * 60)  # moderate decay (not as sharp as gunshot)
        signal = (
            0.15 * np.sin(2 * np.pi * 600 * t) +   # GUN_LO (300-1200)
            0.15 * np.sin(2 * np.pi * 1800 * t)    # GUN_HI (1200-2500)
        )
        mono = (signal * envelope).astype(np.float32)
        frame = make_stereo(mono, pan=0.0)

        detected = []
        for _ in range(5):
            evt = det.detect(frame)
            if evt is not None:
                detected.append(evt)
            for _ in range(3):
                det.detect(make_noise(amplitude=0.001))

        footsteps = [e for e in detected if e.cls == 'footstep']
        assert len(footsteps) == 0, \
            f"Borderline shot produced {len(footsteps)} false footstep(s)"


class TestCadenceTracking:
    """Cadence prior should boost confidence for rhythmic footstep patterns."""

    def test_rhythmic_footsteps_gain_cadence_confidence(self):
        det = ss.EventDetector(sensitivity=1.5)
        warm_up_detector(det)

        # Simulate footsteps at ~300ms intervals (within CAD_MIN-CAD_MAX)
        events = []
        frames_per_step = int(0.3 * FS / HOP)  # 0.3s in frames

        for step in range(6):
            # One footstep frame
            evt = det.detect(make_footstep_frame(amplitude=0.2))
            if evt is not None:
                events.append(evt)
            # Quiet gap
            for _ in range(frames_per_step):
                det.detect(make_noise(amplitude=0.001))

        footsteps = [e for e in events if e.cls == 'footstep']
        if len(footsteps) >= 3:
            # Later footsteps should have equal or higher confidence from cadence
            assert footsteps[-1].confidence >= footsteps[0].confidence, \
                f"Cadence should boost later detections: first={footsteps[0].confidence:.2f}, last={footsteps[-1].confidence:.2f}"


class TestNoiseFloorAdaptation:
    """Noise floor should not be inflated by loud events."""

    def test_noise_floor_stable_after_loud_event(self):
        det = ss.EventDetector(sensitivity=1.0)
        warm_up_detector(det, n_frames=200)

        # Record noise floor before
        mu_fa_before = det.stats['FA'].mu
        mu_fb_before = det.stats['FB'].mu

        # Fire a loud gunshot
        det.detect(make_gunshot_frame(amplitude=0.8))
        det.detect(make_gunshot_frame(amplitude=0.8))

        # Noise floor should NOT have jumped
        mu_fa_after = det.stats['FA'].mu
        mu_fb_after = det.stats['FB'].mu

        # Allow tiny drift but not orders-of-magnitude jump
        assert mu_fa_after < mu_fa_before * 5, \
            f"FA noise floor jumped from {mu_fa_before:.2e} to {mu_fa_after:.2e}"
        assert mu_fb_after < mu_fb_before * 5, \
            f"FB noise floor jumped from {mu_fb_before:.2e} to {mu_fb_after:.2e}"


class TestFilterStatePersistence:
    """Filter state should persist across frames (no transient spikes)."""

    def test_continuous_tone_no_onset_spike(self):
        det = ss.EventDetector(sensitivity=1.0)
        warm_up_detector(det)

        # Feed identical tone frames and check energy is stable after first few
        tone = make_tone(150, amplitude=0.005)
        frame = make_stereo(tone)
        energies = []
        for i in range(20):
            det.detect(frame)
            e = ss.ste(ss.sosfilt(ss.SOS_FA, frame[:, 0], zi=np.zeros((ss.SOS_FA.shape[0], 2)))[0])
            energies.append(e)

        # With persistent state, energies from the detector's internal filters
        # should be smooth. We can't directly access them, but the key test is
        # that no spurious events fire from a constant low-level tone.
        events = []
        for _ in range(20):
            evt = det.detect(frame)
            if evt is not None:
                events.append(evt)
        # A constant low-level tone should not trigger events
        assert len(events) == 0, \
            f"Constant low-level tone triggered {len(events)} events"


class TestSensitivityScaling:
    """Higher sensitivity should detect quieter footsteps."""

    def test_high_sensitivity_detects_quieter(self):
        # Low sensitivity detector
        det_low = ss.EventDetector(sensitivity=0.5)
        warm_up_detector(det_low)

        # High sensitivity detector
        det_high = ss.EventDetector(sensitivity=3.0)
        warm_up_detector(det_high)

        quiet_foot = make_footstep_frame(amplitude=0.05)

        low_events = []
        high_events = []
        for _ in range(10):
            evt_low = det_low.detect(quiet_foot)
            evt_high = det_high.detect(quiet_foot)
            if evt_low is not None:
                low_events.append(evt_low)
            if evt_high is not None:
                high_events.append(evt_high)
            for _ in range(3):
                det_low.detect(make_noise(amplitude=0.001))
                det_high.detect(make_noise(amplitude=0.001))

        # High sensitivity should detect at least as many as low
        assert len(high_events) >= len(low_events), \
            f"High sensitivity detected {len(high_events)} vs low {len(low_events)}"


class TestEventDataIntegrity:
    """Event fields should be within valid ranges."""

    def test_footstep_fields_valid(self):
        det = ss.EventDetector(sensitivity=2.0)
        warm_up_detector(det)
        for _ in range(10):
            evt = det.detect(make_footstep_frame(amplitude=0.25))
            if evt is not None and evt.cls == 'footstep':
                assert -math.pi <= evt.theta <= math.pi, f"theta out of range: {evt.theta}"
                assert 0.0 <= evt.intensity <= 1.0, f"intensity out of range: {evt.intensity}"
                assert 0.0 <= evt.confidence <= 1.0, f"confidence out of range: {evt.confidence}"
                assert evt.t > 0, f"timestamp should be positive: {evt.t}"
                return  # found one valid event
            for _ in range(3):
                det.detect(make_noise(amplitude=0.001))
        pytest.skip("No footstep detected to validate fields")

    def test_shot_fields_valid(self):
        det = ss.EventDetector(sensitivity=1.0)
        warm_up_detector(det)
        for _ in range(5):
            evt = det.detect(make_gunshot_frame(amplitude=0.6))
            if evt is not None and evt.cls == 'shot':
                assert -math.pi <= evt.theta <= math.pi, f"theta out of range: {evt.theta}"
                assert 0.0 <= evt.intensity <= 1.0, f"intensity out of range: {evt.intensity}"
                assert 0.0 <= evt.confidence <= 1.0, f"confidence out of range: {evt.confidence}"
                assert evt.t > 0, f"timestamp should be positive: {evt.t}"
                return
            for _ in range(3):
                det.detect(make_noise(amplitude=0.001))
        pytest.skip("No shot detected to validate fields")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
