#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio→Visual Compass for COD/Warzone (stereo path) — No Haptics
- WASAPI loopback capture (sounddevice)
- Footstep detection (dual-band: 60–250 Hz + 1–4 kHz) with adaptive thresholds + cadence prior
- Gunshot detection (crest/decay guard)
- Direction via GCC-PHAT + ILD fusion
- Overlay: transparent radial ring (PySide6)

Install (Windows):
    pip install numpy scipy sounddevice PySide6
Usage:
    python compass.py --list-devices
    python compass.py --device <index>
"""

import argparse
import logging
import math
import queue
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import sounddevice as sd
from scipy.signal import butter, sosfilt

# Windows-specific audio capture
try:
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    from comtypes import CLSCTX_ALL, CoInitialize, CoUninitialize, GUID, IUnknown
    import comtypes.client
    import ctypes
    from ctypes import wintypes, windll, POINTER, Structure, c_uint32, c_void_p, c_long
    import pyaudio
    import pyaudiowpatch as pyaudio_wpatch
    WINDOWS_AUDIO_AVAILABLE = True
except ImportError:
    WINDOWS_AUDIO_AVAILABLE = False

# ---- Optional UI ----
try:
    from PySide6 import QtCore, QtGui, QtWidgets
    PYSIDE = True
except Exception:
    PYSIDE = False

# =========================
#     CONFIG DEFAULTS
# =========================
FS = 48000                # sample rate
HOP = 480                 # 10 ms hop
WIN = 960                 # 20 ms window
EPS = 1e-12
EAR_DIST = 0.18           # ~18 cm ear spacing
SPEED_SOUND = 343.0

# Bands (Warzone/COD tuned - more specific isolation)
# Footsteps: COD footsteps have energy primarily in low-mid frequencies
FOOT_LOW = (40, 180)      # heel impact, bass thump of steps
FOOT_MID = (250, 800)     # body of footstep sound
FOOT_HIGH = (2000, 4500)  # surface texture (gravel, metal, etc.)

# Gunfire: Much broader spectrum, higher energy, different characteristics
GUN_LOW = (80, 400)       # gunshot bass/thump
GUN_MID = (400, 2000)     # main gunfire body
GUN_HIGH = (2000, 8000)   # gunfire crack/report

# Detection thresholds (higher = less sensitive, fewer false positives)
TH_K_FOOT = 3.5           # footstep threshold multiplier
TH_K_GUN = 4.0            # gunfire threshold multiplier

# Gunfire rejection parameters
CREST_GUNFIRE_MIN = 6.0   # minimum crest factor to consider as gunfire
SPECTRAL_FLAT_GUN = 0.6   # gunfire is more spectrally flat (broadband)
SPECTRAL_FLAT_FOOT = 0.4  # footsteps are more tonal/narrow
ATTACK_TIME_GUN = 0.003   # gunfire attack < 3ms
ATTACK_TIME_FOOT = 0.015  # footsteps have slower attack > 15ms

# Cadence (seconds between steps) - tighter window
CAD_MIN = 0.18            # minimum time between footsteps
CAD_MAX = 0.55            # maximum time between footsteps (slow walk)
CAD_TOLERANCE = 0.15      # how much cadence can vary

# Energy ratio thresholds
FOOT_TO_GUN_RATIO_MIN = 0.3  # footstep bands must have this ratio vs gun bands
GUN_TO_FOOT_RATIO_MIN = 2.0  # gunfire must be this much stronger in gun bands

# UI
UI_FPS = 60
SHOT_DECAY = 0.20         # faster decay for gunshots
FOOT_DECAY = 0.50         # slower decay for footsteps (linger longer)

# Overlay styles
OVERLAY_EDGE = "edge"
OVERLAY_HUD = "hud"
OVERLAY_RING = "ring"

# =========================
#     DSP HELPERS
# =========================

def band_sos(low, high, fs=FS, order=4):
    nyq = fs / 2
    low_norm = max(low / nyq, 0.001)
    high_norm = min(high / nyq, 0.999)
    return butter(order, [low_norm, high_norm], btype='bandpass', output='sos')

# Footstep filters (3 bands for better isolation)
SOS_FOOT_LOW = band_sos(*FOOT_LOW)
SOS_FOOT_MID = band_sos(*FOOT_MID)
SOS_FOOT_HIGH = band_sos(*FOOT_HIGH)

# Gunfire filters (3 bands)
SOS_GUN_LOW = band_sos(*GUN_LOW)
SOS_GUN_MID = band_sos(*GUN_MID)
SOS_GUN_HIGH = band_sos(*GUN_HIGH)

def ste(x):
    """Short-time energy"""
    return float(np.mean(x**2))

def rms(x):
    """Root mean square"""
    return float(np.sqrt(np.mean(x**2)))

def crest(x):
    """Crest factor: peak/RMS ratio - high for impulsive sounds like gunfire"""
    r = rms(x) + EPS
    return float(np.max(np.abs(x)) / r)

def spectral_flatness(x, fs=FS):
    """
    Spectral flatness (Wiener entropy): geometric mean / arithmetic mean of spectrum.
    Returns 0-1 where 1 = white noise (flat), 0 = pure tone.
    Gunfire tends to be more flat (broadband), footsteps more tonal.
    """
    spectrum = np.abs(np.fft.rfft(x))
    spectrum = spectrum[1:]  # skip DC
    spectrum = np.maximum(spectrum, EPS)
    geo_mean = np.exp(np.mean(np.log(spectrum)))
    arith_mean = np.mean(spectrum)
    return float(geo_mean / (arith_mean + EPS))

def attack_time(x, fs=FS, threshold=0.9):
    """
    Estimate attack time: how quickly signal reaches peak.
    Gunfire: very fast attack (< 3ms)
    Footsteps: slower attack (> 10ms)
    Returns time in seconds.
    """
    envelope = np.abs(x)
    # Smooth envelope
    window_size = max(1, int(fs * 0.001))  # 1ms smoothing
    if len(envelope) > window_size:
        envelope = np.convolve(envelope, np.ones(window_size)/window_size, mode='same')

    peak_idx = np.argmax(envelope)
    peak_val = envelope[peak_idx]
    if peak_val < EPS:
        return 1.0  # no signal

    # Find when signal first exceeds threshold of peak (looking backward from peak)
    thresh_val = threshold * peak_val
    attack_start = 0
    for i in range(peak_idx, -1, -1):
        if envelope[i] < thresh_val * 0.1:  # 10% of threshold
            attack_start = i
            break

    attack_samples = peak_idx - attack_start
    return float(attack_samples / fs)

def zero_crossing_rate(x):
    """Zero crossing rate - higher for noisy/broadband signals"""
    signs = np.sign(x)
    signs[signs == 0] = 1
    crossings = np.sum(np.abs(np.diff(signs)) > 0)
    return float(crossings / len(x))

def gcc_phat(x, y, fs=FS, interp=8):
    """
    Generalized Cross-Correlation with Phase Transform.
    Improved with better interpolation and peak detection.
    Returns (time_delay, confidence).
    """
    # Zero-pad for better frequency resolution
    n = int(2 ** np.ceil(np.log2(len(x) * 2)))

    # Apply window to reduce edge effects
    window = np.hanning(len(x))
    x_win = x * window
    y_win = y * window

    X = np.fft.rfft(x_win, n)
    Y = np.fft.rfft(y_win, n)

    # Cross-power spectrum with PHAT weighting
    R = X * np.conj(Y)
    magnitude = np.abs(R)
    R = R / np.maximum(magnitude, EPS)

    # Inverse FFT with interpolation
    cc = np.fft.irfft(R, n * interp)

    # Calculate maximum possible delay based on ear distance
    max_delay_samples = int((EAR_DIST / SPEED_SOUND) * fs * interp) + 1
    max_delay_samples = min(max_delay_samples, len(cc) // 2)

    # Extract relevant portion (center around zero lag)
    cc_center = np.concatenate([cc[-max_delay_samples:], cc[:max_delay_samples + 1]])

    # Find peak with parabolic interpolation for sub-sample accuracy
    idx = int(np.argmax(cc_center))
    peak = cc_center[idx]

    # Parabolic interpolation for better accuracy
    if 0 < idx < len(cc_center) - 1:
        alpha = cc_center[idx - 1]
        beta = cc_center[idx]
        gamma = cc_center[idx + 1]
        denom = alpha - 2*beta + gamma
        if abs(denom) > EPS:
            p = 0.5 * (alpha - gamma) / denom
            idx = idx + p

    # Convert to time delay
    tau = (idx - max_delay_samples) / (fs * interp)

    # Confidence based on peak sharpness
    mean_cc = np.mean(np.abs(cc_center))
    std_cc = np.std(cc_center) + EPS
    conf = float((peak - mean_cc) / std_cc)

    return tau, conf

class RollingStats:
    """Adaptive mean/variance for per-band energy."""
    def __init__(self, hop=HOP, fs=FS, half_life_s=6.0):
        self.mu = 1e-9
        self.var = 1e-6
        self.alpha = math.exp(-(hop/fs)/half_life_s)
    def update(self, x):
        self.mu = self.alpha*self.mu + (1-self.alpha)*x
        d = x - self.mu
        self.var = self.alpha*self.var + (1-self.alpha)*(d*d)
    @property
    def sigma(self):
        return max(math.sqrt(self.var), 1e-9)

# =========================
#    EVENT DETECTION
# =========================

@dataclass
class AudioEvent:
    cls: str                  # 'footstep' or 'shot'
    theta: float              # radians; +right, -left; 0=front
    intensity: float          # 0..1
    confidence: float         # 0..1
    t: float                  # timestamp (monotonic)

class EventDetector:
    """
    Improved event detector with better footstep/gunfire discrimination.
    Uses multiple features: frequency bands, crest factor, spectral flatness,
    attack time, and cadence analysis.
    """

    def __init__(self, sensitivity: float = 1.0, debug: bool = False):
        # Separate stats for footstep and gunfire bands
        self.stats = {
            'foot_low': RollingStats(),
            'foot_mid': RollingStats(),
            'foot_high': RollingStats(),
            'gun_low': RollingStats(),
            'gun_mid': RollingStats(),
            'gun_high': RollingStats(),
        }
        self.recent_foot_times: List[float] = []
        self.recent_gun_times: List[float] = []
        self.last_event_time: float = 0.0
        self.debug = debug

        # Scale thresholds by sensitivity (higher = more sensitive = lower threshold)
        s = max(sensitivity, 0.1)
        self.k_foot = TH_K_FOOT / s
        self.k_gun = TH_K_GUN / s

        # Cooldown to prevent rapid re-triggering
        self.min_event_gap = 0.05  # 50ms minimum between events

    def _compute_direction(self, srcL: np.ndarray, srcR: np.ndarray) -> tuple:
        """
        Compute direction using improved ITD (GCC-PHAT) and ILD fusion.
        Returns (theta_radians, confidence).
        """
        # ITD via GCC-PHAT
        tau, gcc_conf = gcc_phat(srcL, srcR)

        # Clamp tau to physical limits
        max_tau = EAR_DIST / SPEED_SOUND
        tau = float(np.clip(tau, -max_tau, max_tau))

        # Convert ITD to angle (arcsin)
        sin_theta = (SPEED_SOUND * tau) / EAR_DIST
        sin_theta = float(np.clip(sin_theta, -1.0, 1.0))
        theta_itd = math.asin(sin_theta)

        # ILD (Interaural Level Difference)
        rms_l = rms(srcL) + EPS
        rms_r = rms(srcR) + EPS
        ild_db = 20.0 * math.log10(rms_r / rms_l)

        # Convert ILD to angle estimate
        # Typical ILD is ~1-2 dB per 10 degrees for low frequencies, more for high
        # Use a gentler slope for better accuracy
        ild_slope = 2.5  # degrees per dB
        theta_ild = math.radians(np.clip(ild_db * ild_slope, -90, 90))

        # Fusion: weight ITD more for low frequencies, ILD more for high
        # For mixed signals, use balanced weighting
        # ITD is generally more reliable for direction
        theta = 0.7 * theta_itd + 0.3 * theta_ild

        # Direction confidence based on GCC peak quality
        conf_dir = float(np.clip(gcc_conf / 6.0, 0.0, 1.0))

        # Reduce confidence if left/right levels are very different (might be mono or panned)
        level_ratio = max(rms_l, rms_r) / (min(rms_l, rms_r) + EPS)
        if level_ratio > 10.0:  # Very unbalanced, might be hard-panned effect
            conf_dir *= 0.5

        return theta, conf_dir

    def _is_gunfire(self, frame: np.ndarray, e_gun: float, e_foot: float) -> tuple:
        """
        Determine if the current frame is gunfire rather than footsteps.
        Returns (is_gunfire: bool, confidence: float).
        """
        mono = np.mean(frame, axis=1) if frame.ndim > 1 else frame

        # Feature 1: Crest factor (gunfire has very high crest)
        cf = crest(mono)
        crest_score = float(np.clip((cf - CREST_GUNFIRE_MIN) / 4.0, 0.0, 1.0))

        # Feature 2: Spectral flatness (gunfire is more broadband)
        sf = spectral_flatness(mono)
        flatness_score = float(np.clip((sf - SPECTRAL_FLAT_FOOT) / (SPECTRAL_FLAT_GUN - SPECTRAL_FLAT_FOOT), 0.0, 1.0))

        # Feature 3: Attack time (gunfire has very fast attack)
        at = attack_time(mono)
        attack_score = float(np.clip(1.0 - (at / ATTACK_TIME_FOOT), 0.0, 1.0))

        # Feature 4: Energy ratio (gunfire bands vs footstep bands)
        energy_ratio = e_gun / (e_foot + EPS)
        ratio_score = float(np.clip((energy_ratio - 1.0) / (GUN_TO_FOOT_RATIO_MIN - 1.0), 0.0, 1.0))

        # Combine scores with weights
        gunfire_score = (
            0.30 * crest_score +
            0.25 * flatness_score +
            0.25 * attack_score +
            0.20 * ratio_score
        )

        is_gunfire = gunfire_score > 0.5

        if self.debug:
            logging.debug(
                f"Gunfire check: crest={cf:.1f}({crest_score:.2f}) flat={sf:.2f}({flatness_score:.2f}) "
                f"attack={at*1000:.1f}ms({attack_score:.2f}) ratio={energy_ratio:.1f}({ratio_score:.2f}) "
                f"=> score={gunfire_score:.2f} is_gun={is_gunfire}"
            )

        return is_gunfire, gunfire_score

    def _cadence_confidence(self, now: float) -> float:
        """
        Calculate confidence boost based on footstep cadence pattern.
        Regular footstep cadence increases confidence.
        """
        if len(self.recent_foot_times) < 2:
            return 0.0

        # Calculate recent intervals
        intervals = []
        times = self.recent_foot_times[-6:]  # Last 6 footsteps
        for i in range(1, len(times)):
            intervals.append(times[i] - times[i-1])

        if not intervals:
            return 0.0

        # Check if intervals are in valid cadence range
        valid_intervals = [d for d in intervals if CAD_MIN <= d <= CAD_MAX]
        if not valid_intervals:
            return 0.0

        # Check for regularity (low variance in intervals)
        if len(valid_intervals) >= 2:
            mean_interval = np.mean(valid_intervals)
            std_interval = np.std(valid_intervals)
            regularity = 1.0 - min(1.0, std_interval / (mean_interval + EPS))
        else:
            regularity = 0.5

        # Confidence based on how many valid intervals and their regularity
        coverage = len(valid_intervals) / len(intervals)
        cadence_conf = coverage * regularity * 0.3  # Max 0.3 boost

        return float(cadence_conf)

    def detect(self, frame_lr: np.ndarray) -> Optional[AudioEvent]:
        """
        Detect audio events in a stereo frame.
        Returns AudioEvent if detected, None otherwise.
        """
        now = time.monotonic()

        # Cooldown check
        if (now - self.last_event_time) < self.min_event_gap:
            return None

        L, R = frame_lr[:, 0], frame_lr[:, 1]
        mono = (L + R) / 2.0

        # Apply bandpass filters - Footstep bands
        foot_low_L = sosfilt(SOS_FOOT_LOW, L)
        foot_low_R = sosfilt(SOS_FOOT_LOW, R)
        foot_mid_L = sosfilt(SOS_FOOT_MID, L)
        foot_mid_R = sosfilt(SOS_FOOT_MID, R)
        foot_high_L = sosfilt(SOS_FOOT_HIGH, L)
        foot_high_R = sosfilt(SOS_FOOT_HIGH, R)

        # Apply bandpass filters - Gunfire bands
        gun_low_L = sosfilt(SOS_GUN_LOW, L)
        gun_low_R = sosfilt(SOS_GUN_LOW, R)
        gun_mid_L = sosfilt(SOS_GUN_MID, L)
        gun_mid_R = sosfilt(SOS_GUN_MID, R)
        gun_high_L = sosfilt(SOS_GUN_HIGH, L)
        gun_high_R = sosfilt(SOS_GUN_HIGH, R)

        # Calculate energies
        e_foot_low = ste(np.hstack([foot_low_L, foot_low_R]))
        e_foot_mid = ste(np.hstack([foot_mid_L, foot_mid_R]))
        e_foot_high = ste(np.hstack([foot_high_L, foot_high_R]))
        e_gun_low = ste(np.hstack([gun_low_L, gun_low_R]))
        e_gun_mid = ste(np.hstack([gun_mid_L, gun_mid_R]))
        e_gun_high = ste(np.hstack([gun_high_L, gun_high_R]))

        # Update rolling statistics
        self.stats['foot_low'].update(e_foot_low)
        self.stats['foot_mid'].update(e_foot_mid)
        self.stats['foot_high'].update(e_foot_high)
        self.stats['gun_low'].update(e_gun_low)
        self.stats['gun_mid'].update(e_gun_mid)
        self.stats['gun_high'].update(e_gun_high)

        # Combined energies
        e_foot_total = 0.3 * e_foot_low + 0.5 * e_foot_mid + 0.2 * e_foot_high
        e_gun_total = 0.2 * e_gun_low + 0.5 * e_gun_mid + 0.3 * e_gun_high

        # Adaptive thresholds
        th_foot_low = self.stats['foot_low'].mu + self.k_foot * self.stats['foot_low'].sigma
        th_foot_mid = self.stats['foot_mid'].mu + self.k_foot * self.stats['foot_mid'].sigma
        th_foot_high = self.stats['foot_high'].mu + self.k_foot * self.stats['foot_high'].sigma

        th_gun_low = self.stats['gun_low'].mu + self.k_gun * self.stats['gun_low'].sigma
        th_gun_mid = self.stats['gun_mid'].mu + self.k_gun * self.stats['gun_mid'].sigma
        th_gun_high = self.stats['gun_high'].mu + self.k_gun * self.stats['gun_high'].sigma

        # Check if we have significant energy in any band
        foot_triggered = (
            (e_foot_low > th_foot_low) or
            (e_foot_mid > th_foot_mid) or
            (e_foot_high > th_foot_high)
        )

        gun_triggered = (
            (e_gun_low > th_gun_low) or
            (e_gun_mid > th_gun_mid) or
            (e_gun_high > th_gun_high)
        )

        if not foot_triggered and not gun_triggered:
            return None

        # Determine if this is gunfire or footsteps using multiple features
        is_gunfire, gun_score = self._is_gunfire(frame_lr, e_gun_total, e_foot_total)

        if is_gunfire and gun_triggered:
            # It's gunfire
            srcL = gun_low_L + gun_mid_L + gun_high_L
            srcR = gun_low_R + gun_mid_R + gun_high_R
            theta, conf_dir = self._compute_direction(srcL, srcR)

            # Intensity based on energy
            intensity = float(np.clip(math.sqrt(e_gun_total) * 40, 0.0, 1.0))

            # Confidence combines detection certainty and direction confidence
            confidence = float(np.clip(0.5 * gun_score + 0.5 * conf_dir, 0.0, 1.0))

            # Record timing
            self.recent_gun_times.append(now)
            if len(self.recent_gun_times) > 10:
                self.recent_gun_times = self.recent_gun_times[-10:]
            self.last_event_time = now

            return AudioEvent('shot', theta, intensity, confidence, now)

        elif foot_triggered and not is_gunfire:
            # It's a footstep
            # Use weighted combination of footstep bands for direction
            srcL = 0.3 * foot_low_L + 0.5 * foot_mid_L + 0.2 * foot_high_L
            srcR = 0.3 * foot_low_R + 0.5 * foot_mid_R + 0.2 * foot_high_R
            theta, conf_dir = self._compute_direction(srcL, srcR)

            # Intensity based on energy
            intensity = float(np.clip(math.sqrt(e_foot_total) * 60, 0.0, 1.0))

            # Base confidence from threshold excess
            excess_low = max(0, (e_foot_low - th_foot_low) / (th_foot_low + EPS))
            excess_mid = max(0, (e_foot_mid - th_foot_mid) / (th_foot_mid + EPS))
            excess_high = max(0, (e_foot_high - th_foot_high) / (th_foot_high + EPS))
            conf_energy = float(np.clip((0.3*excess_low + 0.5*excess_mid + 0.2*excess_high) / 2.0, 0.0, 0.5))

            # Add cadence confidence
            self.recent_foot_times.append(now)
            if len(self.recent_foot_times) > 12:
                self.recent_foot_times = self.recent_foot_times[-12:]
            conf_cadence = self._cadence_confidence(now)

            # Penalize if this could also be gunfire (uncertain classification)
            classification_penalty = 0.0
            if gun_score > 0.3:
                classification_penalty = gun_score * 0.3

            # Combined confidence
            confidence = float(np.clip(
                conf_energy + 0.3 * conf_dir + conf_cadence - classification_penalty,
                0.0, 1.0
            ))

            self.last_event_time = now

            return AudioEvent('footstep', theta, intensity, confidence, now)

        return None

# =========================
#     STATE / OVERLAY DATA
# =========================

@dataclass
class Marker:
    cls: str
    theta: float
    intensity: float
    confidence: float
    t0: float
    decay: float

    def alpha(self, now: float) -> float:
        age = now - self.t0
        return float(np.clip(self.confidence * math.exp(-age / self.decay), 0.0, 1.0))

class EventState:
    def __init__(self):
        self.lock = threading.Lock()
        self.markers: List[Marker] = []

    def push(self, evt: AudioEvent):
        with self.lock:
            decay = SHOT_DECAY if evt.cls == 'shot' else FOOT_DECAY
            self.markers.append(Marker(evt.cls, evt.theta, evt.intensity, evt.confidence, evt.t, decay))
            # Keep only recent
            now = time.monotonic()
            self.markers = [m for m in self.markers if m.alpha(now) > 0.02]

    def get_markers(self) -> List[Marker]:
        with self.lock:
            now = time.monotonic()
            self.markers = [m for m in self.markers if m.alpha(now) > 0.02]
            return list(self.markers)

# =========================
#         OVERLAY
# =========================

if PYSIDE:
    class EdgeOverlay(QtWidgets.QWidget):
        """
        Edge-of-screen directional indicators.
        Shows arrows/chevrons at screen edges pointing to sound sources.
        """
        def __init__(self, state: EventState, show_fps=UI_FPS):
            super().__init__(None, QtCore.Qt.Window | QtCore.Qt.FramelessWindowHint |
                           QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.Tool)
            self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
            self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
            self.state = state
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.update)
            self.timer.start(int(1000/show_fps))

            # Fullscreen, click-through
            screen = QtWidgets.QApplication.primaryScreen().geometry()
            self.setGeometry(screen)
            self.show()

        def _draw_edge_indicator(self, painter: QtGui.QPainter, theta: float, alpha: int,
                                  is_footstep: bool, intensity: float):
            """Draw an arrow indicator at the screen edge based on direction."""
            rect = self.rect()
            w, h = rect.width(), rect.height()
            cx, cy = w // 2, h // 2

            # Convert theta to screen position
            # theta: 0 = front/top, positive = right, negative = left
            # Map to screen edges

            # Determine which edge and position along that edge
            # Front (0°) = top center, Back (±180°) = bottom center
            # Left (-90°) = left center, Right (+90°) = right center

            deg = math.degrees(theta)
            margin = 40  # Distance from screen edge
            indicator_size = int(30 + 20 * intensity)  # Size based on intensity

            # Color based on type
            if is_footstep:
                color = QtGui.QColor(0, 200, 255, alpha)  # Cyan for footsteps
            else:
                color = QtGui.QColor(255, 80, 40, alpha)  # Orange-red for gunshots

            # Calculate position on screen edge
            # Use angle to determine edge and position
            if -45 <= deg <= 45:
                # Top edge (front)
                x = cx + int((deg / 45.0) * (w // 2 - margin))
                y = margin
                rotation = 180  # Arrow pointing down (into screen = forward)
            elif 45 < deg <= 135:
                # Right edge
                normalized = (deg - 45) / 90.0
                x = w - margin
                y = int(margin + normalized * (h - 2 * margin))
                rotation = 270  # Arrow pointing left (into screen)
            elif -135 <= deg < -45:
                # Left edge
                normalized = (deg + 135) / 90.0
                x = margin
                y = int(margin + normalized * (h - 2 * margin))
                rotation = 90  # Arrow pointing right (into screen)
            else:
                # Bottom edge (behind)
                if deg > 0:
                    normalized = (deg - 135) / 45.0
                else:
                    normalized = (deg + 180) / 45.0 - 1.0
                x = cx + int(normalized * (w // 2 - margin))
                y = h - margin
                rotation = 0  # Arrow pointing up (into screen = behind)

            # Draw chevron/arrow
            painter.save()
            painter.translate(x, y)
            painter.rotate(rotation)

            # Create chevron path
            path = QtGui.QPainterPath()
            s = indicator_size
            path.moveTo(0, -s//2)
            path.lineTo(-s//2, s//2)
            path.moveTo(0, -s//2)
            path.lineTo(s//2, s//2)

            # Draw with glow effect for visibility
            pen_width = 4 if is_footstep else 6
            # Outer glow
            glow_color = QtGui.QColor(color)
            glow_color.setAlpha(alpha // 3)
            painter.setPen(QtGui.QPen(glow_color, pen_width + 4, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
            painter.drawPath(path)
            # Inner line
            painter.setPen(QtGui.QPen(color, pen_width, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
            painter.drawPath(path)

            painter.restore()

        def paintEvent(self, event):
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.Antialiasing)

            now = time.monotonic()
            for m in self.state.get_markers():
                alpha = int(255 * m.alpha(now))
                if alpha <= 5:
                    continue
                self._draw_edge_indicator(painter, m.theta, alpha,
                                         m.cls == 'footstep', m.intensity)
            painter.end()


    class HUDCompassOverlay(QtWidgets.QWidget):
        """
        Small HUD-style compass in a corner of the screen.
        Shows a mini radar-like display with directional indicators.
        """
        def __init__(self, state: EventState, show_fps=UI_FPS, position="bottom-right"):
            super().__init__(None, QtCore.Qt.Window | QtCore.Qt.FramelessWindowHint |
                           QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.Tool)
            self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
            self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
            self.state = state
            self.position = position
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.update)
            self.timer.start(int(1000/show_fps))

            # HUD size and position
            self.hud_size = 150
            self.margin = 30

            screen = QtWidgets.QApplication.primaryScreen().geometry()
            self._setup_position(screen)
            self.show()

        def _setup_position(self, screen_geom):
            """Position the HUD widget in the specified corner."""
            size = self.hud_size + self.margin * 2
            if self.position == "bottom-right":
                x = screen_geom.width() - size
                y = screen_geom.height() - size
            elif self.position == "bottom-left":
                x = 0
                y = screen_geom.height() - size
            elif self.position == "top-right":
                x = screen_geom.width() - size
                y = 0
            elif self.position == "top-left":
                x = 0
                y = 0
            else:  # center-bottom
                x = (screen_geom.width() - size) // 2
                y = screen_geom.height() - size

            self.setGeometry(x, y, size, size)

        def paintEvent(self, event):
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.Antialiasing)

            rect = self.rect()
            cx, cy = rect.center().x(), rect.center().y()
            radius = self.hud_size // 2 - 10

            # Draw background circle (semi-transparent)
            painter.setBrush(QtGui.QColor(0, 0, 0, 100))
            painter.setPen(QtGui.QPen(QtGui.QColor(100, 100, 100, 150), 2))
            painter.drawEllipse(QtCore.QPointF(cx, cy), radius + 5, radius + 5)

            # Draw compass ring
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 80), 1))
            painter.drawEllipse(QtCore.QPointF(cx, cy), radius, radius)
            painter.drawEllipse(QtCore.QPointF(cx, cy), radius * 0.5, radius * 0.5)

            # Draw cardinal direction markers
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 120), 2))
            # Front marker (top)
            painter.drawLine(cx, cy - radius + 5, cx, cy - radius + 15)
            # Small ticks for sides
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 60), 1))
            painter.drawLine(cx + radius - 5, cy, cx + radius - 12, cy)  # Right
            painter.drawLine(cx - radius + 5, cy, cx - radius + 12, cy)  # Left
            painter.drawLine(cx, cy + radius - 5, cx, cy + radius - 12)  # Back

            # Draw player indicator (small triangle at center pointing up)
            painter.setBrush(QtGui.QColor(255, 255, 255, 150))
            painter.setPen(QtCore.Qt.NoPen)
            player_path = QtGui.QPainterPath()
            player_path.moveTo(cx, cy - 8)
            player_path.lineTo(cx - 5, cy + 4)
            player_path.lineTo(cx + 5, cy + 4)
            player_path.closeSubpath()
            painter.drawPath(player_path)

            # Draw event markers
            now = time.monotonic()
            for m in self.state.get_markers():
                alpha = int(255 * m.alpha(now))
                if alpha <= 5:
                    continue

                # Position on compass (theta: 0=front/up, positive=right)
                # Scale by intensity for distance effect
                dist = radius * (0.4 + 0.5 * m.intensity)
                ax = cx + dist * math.sin(m.theta)
                ay = cy - dist * math.cos(m.theta)

                # Color and size based on type
                if m.cls == 'footstep':
                    color = QtGui.QColor(0, 200, 255, alpha)
                    dot_size = 6
                else:
                    color = QtGui.QColor(255, 80, 40, alpha)
                    dot_size = 8

                # Draw dot with glow
                glow_color = QtGui.QColor(color)
                glow_color.setAlpha(alpha // 2)
                painter.setBrush(glow_color)
                painter.setPen(QtCore.Qt.NoPen)
                painter.drawEllipse(QtCore.QPointF(ax, ay), dot_size + 3, dot_size + 3)

                painter.setBrush(color)
                painter.drawEllipse(QtCore.QPointF(ax, ay), dot_size, dot_size)

            painter.end()


    class RingOverlay(QtWidgets.QWidget):
        """
        Original full-screen radial ring compass overlay.
        Shows a large ring around screen center with directional ticks.
        """
        def __init__(self, state: EventState, show_fps=UI_FPS):
            super().__init__(None, QtCore.Qt.Window | QtCore.Qt.FramelessWindowHint |
                           QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.Tool)
            self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
            self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
            self.state = state
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.update)
            self.timer.start(int(1000/show_fps))

            # Fullscreen, click-through
            screen = QtWidgets.QApplication.primaryScreen().geometry()
            self.setGeometry(screen)
            self.show()

        def paintEvent(self, event):
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.Antialiasing)
            rect = self.rect()
            cx, cy = rect.center().x(), rect.center().y()
            radius = min(rect.width(), rect.height())//2 - 20

            # Outer ring
            pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 80), 2)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawEllipse(QtCore.QPointF(cx, cy), radius, radius)

            # 0° marker (front)
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 120), 3))
            painter.drawLine(cx, cy - radius, cx, cy - radius + 20)

            # Draw markers
            now = time.monotonic()
            for m in self.state.get_markers():
                alpha = int(255 * m.alpha(now))
                if alpha <= 0:
                    continue
                color = QtGui.QColor(80, 200, 255, alpha) if m.cls == 'footstep' else QtGui.QColor(255, 120, 80, alpha)
                painter.setPen(QtGui.QPen(color, 6 if m.cls == 'shot' else 4))
                # location on ring edge (0° at top)
                ax = cx + radius * math.sin(m.theta)
                ay = cy - radius * math.cos(m.theta)
                # tick
                painter.drawLine(QtCore.QPointF(ax, ay),
                                 QtCore.QPointF(cx + (radius-25) * math.sin(m.theta),
                                                cy - (radius-25) * math.cos(m.theta)))
                # short arc tail
                painter.setPen(QtGui.QPen(color, 2))
                painter.drawArc(int(cx-radius), int(cy-radius), int(2*radius), int(2*radius),
                                int((90 - math.degrees(m.theta) - 6) * 16), int(12 * 16))
            painter.end()

    # Backward compatibility alias
    CompassOverlay = RingOverlay

# =========================
#   WINDOWS CORE AUDIO CAPTURE
# =========================

if WINDOWS_AUDIO_AVAILABLE:
    class WindowsLoopbackCapture:
        """PyAudioWPatch WASAPI loopback capture - captures real system audio output."""

        def __init__(self, device_name: str, fs=FS, hop=HOP, win=WIN):
            self.device_name = device_name
            self.fs = fs
            self.hop = hop
            self.win = win
            self.stop_flag = threading.Event()
            self.capture_thread = None
            self.pyaudio_instance = None
            self.stream = None
            self.audio_queue = queue.Queue(maxsize=100)

        def _find_wasapi_loopback_device(self):
            """Find WASAPI loopback device using PyAudioWPatch."""
            try:
                # Use PyAudioWPatch for proper WASAPI loopback
                p = pyaudio_wpatch.PyAudio()

                # Get default WASAPI info
                wasapi_info = p.get_host_api_info_by_type(pyaudio_wpatch.paWASAPI)
                logging.debug("WASAPI info: %s", wasapi_info)

                # Get default speakers
                default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
                logging.info("Default speakers: %s", default_speakers['name'])

                # Check if our target device matches default or find it
                if self.device_name.lower() in default_speakers['name'].lower():
                    logging.info("Target device matches default speakers")
                    target_speakers = default_speakers
                else:
                    # Search for device by name
                    target_speakers = None
                    for i in range(p.get_device_count()):
                        try:
                            device_info = p.get_device_info_by_index(i)
                            if (self.device_name.lower() in device_info['name'].lower() and
                                device_info['maxOutputChannels'] > 0):
                                target_speakers = device_info
                                break
                        except Exception as e:
                            logging.debug("Error checking device %d: %s", i, e)
                            continue

                    if not target_speakers:
                        logging.info("Target device not found, using default speakers")
                        target_speakers = default_speakers

                # Find the corresponding loopback device
                if not target_speakers.get("isLoopbackDevice", False):
                    loopback_device = None
                    for loopback in p.get_loopback_device_info_generator():
                        if target_speakers["name"] in loopback["name"]:
                            loopback_device = loopback
                            break

                    if not loopback_device:
                        logging.error("No loopback device found for: %s", target_speakers['name'])
                        return None, None
                else:
                    loopback_device = target_speakers

                logging.info("Found loopback device: (%d) %s",
                           loopback_device['index'], loopback_device['name'])
                return p, loopback_device

            except Exception as e:
                logging.error("Error finding WASAPI device: %s", e)
                return None, None

        def _audio_callback(self, in_data, frame_count, time_info, status):
            """PyAudioWPatch callback for captured loopback audio."""
            if status:
                logging.debug("PyAudioWPatch status: %s", status)

            try:
                # Convert captured loopback audio to numpy array
                audio_data = np.frombuffer(in_data, dtype=np.float32)

                # Reshape based on channels
                if len(audio_data) == frame_count:
                    # Mono - convert to stereo
                    audio_data = np.repeat(audio_data.reshape(-1, 1), 2, axis=1)
                else:
                    # Stereo
                    audio_data = audio_data.reshape(-1, 2)

                # Queue for processing
                try:
                    self.audio_queue.put_nowait(audio_data)
                except queue.Full:
                    pass  # Drop frames if queue full

            except Exception as e:
                logging.debug("Audio callback error: %s", e)

            return (None, pyaudio_wpatch.paContinue)

        def _capture_worker(self, on_frame):
            """Worker thread that processes captured loopback audio."""
            buf = np.zeros((self.win, 2), dtype=np.float32)
            wpos = 0

            while not self.stop_flag.is_set():
                try:
                    # Get real audio data from WASAPI loopback
                    chunk = self.audio_queue.get(timeout=0.1)

                    # Add to sliding window buffer
                    frames_available = chunk.shape[0]
                    i = 0

                    while i < frames_available:
                        n = min(frames_available - i, self.win - wpos)
                        buf[wpos:wpos+n] = chunk[i:i+n]
                        wpos += n
                        i += n

                        if wpos >= self.win:
                            on_frame(buf.copy())
                            # Slide buffer
                            buf[:-self.hop] = buf[self.hop:]
                            wpos = self.win - self.hop

                except queue.Empty:
                    continue
                except Exception as e:
                    logging.debug("Capture worker error: %s", e)
                    time.sleep(0.1)

        def start(self, on_frame):
            """Start PyAudioWPatch WASAPI loopback capture."""
            self.pyaudio_instance, device_info = self._find_wasapi_loopback_device()
            if not self.pyaudio_instance or not device_info:
                raise RuntimeError("Could not find WASAPI loopback device")

            try:
                # Open WASAPI loopback stream using PyAudioWPatch
                self.stream = self.pyaudio_instance.open(
                    format=pyaudio_wpatch.paFloat32,
                    channels=int(device_info['maxInputChannels']),
                    rate=int(device_info['defaultSampleRate']),
                    input=True,
                    input_device_index=device_info['index'],
                    frames_per_buffer=self.hop,
                    stream_callback=self._audio_callback
                )

                self.stream.start_stream()
                logging.info("Started PyAudioWPatch WASAPI loopback on device: %s", device_info['name'])

                # Start processing thread
                self.capture_thread = threading.Thread(target=self._capture_worker, args=(on_frame,), daemon=True)
                self.capture_thread.start()

            except Exception as e:
                logging.error("Failed to start PyAudioWPatch WASAPI loopback: %s", e)
                self.stop()
                raise

        def stop(self):
            """Stop PyAudioWPatch WASAPI loopback capture."""
            self.stop_flag.set()

            if self.stream:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                except:
                    pass

            if self.pyaudio_instance:
                try:
                    self.pyaudio_instance.terminate()
                except:
                    pass

            if self.capture_thread:
                self.capture_thread.join(timeout=1.0)

# =========================
#       AUDIO PIPE
# =========================

class AudioLoop:
    def __init__(self, device_index: Optional[int], fs=FS, hop=HOP, win=WIN, allow_input_fallback: bool = False, force_loopback_alias: bool = False):
        self.fs = fs
        self.hop = hop
        self.win = win
        self.device_index = device_index
        self.q = queue.Queue(maxsize=64)
        self.stream = None
        self.worker_thread = None
        self.stop_flag = threading.Event()
        self._in_channels = 2  # actual stream input channels (1 or 2). We expand to 2 in callback if needed.
        self.allow_input_fallback = allow_input_fallback
        self.force_loopback_alias = force_loopback_alias
        self.windows_capture = None  # Windows fallback capture

    def _callback(self, indata, frames, time_info, status):
        if status:
            logging.debug("Audio status: %s", status)
        try:
            # Ensure we always push 2-channel frames downstream. Duplicate mono if needed.
            data = indata
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            if data.shape[1] >= 2:
                out = data[:, :2].copy()
            else:
                # mono -> stereo
                out = np.repeat(data, 2, axis=1)
            self.q.put_nowait(out)
        except queue.Full:
            pass

    def start(self, on_frame):
        # Try direct WASAPI loopback approach first
        wasapi_loopback_supported = False
        if hasattr(sd, "WasapiSettings"):
            try:
                # Test creating WasapiSettings - some versions support loopback differently
                test_settings = sd.WasapiSettings(exclusive=False)
                wasapi_loopback_supported = True
                logging.debug("WASAPI settings available, will attempt loopback capture")
            except Exception as e:
                logging.debug("WASAPI settings unavailable: %s", e)
        else:
            logging.debug("sounddevice.WasapiSettings not present; will use alternative approaches")

        # Helper to build WASAPI extra settings for loopback
        def make_loopback_extra(exclusive: bool):
            if not wasapi_loopback_supported:
                return None
            try:
                # Try the modern approach first
                return sd.WasapiSettings(exclusive=exclusive, loopback=True)
            except TypeError:
                # Fall back to older approach - just exclusive mode on output device
                try:
                    return sd.WasapiSettings(exclusive=exclusive)
                except Exception as e:
                    logging.debug("Creating WASAPI settings failed (exclusive=%s): %s", exclusive, e)
                    return None
            except Exception as e:
                logging.debug("Creating WASAPI loopback settings failed (exclusive=%s): %s", exclusive, e)
                return None

        # If device_index is None, use default output device
        if self.device_index is None:
            try:
                hais = sd.query_hostapis()
                wasapi_idx = next(i for i,h in enumerate(hais) if 'WASAPI' in h['name'].upper())
                dev_out = sd.query_hostapis(wasapi_idx)['default_output_device']
            except Exception:
                dev_out = sd.default.device[1]  # output
        else:
            dev_out = self.device_index

        # Determine device caps and host
        try:
            dev_caps = sd.query_devices(dev_out)
            host_name = sd.query_hostapis(dev_caps['hostapi'])['name']
        except Exception:
            dev_caps = {'max_output_channels': 0, 'max_input_channels': 0, 'name': str(dev_out)}
            host_name = ""

        def find_loopback_alias(base_name: str) -> Optional[int]:
            try:
                all_devs = sd.query_devices()
            except Exception as e:
                logging.debug("Loopback alias search failed (query_devices): %s", e)
                return None
            target = (base_name or "").lower().replace(" (loopback)", "")
            for i, d in enumerate(all_devs):
                name = d.get("name", "")
                if not name:
                    continue
                name_lower = name.lower()
                if "loopback" not in name_lower:
                    continue
                if target and target not in name_lower:
                    continue
                if d.get("max_input_channels", 0) <= 0:
                    continue
                try:
                    ha_i = sd.query_hostapis(d["hostapi"])["name"]
                except Exception:
                    ha_i = ""
                if ha_i and "WASAPI" not in ha_i.upper():
                    continue
                logging.debug("Found loopback alias %s (%d) for %s", name, i, base_name)
                return i
            return None

        # If device has output channels and host isn't WASAPI, try to remap by name to WASAPI;
        # If it is an input-only device (e.g., 'Stereo Mix'), we'll treat it as direct input capture.
        try:
            if dev_out is not None and dev_caps.get('max_output_channels', 0) > 0:
                if 'WASAPI' not in host_name.upper():
                    target_name = dev_caps.get('name', '')
                    all_devs = sd.query_devices()
                    for i, d in enumerate(all_devs):
                        try:
                            ha_i = sd.query_hostapis(d['hostapi'])['name']
                        except Exception:
                            continue
                        if d.get('max_output_channels', 0) > 0 and 'WASAPI' in ha_i.upper() and d['name'] == target_name:
                            logging.info("Remapping device '%s' to WASAPI index %d for loopback", target_name, i)
                            dev_out = i
                            dev_caps = sd.query_devices(dev_out)
                            host_name = sd.query_hostapis(dev_caps['hostapi'])['name']
                            break
        except Exception as e:
            logging.debug("Device remap check failed: %s", e)

        need_loopback_alias = self.force_loopback_alias or not wasapi_loopback_supported
        if dev_out is not None and dev_caps.get("max_output_channels", 0) > 0 and need_loopback_alias:
            alias_idx = find_loopback_alias(dev_caps.get("name", str(dev_out)))
            if alias_idx is not None:
                logging.info("Using loopback alias device index %s for \"%s\"", alias_idx, dev_caps.get("name", str(dev_out)))
                try:
                    dev_caps = sd.query_devices(alias_idx)
                    host_name = sd.query_hostapis(dev_caps['hostapi'])['name']
                except Exception:
                    dev_caps = {'max_output_channels': dev_caps.get('max_output_channels', 0),
                                'max_input_channels': dev_caps.get('max_input_channels', 0),
                                'name': dev_caps.get('name', str(alias_idx))}
                    host_name = ""
                dev_out = alias_idx
            else:
                if not wasapi_loopback_supported:
                    logging.warning("sounddevice.WasapiSettings unavailable; no '(loopback)' alias found for '%s'.", dev_caps.get("name", str(dev_out)))
                else:
                    logging.warning("Forced loopback alias requested but no matching '(loopback)' device found for '%s'.", dev_caps.get("name", str(dev_out)))

        def try_open_on_device(device_idx: int) -> bool:
            # Determine desired input channels based on output capabilities
            try:
                caps = sd.query_devices(device_idx)
                ha_name = sd.query_hostapis(caps['hostapi'])['name']
                out_ch = int(caps.get('max_output_channels', 0) or 0)
                in_ch = int(caps.get('max_input_channels', 0) or 0)
            except Exception:
                out_ch = 0
                in_ch = 0
                ha_name = ""

            logging.debug("Evaluating device %s via host %s (out=%d in=%d, wasapi_loopback=%s)",
                          device_idx, ha_name, out_ch, in_ch, wasapi_loopback_supported)

            # Determine if we can use WASAPI loopback on this device
            is_output = out_ch > 0
            is_wasapi_host = 'WASAPI' in ha_name.upper()
            can_use_loopback = is_output and is_wasapi_host and wasapi_loopback_supported

            if can_use_loopback:
                # WASAPI loopback: use output channels as basis
                desired_ch = 2 if out_ch >= 2 else 1
                logging.debug("Will attempt WASAPI loopback on output device %s", device_idx)
            else:
                # Regular input capture or fallback
                if in_ch <= 0:
                    logging.debug("Skipping device %s: no input channels (out=%d in=%d)", device_idx, out_ch, in_ch)
                    return False
                desired_ch = 2 if in_ch >= 2 else 1
                logging.debug("Will attempt regular input capture on device %s", device_idx)

            def open_stream_with_channels(ch: int, sr: int, extra, blocksize):
                return sd.InputStream(samplerate=sr, channels=ch, dtype='float32',
                                      blocksize=blocksize, callback=self._callback,
                                      device=device_idx, latency='low', extra_settings=extra)

            # Try combinations: (exclusive False/True) x (fs/44100) x (stereo/mono)
            exclusives = (False, True) if can_use_loopback else (False,)
            for exclusive in exclusives:
                extra = make_loopback_extra(exclusive) if can_use_loopback else None
                for sr in (self.fs, 44100):
                    channel_options = (desired_ch, 1) if desired_ch != 1 else (1,)
                    for ch in channel_options:
                        for bs in (self.hop, None):
                            extra_desc = "wasapi-loopback" if (extra and can_use_loopback) else "standard"
                            logging.debug("Attempting stream dev=%s exclusive=%s sr=%d ch=%d blocksize=%s extra=%s",
                                          device_idx, exclusive, sr, ch, bs, extra_desc)
                            try:
                                self.stream = open_stream_with_channels(ch, sr, extra, bs)
                                self.stream.start()
                                self._in_channels = ch
                                try:
                                    d = sd.query_devices(device_idx)
                                    ha = sd.query_hostapis(d['hostapi'])['name']
                                    kind = "loopback" if can_use_loopback else "input"
                                    logging.info("Opened %s on device %s '%s' via %s (exclusive=%s) at %d Hz, blocksize=%s, with %d ch%s",
                                                 kind, str(device_idx), d['name'], ha, exclusive, sr, str(bs), ch,
                                                 " (duplicated to stereo)" if ch == 1 else "")
                                except Exception:
                                    logging.info("Opened device %s (exclusive=%s) at %d Hz, blocksize=%s, with %d ch",
                                                 str(device_idx), exclusive, sr, str(bs), ch)
                                return True
                            except Exception as e:
                                msg = str(e)
                                logging.debug("Open attempt failed on dev %s (exclusive=%s, sr=%d, ch=%d): %s",
                                              str(device_idx), exclusive, sr, ch, msg)
                                continue
            return False

        if not try_open_on_device(dev_out):
            # Fallback: try default WASAPI output device
            try:
                hais = sd.query_hostapis()
                wasapi_idx = next(i for i,h in enumerate(hais) if 'WASAPI' in h['name'].upper())
                dev_default = sd.query_hostapis(wasapi_idx)['default_output_device']
            except Exception:
                dev_default = None
            if dev_default is not None and dev_default != dev_out:
                logging.warning("Primary device failed; trying default WASAPI output device index %s", dev_default)
                if not try_open_on_device(dev_default):
                    if self.allow_input_fallback:
                        # As a last resort, try any 'Stereo Mix' input device if present
                        try:
                            devs = sd.query_devices()
                            stereo_idx = next((i for i, d in enumerate(devs)
                                               if d.get('max_input_channels', 0) > 0 and 'stereo mix' in d['name'].lower()), None)
                        except Exception:
                            stereo_idx = None
                        if stereo_idx is not None and try_open_on_device(stereo_idx):
                            logging.warning("Fell back to 'Stereo Mix' input device index %s", stereo_idx)
                        else:
                            raise RuntimeError("Failed to open loopback stream on the selected device and the default WASAPI output. Run with --list-outputs and choose the '(loopback)' entry or pass --use-loopback-alias; you can also enable --allow-input-fallback.")
                    else:
                        raise RuntimeError("Failed to open loopback stream on the selected device and the default WASAPI output. Run with --list-outputs and choose the '(loopback)' entry or pass --use-loopback-alias; you can also enable --allow-input-fallback.")
            else:
                if self.allow_input_fallback:
                    # Try 'Stereo Mix' if available
                    try:
                        devs = sd.query_devices()
                        stereo_idx = next((i for i, d in enumerate(devs)
                                           if d.get('max_input_channels', 0) > 0 and 'stereo mix' in d['name'].lower()), None)
                    except Exception:
                        stereo_idx = None
                    if stereo_idx is not None and try_open_on_device(stereo_idx):
                        logging.warning("Fell back to 'Stereo Mix' input device index %s", stereo_idx)
                    else:
                        # Try Windows Core Audio fallback as last resort
                        if WINDOWS_AUDIO_AVAILABLE and self.device_index is not None:
                            try:
                                device_caps = sd.query_devices(self.device_index)
                                device_name = device_caps.get('name', '')
                                logging.info("Attempting Windows Core Audio fallback for device: %s", device_name)
                                self.windows_capture = WindowsLoopbackCapture(device_name, self.fs, self.hop, self.win)
                                self.windows_capture.start(on_frame)
                                logging.info("Successfully started Windows Core Audio fallback")
                                return
                            except Exception as e:
                                logging.error("Windows Core Audio fallback failed: %s", e)

                        raise RuntimeError("Failed to open a loopback stream on the selected device. Run with --list-outputs and choose the '(loopback)' entry or pass --use-loopback-alias; you can also enable --allow-input-fallback.")
                else:
                    # Try Windows Core Audio fallback as last resort
                    if WINDOWS_AUDIO_AVAILABLE and self.device_index is not None:
                        try:
                            device_caps = sd.query_devices(self.device_index)
                            device_name = device_caps.get('name', '')
                            logging.info("Attempting Windows Core Audio fallback for device: %s", device_name)
                            self.windows_capture = WindowsLoopbackCapture(device_name, self.fs, self.hop, self.win)
                            self.windows_capture.start(on_frame)
                            logging.info("Successfully started Windows Core Audio fallback")
                            return
                        except Exception as e:
                            logging.error("Windows Core Audio fallback failed: %s", e)

                    raise RuntimeError("Failed to open a loopback stream on the selected device. Run with --list-outputs and choose the '(loopback)' entry or pass --use-loopback-alias; you can also enable --allow-input-fallback.")

        # worker: build 20 ms window, slide by 10 ms
        buf = np.zeros((self.win, 2), dtype=np.float32)
        wpos = 0

        def worker():
            nonlocal buf, wpos
            while not self.stop_flag.is_set():
                try:
                    chunk = self.q.get(timeout=0.25)
                except queue.Empty:
                    continue
                i = 0
                frames = chunk.shape[0]
                while i < frames:
                    n = min(frames - i, self.win - wpos)
                    buf[wpos:wpos+n] = chunk[i:i+n]
                    wpos += n; i += n
                    if wpos == self.win:
                        on_frame(buf.copy())
                        # slide by HOP
                        buf[:-self.hop] = buf[self.hop:]
                        wpos = self.win - self.hop

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def stop(self):
        self.stop_flag.set()
        if self.stream:
            self.stream.stop()
            self.stream.close()
        if self.worker_thread:
            self.worker_thread.join(timeout=1.0)
        if self.windows_capture:
            self.windows_capture.stop()

# =========================
#          MAIN
# =========================

def build_parser():
    p = argparse.ArgumentParser(
        description="Audio→Visual Compass (COD/Warzone) — Directional Audio Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Overlay Styles:
  edge    - Arrows at screen edges pointing to sound direction (recommended)
  hud     - Small radar compass in corner of screen
  ring    - Large ring around screen center (original style)

Examples:
  python "BotLobbies StepSense.py" --list-outputs
  python "BotLobbies StepSense.py" --device 13 --overlay edge
  python "BotLobbies StepSense.py" --device 13 --overlay hud --hud-position bottom-right
  python "BotLobbies StepSense.py" --device 13 --debug-audio --sensitivity 1.5
"""
    )
    p.add_argument("--device", type=int, default=None,
                   help="WASAPI output device index to loopback-capture (see --list-devices)")
    p.add_argument("--list-devices", action="store_true",
                   help="List audio devices and exit")
    p.add_argument("--list-outputs", action="store_true",
                   help="List only output devices and exit")
    p.add_argument("--choose-device", action="store_true",
                   help="Interactively list devices and prompt for selection, then start")

    # Overlay options
    p.add_argument("--overlay", type=str, default="edge", choices=["edge", "hud", "ring", "none"],
                   help="Overlay style: 'edge' (screen edge arrows), 'hud' (corner compass), 'ring' (center ring), 'none' (disabled)")
    p.add_argument("--no-overlay", action="store_true",
                   help="Disable on-screen compass (same as --overlay none)")
    p.add_argument("--hud-position", type=str, default="bottom-right",
                   choices=["bottom-right", "bottom-left", "top-right", "top-left", "center-bottom"],
                   help="Position for HUD compass overlay (default: bottom-right)")

    # Detection tuning
    p.add_argument("--sensitivity", type=float, default=1.0,
                   help="Detection sensitivity multiplier (higher = more sensitive, default=1.0)")
    p.add_argument("--min-confidence", type=float, default=0.4,
                   help="Minimum event confidence to display, 0.0-1.0 (default=0.4)")
    p.add_argument("--footsteps-only", action="store_true",
                   help="Only show footstep detections, ignore gunfire")
    p.add_argument("--gunfire-only", action="store_true",
                   help="Only show gunfire detections, ignore footsteps")

    # Debug and logging
    p.add_argument("--loglevel", default="INFO",
                   help="Logging level (DEBUG, INFO, WARNING)")
    p.add_argument("--debug-audio", action="store_true",
                   help="Log periodic audio RMS levels to confirm capture")
    p.add_argument("--debug-detection", action="store_true",
                   help="Log detailed detection analysis (gunfire vs footstep scoring)")

    # Audio device options
    p.add_argument("--allow-input-fallback", action="store_true",
                   help="Allow automatic fallback to input devices like 'Stereo Mix' if loopback fails")
    p.add_argument("--use-loopback-alias", action="store_true",
                   help="Prefer matching '(loopback)' input alias when opening output devices")

    return p

def list_devices():
    print("=== Devices (use index with --device) ===")
    print(sd.query_devices())

def list_output_devices():
    devs = sd.query_devices()
    hostapis = sd.query_hostapis()
    def host_name(idx: int) -> str:
        try:
            return hostapis[devs[idx]['hostapi']]['name']
        except Exception:
            return ""
    try:
        default_out = sd.default.device[1]
    except Exception:
        default_out = None

    print("=== Output devices (use index with --device) ===")
    for i, d in enumerate(devs):
        if d.get('max_output_channels', 0) <= 0:
            continue
        mark = "*" if (default_out is not None and i == default_out) else " "
        hn = host_name(i)
        print(f"{mark} {i:>3} {d['name']}, {hn} ({d['max_input_channels']} in, {d['max_output_channels']} out)")

def choose_device_interactive() -> Optional[int]:
    """List only output devices and prompt the user to choose one.
    Prefers WASAPI output devices for reliable loopback capture.
    Returns the selected device index or None if user cancels.
    """
    devs = sd.query_devices()
    hostapis = sd.query_hostapis()

    def host_name(idx: int) -> str:
        try:
            return hostapis[devs[idx]['hostapi']]['name']
        except Exception:
            return ""

    print("=== Output devices (use index with --device) ===")
    outputs = [(i, d) for i, d in enumerate(devs) if d.get('max_output_channels', 0) > 0]
    for i, d in outputs:
        hn = host_name(i)
        print(f"  {i:>3} {d['name']}, {hn} ({d['max_input_channels']} in, {d['max_output_channels']} out)")

    # Recommend WASAPI output devices
    wasapi_out = [(i, d) for i, d in outputs if 'WASAPI' in host_name(i).upper()]
    if wasapi_out:
        print("\nRecommended (WASAPI output for loopback):")
        for i, d in wasapi_out:
            print(f"  {i:>3} {d['name']}, {host_name(i)} ({d['max_input_channels']} in, {d['max_output_channels']} out)")
    try:
        raw = input("\nEnter device index to use (or just press Enter to cancel): ").strip()
    except EOFError:
        return None
    if raw == "":
        return None
    try:
        idx = int(raw)
    except ValueError:
        print("Not a valid integer device index.")
        return None
    if not (0 <= idx < len(devs)) or devs[idx].get('max_output_channels', 0) <= 0:
        print("Index not an output device or out of range.")
        return None
    return idx

def create_overlay(state: EventState, overlay_style: str, hud_position: str):
    """Create the appropriate overlay widget based on style selection."""
    if not PYSIDE:
        return None

    if overlay_style == "edge":
        return EdgeOverlay(state)
    elif overlay_style == "hud":
        return HUDCompassOverlay(state, position=hud_position)
    elif overlay_style == "ring":
        return RingOverlay(state)
    else:
        return None


def main():
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s: %(message)s")

    # Handle list commands first
    if args.list_outputs:
        list_output_devices()
        return
    elif args.list_devices:
        list_devices()
        return

    # Determine overlay style
    overlay_style = args.overlay
    if args.no_overlay:
        overlay_style = "none"

    # Create filter function for event types
    def event_filter(evt: AudioEvent) -> bool:
        if evt is None:
            return False
        if evt.confidence < args.min_confidence:
            return False
        if args.footsteps_only and evt.cls != 'footstep':
            return False
        if args.gunfire_only and evt.cls != 'shot':
            return False
        return True

    def run_with_device(device_index: Optional[int]) -> bool:
        """Run the detector with the specified device. Returns True on success."""
        # Create detector with debug flag
        detector = EventDetector(
            sensitivity=args.sensitivity,
            debug=args.debug_detection
        )
        state = EventState()
        audio = AudioLoop(
            device_index=device_index,
            allow_input_fallback=args.allow_input_fallback,
            force_loopback_alias=args.use_loopback_alias
        )

        def on_frame(frame_lr):
            evt = detector.detect(frame_lr)

            # Debug audio levels
            if args.debug_audio:
                if not hasattr(on_frame, "_acc"):
                    on_frame._acc = 0
                    on_frame._t0 = time.monotonic()
                on_frame._acc += float(np.sqrt(np.mean(frame_lr**2)))
                if (time.monotonic() - on_frame._t0) >= 0.5:
                    rms_val = on_frame._acc / max(1, int(0.5/(HOP/FS)))
                    logging.info("AUDIO RMS ~ %.4f", rms_val)
                    on_frame._acc = 0
                    on_frame._t0 = time.monotonic()

            # Filter and push events
            if event_filter(evt):
                state.push(evt)
                logging.info(
                    f"{evt.cls:8s} az={math.degrees(evt.theta):+06.1f}° "
                    f"I={evt.intensity:.2f} C={evt.confidence:.2f}"
                )

        try:
            audio.start(on_frame)
        except Exception as e:
            logging.error("Failed to open audio device: %s", e)
            return False

        logging.info("Audio capture started successfully")
        logging.info("Overlay style: %s", overlay_style)

        # Run in headless or overlay mode
        if overlay_style == "none" or not PYSIDE:
            if not PYSIDE and overlay_style != "none":
                logging.warning("PySide6 not available; running headless (no overlay).")
            print("Running… Press Ctrl+C to quit.")
            print(f"Detection sensitivity: {args.sensitivity}")
            print(f"Minimum confidence: {args.min_confidence}")
            if args.footsteps_only:
                print("Mode: Footsteps only")
            elif args.gunfire_only:
                print("Mode: Gunfire only")
            try:
                while True:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                print("\nStopping...")
            finally:
                audio.stop()
            return True

        # Qt overlay mode
        app = QtWidgets.QApplication([])
        overlay = create_overlay(state, overlay_style, args.hud_position)
        if overlay is None:
            logging.error("Failed to create overlay")
            audio.stop()
            return False

        logging.info("Overlay created: %s", type(overlay).__name__)

        try:
            app.exec()
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            audio.stop()

        return True

    # Interactive device selection mode
    if args.choose_device:
        while True:
            idx = choose_device_interactive()
            if idx is None:
                return
            if run_with_device(idx):
                return
            print("\nCould not open that device. Please choose another output device.\n")
            continue
    else:
        # Direct device mode
        if not run_with_device(args.device):
            print("\nFailed to start. Try --list-outputs to see available devices.")
            print("Then use --device <index> to select a WASAPI output device.")
            return

if __name__ == "__main__":
    main()
