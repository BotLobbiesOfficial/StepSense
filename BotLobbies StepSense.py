#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio→Visual Compass for COD/Warzone (stereo path) — No Haptics
- WASAPI loopback capture (sounddevice)
- Footstep detection (dual-band: 100–450 Hz + 1–4 kHz) with adaptive thresholds + cadence prior
- Gunshot detection (crest/decay guard + spectral flatness + 2 kHz peak discrimination)
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
from scipy.signal import butter, sosfilt, resample_poly

# Windows-specific audio capture (PyAudioWPatch for WASAPI loopback)
try:
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

# Bands (Warzone / Black Ops tuned from community spectral analysis)
# Footstep "body" zone: 100-450 Hz — heel impact, gear resonance, hollow-surface boom
# Sources: ExpertBeacon, ArtIsWarTools, SteelSeries Sonar community EQ data
FOOT_A = (100, 450)
# Footstep "texture" zone: 1-4 kHz — tread scrape, gravel crunch, metal ring
# 2 kHz is the single most critical frequency for footstep detection per community consensus
FOOT_B = (1000, 4000)
# Gunshot bands: 300-5 kHz broadband, peak at 900-1500 Hz
GUN_1  = (300, 1200)
GUN_2  = (1200, 5000)

# Confidence / thresholds
TH_K_FA = 3.0
TH_K_FB = 2.5
TH_K_G  = 3.5
CREST_SHOT_HARD = 8.0        # definite gunshot (sharp transient)
CREST_SHOT_SOFT = 5.0        # probable gunshot (distant / compressed / suppressed)

# Shot-footstep discrimination
SHOT_SUPPRESS_MIN = 0.12     # minimum suppression after any shot (120 ms)
SHOT_SUPPRESS_MAX = 0.35     # maximum suppression for loud/close shots (350 ms, covers indoor reverb)
GUN_FOOT_RATIO   = 3.0       # if gunshot-band energy ≥ 3× footstep-band → reject as gunfire bleed
SPECTRAL_FLAT_TH = 0.55      # spectral flatness above this → broadband (gunshot-like)

# De-duplication: minimum time between emitting the same event type.
# Prevents the same footstep/shot from triggering on consecutive 10ms frames.
FOOT_RETRIGGER = 0.08        # 80 ms — a single footstep impact lasts ~120-170ms
SHOT_RETRIGGER = 0.05        # 50 ms — shots are sharper transients

# Cadence (seconds between steps) — covers tac-sprint (~200ms) through slow walk/ADS (~600ms)
# Source: biomechanics data mapped to CoD movement speeds
# Tac-sprint: ~200-280ms | Sprint: ~250-333ms | Run: ~375-430ms | Walk/ADS: ~500-600ms
CAD_MIN = 0.15
CAD_MAX = 0.60

# UI
UI_FPS = 60
SHOT_DECAY = 0.25
FOOT_DECAY = 0.45

# =========================
#     DSP HELPERS
# =========================

def band_sos(low, high, fs=FS, order=4):
    return butter(order, [low/(fs/2), high/(fs/2)], btype='bandpass', output='sos')

SOS_FA = band_sos(*FOOT_A)
SOS_FB = band_sos(*FOOT_B)
SOS_G1 = band_sos(*GUN_1)
SOS_G2 = band_sos(*GUN_2)

# Narrow sub-band around 2 kHz — the footstep "sweet spot" per community analysis.
# Footsteps peak here; gunshots pass through but don't concentrate here.
# Used as a discriminator: high e_FPEAK / e_FB ratio = likely footstep.
FOOT_PEAK = (1500, 3000)
SOS_FP = band_sos(*FOOT_PEAK)

def ste(x): 
    return float(np.mean(x**2))

def crest(x):
    rms = np.sqrt(ste(x)) + EPS
    return float(np.max(np.abs(x)) / rms)

def gcc_phat(x, y, fs=FS, interp=4):
    n = int(2 ** np.ceil(np.log2(len(x) + len(y))))
    X = np.fft.rfft(x, n)
    Y = np.fft.rfft(y, n)
    R = X * np.conj(Y)
    R /= np.maximum(np.abs(R), EPS)
    cc = np.fft.irfft(R, n*interp)
    max_shift = int(len(x)*interp//2)
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    idx = int(np.argmax(cc))
    tau = (idx - max_shift) / (fs * interp)
    conf = float((cc[idx] - np.mean(cc)) / (np.std(cc) + EPS))
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
    def __init__(self, sensitivity: float = 1.0):
        self.stats = {
            'FA': RollingStats(),
            'FB': RollingStats(),
            'G' : RollingStats(),
        }
        self.recent_foot_times: List[float] = []
        self.last_shot_time: float = -1.0  # sentinel: no shot has occurred yet
        self.last_foot_time: float = -1.0  # for footstep de-duplication
        self.last_shot_suppress: float = SHOT_SUPPRESS_MIN  # adaptive suppression duration
        # Lower K -> more sensitive; scale Ks by 1/sensitivity
        s = max(sensitivity, 1e-3)
        self.k_fa = TH_K_FA / s
        self.k_fb = TH_K_FB / s
        self.k_g  = TH_K_G  / s

        # Persistent filter states — carry IIR state across frames so filters
        # don't restart from zero every 20ms (fixes Band A underestimation for
        # low-frequency footstep components that need multiple cycles to ring up)
        self._zi = {
            'FA_L': np.zeros((SOS_FA.shape[0], 2)),
            'FA_R': np.zeros((SOS_FA.shape[0], 2)),
            'FB_L': np.zeros((SOS_FB.shape[0], 2)),
            'FB_R': np.zeros((SOS_FB.shape[0], 2)),
            'FP_L': np.zeros((SOS_FP.shape[0], 2)),
            'FP_R': np.zeros((SOS_FP.shape[0], 2)),
            'G1_L': np.zeros((SOS_G1.shape[0], 2)),
            'G1_R': np.zeros((SOS_G1.shape[0], 2)),
            'G2_L': np.zeros((SOS_G2.shape[0], 2)),
            'G2_R': np.zeros((SOS_G2.shape[0], 2)),
        }

    def _dir_from_lr(self, srcL, srcR):
        tau, cc = gcc_phat(srcL, srcR)
        tau = float(np.clip(tau, -EAR_DIST/SPEED_SOUND, EAR_DIST/SPEED_SOUND))
        theta_itd = math.asin((SPEED_SOUND * tau) / EAR_DIST)
        ild = 20*math.log10((math.sqrt(ste(srcR))+EPS)/(math.sqrt(ste(srcL))+EPS))
        theta_ild = math.radians(np.clip(ild * 3.0, -90, 90))  # crude ILD→angle slope
        theta = 0.65*theta_itd + 0.35*theta_ild
        conf_dir = float(np.clip(cc/8.0, 0.0, 1.0))
        return theta, conf_dir

    def _spectral_flatness(self, energies: List[float]) -> float:
        """Ratio of geometric mean to arithmetic mean of band energies.
        Close to 1.0 = broadband (gunshot-like), close to 0.0 = narrowband (footstep-like)."""
        arr = np.array([max(e, EPS) for e in energies])
        geo = np.exp(np.mean(np.log(arr)))
        ari = np.mean(arr)
        return float(geo / (ari + EPS))

    def _filt(self, sos, x, key):
        """Apply IIR filter with persistent state across frames."""
        y, self._zi[key] = sosfilt(sos, x, zi=self._zi[key])
        return y

    def detect(self, frame_lr: np.ndarray, t: Optional[float] = None) -> Optional[AudioEvent]:
        """Detect audio events in a stereo frame.
        Args:
            frame_lr: (WIN, 2) numpy array of stereo audio samples.
            t: Optional timestamp in seconds. If None, uses time.monotonic().
               Pass file-position timestamps for offline/file-based processing.
        """
        L, R = frame_lr[:,0], frame_lr[:,1]

        # Bandpass with persistent filter state (avoids IIR ring-up transient each frame)
        FA_L = self._filt(SOS_FA, L, 'FA_L')
        FA_R = self._filt(SOS_FA, R, 'FA_R')
        FB_L = self._filt(SOS_FB, L, 'FB_L')
        FB_R = self._filt(SOS_FB, R, 'FB_R')
        FP_L = self._filt(SOS_FP, L, 'FP_L')
        FP_R = self._filt(SOS_FP, R, 'FP_R')
        G1_L = self._filt(SOS_G1, L, 'G1_L')
        G1_R = self._filt(SOS_G1, R, 'G1_R')
        G2_L = self._filt(SOS_G2, L, 'G2_L')
        G2_R = self._filt(SOS_G2, R, 'G2_R')

        # Energies
        e_FA = ste(np.hstack((FA_L, FA_R)))
        e_FB = ste(np.hstack((FB_L, FB_R)))
        e_FP = ste(np.hstack((FP_L, FP_R)))   # footstep peak sub-band energy
        gun_combined = np.hstack((G1_L+G2_L, G1_R+G2_R))
        e_G  = ste(gun_combined)
        # Crest factor on gun-band signal only — not the raw mix.
        # Raw mix includes music/ambient/UI that inflates RMS and suppresses
        # crest, causing distant gunshots to miss the threshold in-game.
        cf   = crest(gun_combined)

        # Update adaptive floors
        self.stats['FA'].update(e_FA)
        self.stats['FB'].update(e_FB)
        self.stats['G'].update(e_G)

        # Adaptive thresholds
        th_FA = self.stats['FA'].mu + self.k_fa * self.stats['FA'].sigma
        th_FB = self.stats['FB'].mu + self.k_fb * self.stats['FB'].sigma
        th_G  = self.stats['G' ].mu + self.k_g  * self.stats['G' ].sigma

        now = t if t is not None else time.monotonic()

        # --- Shot detection (two-tier: hard and soft crest thresholds) ---
        hard_shot = (e_G > th_G) and (cf >= CREST_SHOT_HARD)
        soft_shot = (e_G > th_G) and (cf >= CREST_SHOT_SOFT) and not hard_shot

        # Spectral flatness across all bands — gunshots excite everything
        sf = self._spectral_flatness([e_FA, e_FB, e_G])

        if hard_shot or (soft_shot and sf > SPECTRAL_FLAT_TH):
            # De-duplicate: don't re-fire if we just emitted a shot on an adjacent frame
            if (now - self.last_shot_time) < SHOT_RETRIGGER:
                return None
            self.last_shot_time = now
            # Clear cadence history — shot reverb contaminates step timing
            self.recent_foot_times.clear()

            srcL, srcR = (G1_L+G2_L), (G1_R+G2_R)
            theta, conf_dir = self._dir_from_lr(srcL, srcR)
            intensity = float(np.clip(math.sqrt(e_G) * 20, 0.0, 1.0))
            confidence = float(np.clip((conf_dir*0.6) + 0.4, 0.0, 1.0))
            # Slightly lower confidence for soft-shot detections
            if soft_shot:
                confidence *= 0.8
            # Adaptive suppression: louder shots get longer suppression (more reverb)
            self.last_shot_suppress = SHOT_SUPPRESS_MIN + (SHOT_SUPPRESS_MAX - SHOT_SUPPRESS_MIN) * intensity
            return AudioEvent('shot', theta, intensity, confidence, now)

        # --- Post-shot suppression: ignore footstep candidates shortly after a shot ---
        if (now - self.last_shot_time) < self.last_shot_suppress:
            return None

        # --- Footstep de-duplication: same footstep spans multiple 10ms frames ---
        if (now - self.last_foot_time) < FOOT_RETRIGGER:
            return None

        # --- Footstep candidate gate ---
        foot_hit = (e_FA > th_FA) or (e_FB > th_FB)
        if not foot_hit:
            return None

        # Cross-band energy ratio guard: if gunshot bands dominate, this is
        # likely gunfire bleed that didn't trigger the shot detector.
        # Use e_FP (1.5-3 kHz, no gun overlap) instead of e_FB (1-4 kHz, partial overlap)
        # for a cleaner comparison against gun-band energy.
        e_foot_clean = 0.4 * e_FA + 0.6 * e_FP
        if e_G > GUN_FOOT_RATIO * e_foot_clean and e_G > th_G * 0.7:
            logging.debug("Rejected footstep candidate: gun-band energy %.2e >> foot-band %.2e", e_G, e_foot_clean)
            return None

        # Spectral flatness guard: broadband energy bursts are not footsteps
        if sf > SPECTRAL_FLAT_TH and cf > CREST_SHOT_SOFT * 0.8:
            logging.debug("Rejected footstep candidate: spectral flatness %.2f with crest %.1f", sf, cf)
            return None

        # 2 kHz peak concentration check: real footsteps concentrate energy in the
        # 1.5-3 kHz sub-band. If Band B is hot but the 2 kHz peak isn't dominant,
        # the energy is likely gunshot bleed spread across 1-5 kHz.
        if e_FB > th_FB and e_FP > EPS:
            peak_ratio = e_FP / (e_FB + EPS)
            # Footsteps: peak_ratio typically > 0.4 (energy concentrated near 2 kHz)
            # Gunshots: peak_ratio typically < 0.3 (energy spread across full 1-5 kHz)
            if peak_ratio < 0.20 and e_G > th_G * 0.5:
                logging.debug("Rejected footstep: low 2kHz peak ratio %.2f (gun-like spread)", peak_ratio)
                return None

        srcL = 0.4*FA_L + 0.6*FB_L
        srcR = 0.4*FA_R + 0.6*FB_R
        theta, conf_dir = self._dir_from_lr(srcL, srcR)

        # Cadence prior — check timing BEFORE appending, so we evaluate against
        # previously confirmed footsteps rather than self-reinforcing marginal detections
        conf_cad = 0.0
        if len(self.recent_foot_times) >= 2:
            d1 = now - self.recent_foot_times[-1]
            d2 = self.recent_foot_times[-1] - self.recent_foot_times[-2]
            good1 = CAD_MIN <= d1 <= CAD_MAX
            good2 = CAD_MIN <= d2 <= CAD_MAX
            conf_cad = 0.25*(1.0 if good1 else 0.0) + 0.25*(1.0 if good2 else 0.0)

        e_foot = 0.4 * e_FA + 0.6 * e_FB
        intensity = float(np.clip(np.sqrt(e_foot) * 20, 0.0, 1.0))
        # Per-band excess ratios — clamp each to [0,1] individually so a non-triggering
        # band doesn't subtract from confidence (fixes metal/concrete footsteps weak in Band A)
        excess_a = float(np.clip((e_FA - th_FA) / (th_FA + EPS), 0, 1))
        excess_b = float(np.clip((e_FB - th_FB) / (th_FB + EPS), 0, 1))
        conf_base = 0.4 * excess_a + 0.6 * excess_b

        # Boost confidence when energy is concentrated around 2 kHz (footstep-like)
        peak_ratio = e_FP / (e_FB + EPS) if e_FB > EPS else 0.0
        conf_peak = float(np.clip(peak_ratio * 0.3, 0.0, 0.15))  # up to +0.15 bonus

        confidence = float(np.clip(0.45*conf_base + 0.25*conf_dir + conf_peak + conf_cad, 0.0, 1.0))

        # Record this footstep for cadence and de-duplication
        self.last_foot_time = now
        self.recent_foot_times.append(now)
        if len(self.recent_foot_times) > 12:
            self.recent_foot_times = self.recent_foot_times[-12:]

        return AudioEvent('footstep', theta, intensity, confidence, now)

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
            # Always use wall-clock time for display decay, not the detector's
            # logical timestamp (which may be file-position in --file mode)
            now = time.monotonic()
            self.markers.append(Marker(evt.cls, evt.theta, evt.intensity, evt.confidence, now, decay))
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
    class CompassOverlay(QtWidgets.QWidget):
        def __init__(self, state: EventState, show_fps=UI_FPS):
            super().__init__(None, QtCore.Qt.Window | QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.Tool)
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
            # Compass ring sized to ~1/3 of the shorter screen dimension
            radius = min(rect.width(), rect.height()) // 6

            # Outer ring
            pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 140), 2)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawEllipse(QtCore.QPointF(cx, cy), radius, radius)

            # Cardinal markers (front/back/left/right)
            marker_len = 12
            for angle, label in [(0, "F"), (180, "B"), (90, "R"), (270, "L")]:
                rad = math.radians(angle)
                ox = cx + radius * math.sin(rad)
                oy = cy - radius * math.cos(rad)
                ix = cx + (radius - marker_len) * math.sin(rad)
                iy = cy - (radius - marker_len) * math.cos(rad)
                painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 180), 2))
                painter.drawLine(QtCore.QPointF(ox, oy), QtCore.QPointF(ix, iy))

            # Draw event markers
            now = time.monotonic()
            tick_len = max(30, radius // 4)
            for m in self.state.get_markers():
                alpha = int(255 * m.alpha(now))
                if alpha <= 0:
                    continue
                is_foot = m.cls == 'footstep'
                color = QtGui.QColor(80, 220, 255, alpha) if is_foot else QtGui.QColor(255, 100, 60, alpha)
                pen_w = 5 if is_foot else 8
                painter.setPen(QtGui.QPen(color, pen_w))
                # Tick from ring edge inward
                ox = cx + radius * math.sin(m.theta)
                oy = cy - radius * math.cos(m.theta)
                ix = cx + (radius - tick_len) * math.sin(m.theta)
                iy = cy - (radius - tick_len) * math.cos(m.theta)
                painter.drawLine(QtCore.QPointF(ox, oy), QtCore.QPointF(ix, iy))
                # Arc span
                arc_span = 16 if is_foot else 24
                painter.setPen(QtGui.QPen(color, 3))
                painter.drawArc(int(cx-radius), int(cy-radius), int(2*radius), int(2*radius),
                                int((90 - math.degrees(m.theta) - arc_span/2) * 16), int(arc_span * 16))
            painter.end()

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
                        p.terminate()
                        return None, None
                else:
                    loopback_device = target_speakers

                logging.info("Found loopback device: (%d) %s",
                           loopback_device['index'], loopback_device['name'])
                return p, loopback_device

            except Exception as e:
                logging.error("Error finding WASAPI device: %s", e)
                try:
                    p.terminate()  # noqa: F821 — p may not be bound if PyAudio() itself failed
                except (NameError, UnboundLocalError, Exception):
                    pass
                return None, None

        def _audio_callback(self, in_data, frame_count, time_info, status):
            """PyAudioWPatch callback for captured loopback audio."""
            if status:
                logging.debug("PyAudioWPatch status: %s", status)

            try:
                # Convert captured loopback audio to numpy array
                audio_data = np.frombuffer(in_data, dtype=np.float32)

                # Reshape based on channels (handle mono, stereo, and surround)
                if frame_count > 0 and len(audio_data) % frame_count == 0:
                    n_channels = len(audio_data) // frame_count
                else:
                    n_channels = 1
                if n_channels <= 1:
                    # Mono - duplicate to stereo
                    audio_data = np.repeat(audio_data.reshape(-1, 1), 2, axis=1)
                elif n_channels == 2:
                    audio_data = audio_data.reshape(-1, 2)
                else:
                    # Surround (5.1, 7.1, etc.) - take first two channels (L/R)
                    audio_data = audio_data.reshape(-1, n_channels)[:, :2]

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
                                                 " (duplicated to stereo — NO directional info)" if ch == 1 else "")
                                    if ch == 1:
                                        logging.warning("Mono capture: all detections will show CENTER direction. Use a stereo device for directional info.")
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

        # --- Fallback chain if primary device fails ---
        def try_stereo_mix() -> bool:
            """Try opening 'Stereo Mix' input device as fallback."""
            if not self.allow_input_fallback:
                return False
            try:
                devs = sd.query_devices()
                stereo_idx = next((i for i, d in enumerate(devs)
                                   if d.get('max_input_channels', 0) > 0 and 'stereo mix' in d['name'].lower()), None)
            except Exception:
                stereo_idx = None
            if stereo_idx is not None and try_open_on_device(stereo_idx):
                logging.warning("Fell back to 'Stereo Mix' input device index %s", stereo_idx)
                return True
            return False

        def try_windows_capture(on_frame_cb) -> bool:
            """Try PyAudioWPatch WASAPI loopback as last-resort fallback."""
            if not WINDOWS_AUDIO_AVAILABLE or self.device_index is None:
                return False
            try:
                caps = sd.query_devices(self.device_index)
                name = caps.get('name', '')
                logging.info("Attempting Windows Core Audio fallback for device: %s", name)
                self.windows_capture = WindowsLoopbackCapture(name, self.fs, self.hop, self.win)
                self.windows_capture.start(on_frame_cb)
                logging.info("Successfully started Windows Core Audio fallback")
                return True
            except Exception as e:
                logging.error("Windows Core Audio fallback failed: %s", e)
                return False

        if not try_open_on_device(dev_out):
            # Fallback: try default WASAPI output device
            try:
                hais = sd.query_hostapis()
                wasapi_idx = next(i for i,h in enumerate(hais) if 'WASAPI' in h['name'].upper())
                dev_default = sd.query_hostapis(wasapi_idx)['default_output_device']
            except Exception:
                dev_default = None

            opened = False
            if dev_default is not None and dev_default != dev_out:
                logging.warning("Primary device failed; trying default WASAPI output device index %s", dev_default)
                opened = try_open_on_device(dev_default)

            # Try pyaudiowpatch WASAPI loopback first — it targets the specific device
            # and works with USB headphones. Stereo Mix is last resort since it only
            # captures from the default onboard sound card, not USB devices.
            if not opened:
                opened = try_windows_capture(on_frame) or try_stereo_mix()

            if not opened:
                raise RuntimeError("Failed to open a loopback stream on the selected device. "
                                   "Run with --list-outputs and choose the '(loopback)' entry or "
                                   "pass --use-loopback-alias; you can also enable --allow-input-fallback.")
            if self.windows_capture:
                return  # capture running via WindowsLoopbackCapture, skip worker thread

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
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
        if self.worker_thread:
            self.worker_thread.join(timeout=1.0)
        if self.windows_capture:
            self.windows_capture.stop()

# =========================
#          MAIN
# =========================

def build_parser():
    p = argparse.ArgumentParser(description="Audio→Visual Compass (COD/Warzone) — No Haptics")
    p.add_argument("--device", type=int, default=None, help="WASAPI output device index to loopback-capture (see --list-devices)")
    p.add_argument("--file", type=str, default=None, help="Play a WAV file through the detector instead of live capture")
    p.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    p.add_argument("--list-outputs", action="store_true", help="List only output devices and exit")
    p.add_argument("--choose-device", action="store_true", help="Interactively list devices and prompt for selection, then start")
    p.add_argument("--no-overlay", action="store_true", help="Disable on-screen compass")
    p.add_argument("--loglevel", default="INFO", help="Logging level (DEBUG, INFO, WARNING)")
    p.add_argument("--min-confidence", type=float, default=0.5, help="Minimum event confidence to display (0..1)")
    p.add_argument("--sensitivity", type=float, default=1.0, help="Detection sensitivity (higher = more sensitive)")
    p.add_argument("--debug-audio", action="store_true", help="Log periodic audio RMS levels to confirm capture")
    p.add_argument("--allow-input-fallback", action="store_true", help="Allow automatic fallback to input devices like 'Stereo Mix' if loopback fails (default: disabled)")
    p.add_argument("--use-loopback-alias", action="store_true", help="Prefer matching '(loopback)' input alias when opening output devices")
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

def make_on_frame(detector, state, args, use_file_time=False):
    """Build the per-frame audio callback. Shared by --choose-device, normal, and --file paths."""
    def on_frame(frame_lr, file_time=None):
        t = file_time if use_file_time else None
        evt = detector.detect(frame_lr, t=t)
        if args.debug_audio:
            # Lightweight RMS meter every ~0.5s
            if not hasattr(on_frame, "_acc"):
                on_frame._acc = 0
                on_frame._t0 = time.monotonic()
            on_frame._acc += float(np.sqrt(np.mean(frame_lr**2)))
            if (time.monotonic() - on_frame._t0) >= 0.5:
                rms = on_frame._acc / max(1, int(0.5/(HOP/FS)))
                logging.info("AUDIO RMS ~ %.4f", rms)
                on_frame._acc = 0
                on_frame._t0 = time.monotonic()
        if evt and evt.confidence >= args.min_confidence:
            state.push(evt)
            logging.debug(f"{evt.cls:8s} az={math.degrees(evt.theta):+05.1f}° I={evt.intensity:.2f} C={evt.confidence:.2f}")
    return on_frame


class FilePlayback:
    """Feed a WAV file through the detector at real-time speed for overlay testing."""

    def __init__(self, file_path: str, fs=FS, hop=HOP, win=WIN):
        import wave

        w = wave.open(file_path, 'r')
        n_ch = w.getnchannels()
        sw = w.getsampwidth()
        wav_fs = w.getframerate()
        n_frames = w.getnframes()
        raw = w.readframes(n_frames)
        w.close()

        # Convert to float32
        if sw == 2:
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sw == 3:
            # 24-bit: pad each 3-byte sample to 4 bytes, then read as int32
            raw_bytes = np.frombuffer(raw, dtype=np.uint8)
            n_samples = len(raw_bytes) // 3
            padded = np.zeros(n_samples * 4, dtype=np.uint8)
            padded[1::4] = raw_bytes[0::3]
            padded[2::4] = raw_bytes[1::3]
            padded[3::4] = raw_bytes[2::3]
            samples = padded.view(np.int32).astype(np.float32) / 2147483648.0
        elif sw == 4:
            samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

        samples = samples.reshape(-1, n_ch)
        if n_ch == 1:
            samples = np.repeat(samples, 2, axis=1)
        elif n_ch > 2:
            samples = samples[:, :2]

        # Resample to target rate if needed
        if wav_fs != fs:
            g = math.gcd(fs, wav_fs)
            up, down = fs // g, wav_fs // g
            samples = np.column_stack([
                resample_poly(samples[:, 0], up, down),
                resample_poly(samples[:, 1], up, down)
            ]).astype(np.float32)
            logging.info("Resampled %d Hz -> %d Hz (%d -> %d samples)", wav_fs, fs, n_frames, len(samples))

        self.samples = samples
        self.fs = fs
        self.hop = hop
        self.win = win
        self.duration = len(samples) / fs
        self.stop_flag = threading.Event()
        self.worker_thread = None
        logging.info("Loaded %s: %.2fs, %d samples at %d Hz", file_path, self.duration, len(samples), fs)

    def start(self, on_frame):
        """Feed frames at real-time speed in a background thread."""
        def worker():
            t_start = time.monotonic()
            for i in range(0, len(self.samples) - self.win, self.hop):
                if self.stop_flag.is_set():
                    break
                frame = self.samples[i:i+self.win]
                file_time = i / self.fs

                # Pace to real-time: wait until we should be at this point
                target = t_start + file_time
                now = time.monotonic()
                if target > now:
                    time.sleep(target - now)

                on_frame(frame, file_time=file_time)

            # Keep running briefly so the last markers can decay visually
            time.sleep(1.0)
            logging.info("File playback complete")

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def stop(self):
        self.stop_flag.set()
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)

def run_event_loop(audio, state, args):
    """Run the overlay (or headless) event loop, then clean up audio on exit."""
    if args.no_overlay or not PYSIDE:
        if not PYSIDE and not args.no_overlay:
            logging.warning("PySide6 not available; running headless (no overlay).")
        print("Running… Press Ctrl+C to quit.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        finally:
            audio.stop()
        return

    app = QtWidgets.QApplication([])
    overlay = CompassOverlay(state)
    try:
        app.exec()
    except KeyboardInterrupt:
        pass
    finally:
        audio.stop()

def main():
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s: %(message)s")

    if args.list_outputs:
        list_output_devices()
        return
    if args.list_devices:
        list_devices()
        return

    # --- File playback mode ---
    if args.file:
        detector = EventDetector(sensitivity=args.sensitivity)
        state = EventState()
        on_frame = make_on_frame(detector, state, args, use_file_time=True)
        playback = FilePlayback(args.file)
        playback.start(on_frame)
        print("Playing %s (%.1fs)... overlay will show detections in real-time." % (args.file, playback.duration))
        run_event_loop(playback, state, args)
        return

    if args.choose_device:
        # Interactive selection with retry if stream open fails
        while True:
            idx = choose_device_interactive()
            if idx is None:
                return
            args.device = idx
            detector = EventDetector(sensitivity=args.sensitivity)
            state = EventState()
            audio = AudioLoop(device_index=args.device, allow_input_fallback=args.allow_input_fallback, force_loopback_alias=args.use_loopback_alias)
            on_frame = make_on_frame(detector, state, args)
            try:
                audio.start(on_frame)
                break  # success
            except Exception as e:
                logging.error("Failed to open device %s: %s", idx, e)
                print("\nCould not open that device. Please choose another output device.\n")
                continue
    else:
        detector = EventDetector(sensitivity=args.sensitivity)
        state = EventState()
        audio = AudioLoop(device_index=args.device, allow_input_fallback=args.allow_input_fallback, force_loopback_alias=args.use_loopback_alias)
        on_frame = make_on_frame(detector, state, args)
        audio.start(on_frame)

    run_event_loop(audio, state, args)

if __name__ == "__main__":
    main()
