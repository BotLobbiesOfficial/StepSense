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

# Bands (COD/Warzone-tuned based on competitive audio analysis)
#
# Key insight: the 3.5-6 kHz region has footstep texture detail but virtually
# no gunfire energy — this is the best discrimination zone.  The overlapping
# 800-2500 Hz region (where both footsteps and gunfire live) is intentionally
# excluded from the footstep bands to eliminate cross-contamination.
#
# Footstep bands — ZERO overlap with gun bands
FOOT_A = (80, 250)        # impact / heel thump (80 Hz floor cuts sub-bass rumble)
FOOT_B = (2500, 6000)     # clarity / texture detail (COD competitive sweet spot ~4 kHz)
# Gunshot bands — ZERO overlap with footstep bands
GUN_LO  = (300, 1200)     # muzzle blast body + small-arms core (peaks 900-1500 Hz)
GUN_HI  = (1200, 2500)    # gunshot crack (narrowed — above 2.5 kHz is footstep territory)

# Confidence / thresholds (sigma multipliers for adaptive noise floor)
TH_K_FA  = 3.0            # footstep low band
TH_K_FB  = 2.5            # footstep high band
TH_K_GLO = 3.5            # gun low band
TH_K_GHI = 3.0            # gun high band
CREST_SHOT = 5.0           # crest factor for shot detection (lowered: game audio is compressed)

# Cadence (seconds between steps)
CAD_MIN = 0.12
CAD_MAX = 0.55             # widened slightly for slower walk speeds

# Shot suppression – blank footstep detection after a shot for this many seconds
SHOT_BLANKING_S = 0.15

# Spectral ratio: if gun-band energy / foot-band energy exceeds this, reject as non-footstep
# With zero-overlap bands this can be tighter than before
GUN_FOOT_RATIO_REJECT = 2.0

# UI
UI_FPS = 60
SHOT_DECAY = 0.25
FOOT_DECAY = 0.45

# =========================
#     DSP HELPERS
# =========================

def band_sos(low, high, fs=FS, order=4):
    return butter(order, [low/(fs/2), high/(fs/2)], btype='bandpass', output='sos')

SOS_FA  = band_sos(*FOOT_A)
SOS_FB  = band_sos(*FOOT_B)
SOS_GLO = band_sos(*GUN_LO)
SOS_GHI = band_sos(*GUN_HI)

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
    peak = cc[idx]
    tau = (idx - max_shift) / (fs * interp)
    conf = float((peak - np.mean(cc)) / (np.std(cc) + EPS))
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
        # Separate rolling stats for each band (gun bands tracked independently)
        self.stats = {
            'FA' : RollingStats(),
            'FB' : RollingStats(),
            'GLO': RollingStats(),
            'GHI': RollingStats(),
        }
        self.recent_foot_times: List[float] = []
        self.last_shot_time: float = 0.0   # for post-shot blanking

        # Lower K -> more sensitive; scale Ks by 1/sensitivity
        s = max(sensitivity, 1e-3)
        self.k_fa  = TH_K_FA  / s
        self.k_fb  = TH_K_FB  / s
        self.k_glo = TH_K_GLO / s
        self.k_ghi = TH_K_GHI / s

        # Persistent filter states (zi) for continuous filtering across frames
        n_sos_fa  = SOS_FA.shape[0]
        n_sos_fb  = SOS_FB.shape[0]
        n_sos_glo = SOS_GLO.shape[0]
        n_sos_ghi = SOS_GHI.shape[0]
        # zi shape: (n_sections, 2) per channel
        self.zi = {
            'FA_L':  np.zeros((n_sos_fa,  2)),  'FA_R':  np.zeros((n_sos_fa,  2)),
            'FB_L':  np.zeros((n_sos_fb,  2)),  'FB_R':  np.zeros((n_sos_fb,  2)),
            'GLO_L': np.zeros((n_sos_glo, 2)),  'GLO_R': np.zeros((n_sos_glo, 2)),
            'GHI_L': np.zeros((n_sos_ghi, 2)),  'GHI_R': np.zeros((n_sos_ghi, 2)),
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

    def _filt(self, sos, x, key):
        """Apply sosfilt with persistent state to avoid frame-boundary transients."""
        y, self.zi[key] = sosfilt(sos, x, zi=self.zi[key])
        return y

    def detect(self, frame_lr: np.ndarray) -> Optional[AudioEvent]:
        L, R = frame_lr[:,0], frame_lr[:,1]

        # ---- Bandpass with persistent filter state ----
        FA_L  = self._filt(SOS_FA,  L, 'FA_L')
        FA_R  = self._filt(SOS_FA,  R, 'FA_R')
        FB_L  = self._filt(SOS_FB,  L, 'FB_L')
        FB_R  = self._filt(SOS_FB,  R, 'FB_R')
        GLO_L = self._filt(SOS_GLO, L, 'GLO_L')
        GLO_R = self._filt(SOS_GLO, R, 'GLO_R')
        GHI_L = self._filt(SOS_GHI, L, 'GHI_L')
        GHI_R = self._filt(SOS_GHI, R, 'GHI_R')

        # ---- Per-band energies (computed correctly: sum energies, not signals) ----
        e_FA  = (ste(FA_L)  + ste(FA_R))  / 2.0
        e_FB  = (ste(FB_L)  + ste(FB_R))  / 2.0
        e_GLO = (ste(GLO_L) + ste(GLO_R)) / 2.0
        e_GHI = (ste(GHI_L) + ste(GHI_R)) / 2.0
        e_G   = e_GLO + e_GHI   # total gun-band energy (sum of band energies)

        # Crest factor on the broadband signal
        cf = crest(np.hstack((L, R)))

        now = time.monotonic()

        # ---- Adaptive noise floor: only update when signal is NOT event-like ----
        # We check if any band is strongly above its current floor; if so, skip update
        # to avoid inflating the noise estimate with event energy.
        fa_snr  = (e_FA  - self.stats['FA'].mu)  / (self.stats['FA'].sigma  + EPS)
        fb_snr  = (e_FB  - self.stats['FB'].mu)  / (self.stats['FB'].sigma  + EPS)
        glo_snr = (e_GLO - self.stats['GLO'].mu) / (self.stats['GLO'].sigma + EPS)
        ghi_snr = (e_GHI - self.stats['GHI'].mu) / (self.stats['GHI'].sigma + EPS)

        is_quiet = max(fa_snr, fb_snr, glo_snr, ghi_snr) < 2.0
        if is_quiet:
            self.stats['FA'].update(e_FA)
            self.stats['FB'].update(e_FB)
            self.stats['GLO'].update(e_GLO)
            self.stats['GHI'].update(e_GHI)

        # ---- Adaptive thresholds ----
        th_FA  = self.stats['FA'].mu  + self.k_fa  * self.stats['FA'].sigma
        th_FB  = self.stats['FB'].mu  + self.k_fb  * self.stats['FB'].sigma
        th_GLO = self.stats['GLO'].mu + self.k_glo * self.stats['GLO'].sigma
        th_GHI = self.stats['GHI'].mu + self.k_ghi * self.stats['GHI'].sigma

        # ---- Gunshot detection ----
        # A shot needs: (a) elevated energy in EITHER gun band, AND (b) high crest factor
        gun_energy_hit = (e_GLO > th_GLO) or (e_GHI > th_GHI)
        shot_like = gun_energy_hit and (cf >= CREST_SHOT)

        # Also classify as shot-like if crest is very high (>= 1.5x threshold) even if
        # gun band energy is only moderately above noise — catches suppressed/distant shots
        if cf >= CREST_SHOT * 1.5 and e_G > (self.stats['GLO'].mu + self.stats['GHI'].mu) * 1.5:
            shot_like = True

        if shot_like:
            self.last_shot_time = now
            srcL = GLO_L + GHI_L
            srcR = GLO_R + GHI_R
            theta, conf_dir = self._dir_from_lr(srcL, srcR)
            intensity = float(min(1.0, math.sqrt(e_G) * 60))
            confidence = float(np.clip((conf_dir*0.6) + 0.4, 0.0, 1.0))
            return AudioEvent('shot', theta, intensity, confidence, now)

        # ---- Post-shot blanking: suppress footstep detection shortly after a gunshot ----
        if (now - self.last_shot_time) < SHOT_BLANKING_S:
            return None

        # ---- Footstep detection ----
        foot_hit_A = e_FA > th_FA
        foot_hit_B = e_FB > th_FB
        if not (foot_hit_A or foot_hit_B):
            return None

        # ---- Spectral ratio rejection: reject if gun-band energy dominates ----
        # Footsteps have most energy in FOOT_A (60-250 Hz); gunfire is broadband.
        e_foot_total = e_FA + e_FB + EPS
        if e_G / e_foot_total > GUN_FOOT_RATIO_REJECT:
            logging.debug("Rejected footstep: gun/foot ratio %.1f > %.1f",
                          e_G / e_foot_total, GUN_FOOT_RATIO_REJECT)
            return None

        # Additional rejection: if crest factor is high (sharp transient) and gun bands
        # are elevated, this is likely a gunshot that didn't quite meet the shot threshold
        if cf >= CREST_SHOT * 0.7 and gun_energy_hit:
            logging.debug("Rejected footstep: borderline shot (crest=%.1f, gun_hit=True)", cf)
            return None

        # ---- Direction estimation ----
        # Weight source signal toward whichever footstep band triggered
        if foot_hit_A and foot_hit_B:
            srcL = 0.5*FA_L + 0.5*FB_L
            srcR = 0.5*FA_R + 0.5*FB_R
        elif foot_hit_A:
            srcL, srcR = FA_L, FA_R
        else:
            srcL, srcR = FB_L, FB_R
        theta, conf_dir = self._dir_from_lr(srcL, srcR)

        # ---- Confidence calculation (fixed: no negative terms) ----
        # Compute per-band excess ratio, clamped to [0, inf) before weighting
        excess_A = max(0.0, (e_FA - th_FA) / (th_FA + EPS)) if foot_hit_A else 0.0
        excess_B = max(0.0, (e_FB - th_FB) / (th_FB + EPS)) if foot_hit_B else 0.0
        # Weight toward whichever band(s) actually triggered
        if foot_hit_A and foot_hit_B:
            conf_base = float(np.clip(0.4*excess_A + 0.6*excess_B, 0.0, 1.0))
        elif foot_hit_A:
            conf_base = float(np.clip(excess_A, 0.0, 1.0))
        else:
            conf_base = float(np.clip(excess_B, 0.0, 1.0))

        # ---- Cadence prior (only update history AFTER passing all rejection checks) ----
        conf_cad = 0.0
        if len(self.recent_foot_times) >= 2:
            d1 = now - self.recent_foot_times[-1]
            d2 = self.recent_foot_times[-1] - self.recent_foot_times[-2]
            good1 = CAD_MIN <= d1 <= CAD_MAX
            good2 = CAD_MIN <= d2 <= CAD_MAX
            conf_cad = 0.2*(1.0 if good1 else 0.0) + 0.15*(1.0 if good2 else 0.0)
            # Bonus: if both intervals are consistent (similar duration), extra confidence
            if good1 and good2 and d2 > 0:
                ratio = d1 / d2
                if 0.6 <= ratio <= 1.67:  # intervals within ~60% of each other
                    conf_cad += 0.15

        # Now append to cadence history (only after all rejection gates passed)
        self.recent_foot_times.append(now)
        if len(self.recent_foot_times) > 12:
            self.recent_foot_times = self.recent_foot_times[-12:]

        intensity = float(np.clip(np.sqrt(0.4*e_FA + 0.6*e_FB) * 70, 0.0, 1.0))
        confidence = float(np.clip(0.45*conf_base + 0.25*conf_dir + conf_cad, 0.0, 1.0))
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
            self._stream_channels = 2  # actual channel count of the opened stream

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
                ch = self._stream_channels

                # Reshape to (frames, channels) using known channel count
                audio_data = audio_data.reshape(-1, ch)

                if ch == 1:
                    # Mono -> duplicate to stereo
                    audio_data = np.repeat(audio_data, 2, axis=1)
                elif ch > 2:
                    # Multi-channel (5.1/7.1) -> take first two channels (L/R)
                    audio_data = audio_data[:, :2].copy()

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
                self._stream_channels = int(device_info['maxInputChannels'])
                self.stream = self.pyaudio_instance.open(
                    format=pyaudio_wpatch.paFloat32,
                    channels=self._stream_channels,
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
    p = argparse.ArgumentParser(description="Audio→Visual Compass (COD/Warzone) — No Haptics")
    p.add_argument("--device", type=int, default=None, help="WASAPI output device index to loopback-capture (see --list-devices)")
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

def main():
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s: %(message)s")

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

            def on_frame(frame_lr):
                evt = detector.detect(frame_lr)
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
                    lvl = logging.INFO if args.debug_audio else logging.DEBUG
                    logging.log(lvl, "%8s az=%+05.1f° I=%.2f C=%.2f",
                                evt.cls, math.degrees(evt.theta), evt.intensity, evt.confidence)
            try:
                audio.start(on_frame)
                break  # success
            except Exception as e:
                logging.error("Failed to open device %s: %s", idx, e)
                print("\nCould not open that device. Please choose another output device.\n")
                continue
        # proceed with overlay/headless using objects created above
        # Overlay or headless
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
        return
    elif args.list_outputs:
        list_output_devices()
        return
    elif args.list_devices:
        list_devices()
        return

    detector = EventDetector(sensitivity=args.sensitivity)
    state = EventState()
    audio = AudioLoop(device_index=args.device, allow_input_fallback=args.allow_input_fallback, force_loopback_alias=args.use_loopback_alias)

    def on_frame(frame_lr):
        evt = detector.detect(frame_lr)
        if args.debug_audio:
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
            lvl = logging.INFO if args.debug_audio else logging.DEBUG
            logging.log(lvl, "%8s az=%+05.1f° I=%.2f C=%.2f",
                        evt.cls, math.degrees(evt.theta), evt.intensity, evt.confidence)

    # Start audio
    audio.start(on_frame)

    # Overlay or headless
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

    # Qt overlay loop
    app = QtWidgets.QApplication([])
    overlay = CompassOverlay(state)
    try:
        app.exec()
    except KeyboardInterrupt:
        pass
    finally:
        audio.stop()

if __name__ == "__main__":
    main()
