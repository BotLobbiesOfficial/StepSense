# BotLobbies StepSense - Audio Compass

## Overview
Python-based audio compass for detecting directional footsteps and gunshots in games like Call of Duty/Warzone. Uses real-time audio analysis to provide visual directional alerts showing where sounds are coming from.

## Key Features
- Real-time WASAPI loopback audio capture (USB and 3.5mm headphones)
- **Improved footstep vs gunfire discrimination** using multi-feature analysis
- Directional audio analysis using GCC-PHAT and ILD algorithms
- Three overlay styles: Edge indicators, HUD compass, or Ring overlay
- Confidence scoring with cadence detection for footsteps
- Debug modes for audio monitoring and tuning

## Overlay Styles

### Edge Overlay (Recommended)
Screen-edge arrows/chevrons pointing to sound sources. Minimal screen obstruction.
```bash
python "BotLobbies StepSense.py" --device 13 --overlay edge
```

### HUD Compass
Small radar-style compass in a screen corner. Shows sounds as dots on a mini-map.
```bash
python "BotLobbies StepSense.py" --device 13 --overlay hud --hud-position bottom-right
```

### Ring Overlay
Original full-screen radial ring around screen center.
```bash
python "BotLobbies StepSense.py" --device 13 --overlay ring
```

## Detection Algorithm

### Footstep vs Gunfire Discrimination
The detector uses multiple features to distinguish footsteps from gunfire:

1. **Frequency Bands**
   - Footsteps: Low (40-180 Hz), Mid (250-800 Hz), High (2000-4500 Hz)
   - Gunfire: Low (80-400 Hz), Mid (400-2000 Hz), High (2000-8000 Hz)

2. **Spectral Features**
   - Crest factor: Gunfire has high peak-to-RMS ratio
   - Spectral flatness: Gunfire is more broadband, footsteps more tonal
   - Attack time: Gunfire < 3ms, footsteps > 15ms

3. **Temporal Features**
   - Cadence detection: Regular footstep intervals (0.18-0.55s) boost confidence
   - Cooldown prevents rapid re-triggering

### Direction Calculation
- **ITD (Interaural Time Difference)**: GCC-PHAT cross-correlation with parabolic interpolation
- **ILD (Interaural Level Difference)**: dB difference between left/right channels
- **Fusion**: 70% ITD + 30% ILD for robust direction estimation

## Usage

### List Available Devices
```bash
python "BotLobbies StepSense.py" --list-outputs
```

### Basic Usage (Edge Overlay - Default)
```bash
python "BotLobbies StepSense.py" --device 13
```

### Footsteps Only Mode
```bash
python "BotLobbies StepSense.py" --device 13 --footsteps-only
```

### Debug Mode (For Tuning)
```bash
python "BotLobbies StepSense.py" --device 13 --debug-audio --debug-detection --loglevel DEBUG
```

### Interactive Device Selection
```bash
python "BotLobbies StepSense.py" --choose-device
```

## Command Line Options

### Device Selection
- `--device N` - Audio device index (use --list-outputs to find)
- `--list-outputs` - List output devices with WASAPI info
- `--list-devices` - List all audio devices
- `--choose-device` - Interactive device selection

### Overlay Options
- `--overlay [edge|hud|ring|none]` - Overlay style (default: edge)
- `--hud-position [bottom-right|bottom-left|top-right|top-left|center-bottom]` - HUD position
- `--no-overlay` - Disable overlay (same as --overlay none)

### Detection Tuning
- `--sensitivity X.X` - Detection sensitivity multiplier (default: 1.0, higher = more sensitive)
- `--min-confidence X.X` - Minimum confidence threshold 0.0-1.0 (default: 0.4)
- `--footsteps-only` - Only show footstep detections
- `--gunfire-only` - Only show gunfire detections

### Debug Options
- `--debug-audio` - Log periodic RMS levels
- `--debug-detection` - Log detailed detection analysis (gunfire scoring)
- `--loglevel [DEBUG|INFO|WARNING]` - Logging verbosity

### Audio Fallback Options
- `--allow-input-fallback` - Allow fallback to 'Stereo Mix' if loopback fails
- `--use-loopback-alias` - Prefer '(loopback)' input alias devices

## Dependencies
```bash
pip install numpy scipy sounddevice PySide6 pyaudiowpatch
```

- `numpy` - Audio processing
- `scipy` - Signal processing and filtering
- `sounddevice` - Primary audio interface
- `PySide6` - Overlay GUI
- `pyaudiowpatch` - USB headphone WASAPI loopback support

## USB Headphone Support
Uses PyAudioWPatch for WASAPI loopback capture from USB audio devices:
- Automatic loopback device discovery
- Fallback when standard sounddevice fails
- Works with USB headsets that don't expose standard loopback

## Tuning Tips

### Too Many False Positives
- Increase `--min-confidence` (try 0.5 or 0.6)
- Decrease `--sensitivity` (try 0.8)

### Missing Footsteps
- Increase `--sensitivity` (try 1.5 or 2.0)
- Decrease `--min-confidence` (try 0.3)

### Wrong Directions
- Ensure stereo audio is properly configured
- Check that headphones are on correct ears (L/R)
- Use `--debug-detection` to see direction confidence

## Visual Indicators
- **Cyan/Blue**: Footstep detections
- **Orange/Red**: Gunfire detections
- **Brighter = Higher confidence/intensity**
- **Direction**: 0° = front, +90° = right, -90° = left, ±180° = behind
