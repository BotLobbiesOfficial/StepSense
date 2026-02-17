# BotLobbies StepSense - Audio Compass

## Overview
Python-based audio compass for detecting directional footsteps and gunshots in games like Call of Duty/Warzone. Uses real-time audio analysis to provide azimuth (direction) readings of audio events.

## Key Features
- Real-time WASAPI loopback audio capture from USB headphones
- Directional audio analysis using GCC-PHAT and ILD algorithms
- Bandpass filtering for footstep/gunshot frequency isolation
- Confidence scoring for detection accuracy
- Debug modes for audio monitoring and calibration

## USB Headphone Audio Capture Solution

### Problem Solved
Originally, the script failed with USB headphones due to "Invalid number of channels [PaErrorCode -9998]" errors. Standard Python audio libraries (sounddevice, PyAudio) cannot capture system audio output from USB devices using WASAPI loopback.

### Solution: PyAudioWPatch Integration
Added PyAudioWPatch library support to enable proper WASAPI loopback capture from USB audio devices.

#### Key Implementation Changes:
1. **PyAudioWPatch Library**: Installed `pyaudiowpatch` - a PyAudio fork with Windows WASAPI loopback support
2. **WindowsLoopbackCapture Class**: New class specifically for PyAudioWPatch-based audio capture
3. **Loopback Device Discovery**: Proper detection of `[Loopback]` devices using `get_loopback_device_info_generator()`
4. **Fallback Integration**: WindowsLoopbackCapture used as fallback when standard sounddevice fails

#### Technical Details:
- Uses `pyaudiowpatch.PyAudio()` instead of standard PyAudio
- Finds loopback devices with `p.get_loopback_device_info_generator()`
- Opens streams using `maxInputChannels` for loopback devices
- No special parameters needed - standard PyAudio `open()` method works with loopback devices

## Usage

### Basic Usage
```bash
python "BotLobbies StepSense.py" --device 13
```

### Debug Mode (Recommended for Testing)
```bash
python "BotLobbies StepSense.py" --device 13 --debug-audio --sensitivity 2.0 --min-confidence 0.3 --loglevel DEBUG
```

### List Available Devices
```bash
python "BotLobbies StepSense.py" --list-outputs
```

## Testing the Audio Compass

1. **Start with debug mode** to see RMS levels and detections
2. **Play stereo audio** (YouTube videos, games)
3. **Look for**:
   - Changing RMS values: `AUDIO RMS ~ 0.0052`
   - Footstep detections: `footstep az=+03.2° I=1.00 C=0.50`
   - Directional accuracy: Left = negative angles (-), Right = positive angles (+)

## Dependencies
- `numpy` - Audio processing
- `sounddevice` - Primary audio interface
- `pyaudiowpatch` - USB headphone WASAPI loopback support
- `scipy` - Signal processing and filtering

## Command Line Options
- `--device N` - Audio device index (use --list-outputs to find)
- `--debug-audio` - Show RMS levels and detection details
- `--sensitivity X.X` - Detection sensitivity (default 1.0, higher = more sensitive)
- `--min-confidence X.X` - Minimum confidence threshold (default 0.5)
- `--loglevel DEBUG` - Verbose logging

## Audio Compass Theory
Uses cross-correlation (GCC-PHAT) and Interaural Level Difference (ILD) to determine the azimuth angle of audio sources. Filters audio into separate bands — footstep low (60-250 Hz), footstep high (800-3000 Hz), gun low (200-900 Hz), gun high (1500-6000 Hz) — with spectral ratio discrimination, crest factor analysis, and post-shot blanking to separate footsteps from gunfire. Adaptive noise floor tracking (updated only during quiet frames) provides robust threshold adaptation.

## Notes
- Designed specifically for Windows WASAPI
- Requires USB headphones or WASAPI-compatible output devices
- PyAudioWPatch enables capture of actual system audio output, not microphone input
- Loopback devices appear as virtual input devices with `[Loopback]` suffix