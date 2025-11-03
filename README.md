# Moodify

Smart desk lamp: webcam-driven emotion detection on your computer, smooth ambient light on an ESP32, and music played locally on your laptop.

## Overview
- Brain (Python): Captures webcam frames, runs DeepFace emotion detection, stabilizes predictions via a 50-frame buffer with a strict-majority rule, calls the ESP32 over serial/HTTP when the stable emotion changes, and plays background music locally or via Spotify with smooth fades.
- Actuator (ESP32): Receives commands over Serial (USB) or optional HTTP (/mood?emotion=...), and smoothly blends LED colors using Adafruit_NeoPixel.

Supported emotions: `happy`, `sad`, `angry`, `neutral`, `fear` (stress proxy), `disgust`, `surprise`.

## Hardware
- ESP32 dev board
- WS2812B LED strip (NeoPixel)

### Wiring (example)
- WS2812B data -> ESP32 GPIO 23 (configurable in sketch)
- WS2812B 5V and GND -> suitable 5V supply; share GND with ESP32

## ESP32 Setup
1. Open `mood_actuator.ino` in Arduino IDE / PlatformIO.
2. Install libraries: Adafruit NeoPixel (ESP32 core provides Serial and optional WiFi/WebServer).
3. Set your Wi-Fi credentials at the top of the sketch.
4. Flash to ESP32. Open Serial Monitor to see the assigned IP.
5. No DFPlayer needed. Music is played on the host computer.

## Python Setup
Requires Python 3.8+.

Install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional: create a .env file for configuration (loaded automatically):

```
# .env
ESP32_IP="http://192.168.1.105"

SPOTIFY_CLIENT_ID="..."
SPOTIFY_CLIENT_SECRET="..."
SPOTIFY_REDIRECT_URI="http://127.0.0.1:8888/callback"
SPOTIFY_DEVICE="MacBook"

SP_HAPPY="spotify:playlist:..."
SP_SAD="spotify:playlist:..."
SP_ANGRY="spotify:playlist:..."
SP_NEUTRAL="spotify:playlist:..."
SP_FEAR="spotify:playlist:..."
SP_DISGUST="spotify:playlist:..."
SP_SURPRISE="spotify:playlist:..."
```

All values can still be overridden via command-line flags.

### Custom training (optional)
You can train a per-user classifier on top of DeepFace embeddings using your own images.

1) Prepare a dataset directory with one folder per emotion (must be canonical names):

```
dataset/
	happy/    img001.jpg img002.png ...
	sad/      ...
	angry/    ...
	neutral/  ...
	fear/     ...
	disgust/  ...
	surprise/ ...
```

2) Train the classifier and save it:

```bash
source .venv/bin/activate
python custom_emotion_trainer.py --data-dir ./dataset --model-out custom_emotions.pkl --embedding-model Facenet512 --algo logreg --detector-backend opencv
```

3) Use the classifier in Moodify:

```bash
python mood_detector.py --serial-port /dev/cu.SLAB_USBtoUART --display \
	--custom-classifier ./custom_emotions.pkl --embedding-model Facenet512
```

Notes:
- Only canonical labels are allowed: happy, sad, angry, neutral, fear, disgust, surprise
- You can adjust the stability window (`--buffer-len`) and strict-majority threshold (`--majority-frac`).
- If you skip `--custom-classifier`, Moodify uses the built-in DeepFace emotion head.
 - You can also pick the engine and detector from the Desktop app under Run → Detection and Training → Training options.

Prepare music files on your laptop (MP3/OGG). Place them under `./media/mp3/` with these names:

```
media/mp3/
	0001_happy.mp3
	0002_sad.mp3
	0003_angry.mp3
	0004_neutral.mp3
	0005_fear.mp3
	0006_disgust.mp3
	0007_surprise.mp3
```

Run the Brain (Wi‑Fi mode, replace IP as needed):
```bash
python mood_detector.py --esp32-ip http://<ESP32_IP>
```

Cable‑only mode (USB serial):
```bash
python mood_detector.py --serial-port /dev/cu.SLAB_USBtoUART --display --music-dir ./media/mp3
```
Replace `/dev/cu.SLAB_USBtoUART` with your actual device; on macOS with CP210x it usually matches that name. Use `--baud` to change from the default 115200.

Optional flags:
- `--display` to show the webcam preview with overlay text.
- `--camera-index` to select a different camera.
- `--buffer-len` to adjust stability window (default 50). Set to 0 to auto-size to ~2.5s based on measured FPS.
- `--majority-frac` to require a minimum fraction for the winner (default 0.5 = strict >50% majority). If the top emotion doesn't reach the threshold or there's a tie, the system holds the previous state.
- `--hold-seconds` to enforce a cooldown minimum time before switching to a new stable emotion (prevents flicker).
- `--no-face-grace` to wait a short grace period after losing the face before switching to neutral/pause.
- `--music-dir` to point to a custom folder with tracks (default ./media/mp3).
- `--no-music` to disable local music playback.
 - `--custom-classifier` to use a trained classifier (.pkl) on DeepFace embeddings.
 - `--embedding-model` to pick the embedding backbone used with the custom classifier (default Facenet512).
 - `--emotion-engine {auto|deepface|custom}` to choose between the built‑in DeepFace CNN head, your custom classifier, or auto (use custom when provided; default auto).
 - `--detector-backend {opencv|retinaface|mtcnn|ssd|dlib}` to select the face detector passed to DeepFace (default opencv). RetinaFace is accurate but heavier; OpenCV is fastest.
	 Note: mediapipe is intentionally excluded from requirements due to protobuf/TensorFlow version conflicts.

## Usage
- Start the ESP32 first; note its IP.
- Run the Python Brain with the ESP32 IP.
- The system will stabilize on a dominant emotion and send a command when it changes.

## Troubleshooting
- First DeepFace run downloads models; allow time.
- If detection is jumpy, ensure good lighting and keep the camera steady.
- If HTTP sends fail, verify ESP32 IP, that both devices are on the same LAN, and that the ESP32 Serial log shows requests.
- If music doesn't play on the laptop, ensure the files exist at the configured `--music-dir` path and that your system audio device is available. MP3 support depends on your pygame/SDL build; if MP3 fails, convert to OGG.
- If LEDs flicker, add a level shifter and a 300–500 ohm resistor on the data line, and power the strip adequately.

## License
MIT
