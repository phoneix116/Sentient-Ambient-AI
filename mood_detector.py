#!/usr/bin/env python3
"""
Moodify - Brain (Host Computer)

Captures webcam frames, uses DeepFace to detect emotion, stabilizes predictions
with a fixed-size deque buffer (statistical mode), and sends commands to an ESP32
Actuator over HTTP whenever the stable emotion changes.

Requirements:
- Python 3.8+
- opencv-python
- deepface
- requests

Usage:
- Update the ESP32 IP address via CLI arg `--esp32-ip` or env var `ESP32_IP`.
- Run the script; press 'q' to quit the preview window.

Notes:
- DeepFace will download models on first run; allow some time.
- Supports the full DeepFace emotion set (happy, sad, angry, neutral, fear,
    disgust, surprise).
"""

import os
import cv2
import time
import argparse
import collections
import threading
import statistics
import requests
import queue
from typing import Deque, Optional, Dict
import math
from collections import Counter

# Optional .env support (configuration file)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Optional serial support
try:
    import serial  # type: ignore
except Exception:
    serial = None  # will handle at runtime

# Optional Spotify support
try:
    import spotipy  # type: ignore
    from spotipy.oauth2 import SpotifyOAuth  # type: ignore
    from spotipy.exceptions import SpotifyException  # type: ignore
except Exception:
    spotipy = None
    SpotifyOAuth = None
    SpotifyException = None

try:
    from deepface import DeepFace
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Failed to import deepface. Please install dependencies from requirements.txt"
    ) from e

# Canonical set of emotions the actuator supports
CANON_EMOTIONS = {"happy", "sad", "angry", "neutral", "fear", "disgust", "surprise"}

# Map DeepFace raw emotions to our canonical set
RAW_TO_CANON = {
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "neutral": "neutral",
    "fear": "fear",  # used as proxy for stress
    "disgust": "disgust",
    "surprise": "surprise",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Moodify Brain - Emotion to ESP32 bridge")
    parser.add_argument(
        "--esp32-ip",
        default=os.getenv("ESP32_IP", "http://192.168.1.105"),
        help="Base URL for ESP32 (e.g., http://192.168.1.105)",
    )
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index (default 0)")
    parser.add_argument("--buffer-len", type=int, default=50, help="Stability buffer size (default 50)")
    parser.add_argument("--hold-seconds", type=float, default=5.0, help="Minimum time to hold a stable emotion before allowing a change")
    parser.add_argument("--no-face-grace", type=float, default=1.5, help="Seconds to wait after losing face before applying no-face behavior")
    parser.add_argument(
        "--majority-frac",
        type=float,
        default=0.5,
        help="Required fraction for a winner within the buffer. Default 0.5 means strict majority (>50%%).",
    )
    parser.add_argument("--min-interval", type=float, default=1.0, help="Min seconds between HTTP sends")
    parser.add_argument("--display", action="store_true", help="Show preview window with overlay")
    parser.add_argument("--width", type=int, default=640, help="Camera width")
    parser.add_argument("--height", type=int, default=480, help="Camera height")
    parser.add_argument(
        "--video-backend",
        choices=["auto", "default", "dshow", "msmf", "avfoundation", "v4l2"],
        default=os.getenv("MOODIFY_VIDEO_BACKEND", "auto").lower(),
        help="OpenCV video API backend to use for the webcam. 'auto' tries sensible fallbacks per OS (Windows: dshow→msmf→default).",
    )
    parser.add_argument(
        "--no-face-behavior",
        choices=["neutral", "pause"],
        default=os.getenv("NO_FACE_BEHAVIOR", "neutral").lower(),
        help="What to do when no face is stably detected: switch to 'neutral' (default) or 'pause' the music.",
    )
    # Local music playback controls
    parser.add_argument(
        "--music-dir",
        default=os.getenv("MOODIFY_MUSIC_DIR", os.path.join(os.getcwd(), "media", "mp3")),
        help="Directory containing emotion tracks (default ./media/mp3)",
    )
    parser.add_argument("--no-music", action="store_true", help="Disable local music playback")
    # Spotify controls (optional; requires Premium)
    parser.add_argument("--spotify", action="store_true", help="Use Spotify instead of local files")
    parser.add_argument("--spotify-device", default=os.getenv("SPOTIFY_DEVICE"), help="Substring of target device name (e.g., 'MacBook')")
    parser.add_argument("--spotify-client-id", default=os.getenv("SPOTIFY_CLIENT_ID"))
    parser.add_argument("--spotify-client-secret", default=os.getenv("SPOTIFY_CLIENT_SECRET"))
    parser.add_argument(
        "--spotify-redirect-uri",
        default=os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback"),
    )
    # Per-emotion playlist/album/track URIs or URLs (env-overridable via .env)
    parser.add_argument("--sp-happy", default=os.getenv("SP_HAPPY"))
    parser.add_argument("--sp-sad", default=os.getenv("SP_SAD"))
    parser.add_argument("--sp-angry", default=os.getenv("SP_ANGRY"))
    parser.add_argument("--sp-neutral", default=os.getenv("SP_NEUTRAL"))
    parser.add_argument("--sp-fear", default=os.getenv("SP_FEAR"))
    parser.add_argument("--sp-disgust", default=os.getenv("SP_DISGUST"))
    parser.add_argument("--sp-surprise", default=os.getenv("SP_SURPRISE"))
    # Serial controls (cable-only mode)
    parser.add_argument("--serial-port", default=None, help="Serial device for ESP32 (e.g., /dev/cu.SLAB_USBtoUART)")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate (default 115200)")
    parser.add_argument("--serial-log", action="store_true", help="Print ESP32 serial output while connected")
    # Emotion analysis engine and detector backend
    parser.add_argument(
        "--emotion-engine",
        choices=["auto", "deepface", "custom"],
        default=os.getenv("MOODIFY_ENGINE", "auto").lower(),
        help="Which engine to use: 'custom' (your trained model), 'deepface' (built-in CNN), or 'auto' (use custom if provided).",
    )
    parser.add_argument(
        "--detector-backend",
        choices=["opencv", "retinaface", "mediapipe", "mtcnn", "ssd", "dlib"],
        default=os.getenv("MOODIFY_DETECTOR", "opencv").lower(),
        help="Face detector backend passed to DeepFace (default opencv).",
    )
    # Custom classifier on embeddings (optional)
    parser.add_argument("--custom-classifier", default=os.getenv("MOODIFY_CUSTOM_MODEL"), help="Path to a joblib .pkl trained by custom_emotion_trainer.py")
    parser.add_argument("--embedding-model", default=os.getenv("MOODIFY_EMBEDDING_MODEL", "Facenet512"), help="DeepFace embedding backbone for custom classifier (e.g., Facenet512)")
    # Optional simple control file for GUI commands (pause/play/volume)
    parser.add_argument("--control-file", default=None, help="Path to a JSON control file written by the desktop app for pause/play/volume")
    return parser.parse_args()


def _resolve_detector_backend(detector_backend: str) -> str:
    """Return a compatible detector backend, falling back when needed.

    mediapipe 0.10.x conflicts with protobuf >=4/5 (required by TF 2.20+).
    If that conflict is detected, we fall back to opencv and log a warning.
    """
    backend = detector_backend
    if backend == "mediapipe":
        try:
            from google.protobuf import __version__ as pb_ver  # type: ignore
            major = int(str(pb_ver).split(".")[0])
            if major >= 4:
                print("[warn] mediapipe backend incompatible with protobuf>=4 (TF 2.20 uses protobuf>=5). Falling back to opencv.")
                backend = "opencv"
            else:
                # ensure mediapipe import succeeds
                __import__("mediapipe")
        except Exception as e:
            print(f"[warn] mediapipe backend unavailable ({e}). Falling back to opencv.")
            backend = "opencv"
    return backend


def analyze_emotion(frame, predictor=None, embedding_model: str = "Facenet512", detector_backend: str = "opencv") -> Optional[str]:
    """Return canonical emotion from a frame using DeepFace.

    Returns one of CANON_EMOTIONS, or the string 'no_face' if no face was
    confidently detected by the model. Returns None only on unexpected
    processing errors.
    """
    # If a custom predictor is provided, use embedding + classifier path
    # Resolve detector backend compatibility once per call (cheap)
    _detector = _resolve_detector_backend(detector_backend)
    if _detector != detector_backend:
        # Log downgrade only occasionally would be nicer, but keep simple here
        try:
            print(f"[info] detector backend '{detector_backend}' -> using '{_detector}'")
        except Exception:
            pass

    # Build a small fallback chain of detector backends to improve robustness
    primary = _resolve_detector_backend(detector_backend)
    # Try strong→fast: retinaface, opencv, mtcnn (ensure uniqueness and keep primary first)
    candidates = [primary] + [b for b in ["retinaface", "opencv", "mtcnn"] if b != primary]
    backends_to_try = []
    for b in candidates:
        if b not in backends_to_try:
            backends_to_try.append(b)

    if predictor is not None:
        try:
            emb = None
            used_be = None
            for be in backends_to_try:
                try:
                    reps = DeepFace.represent(
                        img_path=frame,
                        model_name=embedding_model,
                        enforce_detection=False,
                        detector_backend=be,
                        align=True,
                    )
                    if isinstance(reps, list) and reps:
                        emb = reps[0].get("embedding")
                    elif isinstance(reps, dict):
                        emb = reps.get("embedding")
                    if emb is not None:
                        used_be = be
                        break
                except Exception:
                    emb = None
                    continue
            if used_be and used_be != primary:
                try:
                    print(f"[info] detector fallback: '{primary}' -> '{used_be}' (custom engine)")
                except Exception:
                    pass
            if isinstance(reps, list) and reps:
                emb = reps[0].get("embedding")
            elif isinstance(reps, dict):
                emb = reps.get("embedding")
            else:
                return "no_face"
            if emb is None:
                return "no_face"
            import numpy as _np  # local import to avoid global hard dep for non-custom path
            arr = _np.asarray(emb, dtype=_np.float32).reshape(1, -1)
            # Validate embedding dimension if saved in predictor
            try:
                exp_dim = int(predictor.get("embedding_dim", 0))
                if exp_dim and arr.shape[1] != exp_dim:
                    print(f"[error] embedding dimension mismatch: expected {exp_dim}, got {arr.shape[1]} (embedding={embedding_model})")
            except Exception:
                pass
            pred = predictor["pipeline"].predict(arr)[0]
            label = predictor["labels"][int(pred)] if hasattr(pred, "__int__") else str(pred)
            label_l = label.lower()
            if label_l in CANON_EMOTIONS:
                return label_l
            return RAW_TO_CANON.get(label_l, "no_face")
        except Exception as e:
            # Surface common mismatch when embedding dims don't match the trained pipeline
            try:
                msg = str(e)
                if "features" in msg or "shape" in msg or "Found array with" in msg:
                    print(f"[error] custom classifier failed: {msg} (embedding={embedding_model})")
            except Exception:
                pass
            return "no_face"
    # Else, fallback to DeepFace built-in emotion head
    try:
        analysis = None
        used_be = None
        for be in backends_to_try:
            try:
                analysis = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False, detector_backend=be)
                used_be = be
                break
            except Exception:
                analysis = None
                continue
        if used_be and used_be != primary:
            try:
                print(f"[info] detector fallback: '{primary}' -> '{used_be}' (deepface engine)")
            except Exception:
                pass
        # DeepFace may return a dict or a list of dicts depending on version
        if isinstance(analysis, list) and analysis:
            dominant = analysis[0].get("dominant_emotion")
        elif isinstance(analysis, dict):
            dominant = analysis.get("dominant_emotion")
        else:
            return "no_face"
        if not dominant:
            return "no_face"
        dominant_l = str(dominant).lower()
        mapped = RAW_TO_CANON.get(dominant_l)
        return mapped if mapped in CANON_EMOTIONS else "no_face"
    except Exception:
        # Common on blurry frames or intermittent detection issues
        return "no_face"


def stable_mode(buf: Deque[str], majority_frac: float = 0.5) -> Optional[str]:
    """Return the stable emotion only if a unique winner reaches a majority threshold.

    - Waits until the buffer is full
    - Computes frequency counts
    - Requires a unique top emotion
    - Requires the top emotion to meet the majority threshold (strict >50% when majority_frac=0.5)
    Returns None when unstable (tie or insufficient majority).
    """
    if not buf or len(buf) < buf.maxlen:
        return None
    try:
        counts = Counter(buf)
        top2 = counts.most_common(2)
        if not top2:
            return None
        # Check unique winner
        if len(top2) > 1 and top2[0][1] == top2[1][1]:
            return None
        winner, win_count = top2[0]
        # Compute minimum required count based on majority fraction
        n = len(buf)
        if majority_frac <= 0.5:
            # Strict majority: > n/2
            min_count = math.floor(n * 0.5) + 1
        else:
            min_count = math.ceil(n * majority_frac)
        if win_count >= min_count:
            return winner
        return None
    except Exception:
        return None


def send_command(esp32_ip: str, emotion: str, timeout: float = 2.0) -> bool:
    url = f"{esp32_ip.rstrip('/')}/mood?emotion={emotion}"
    try:
        r = requests.get(url, timeout=timeout)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


class SerialManager:
    def __init__(self, port: str, baud: int = 115200, log: bool = False):
        if serial is None:
            raise RuntimeError("pyserial is not installed. Run: pip install pyserial")
        self.port = port
        self.baud = baud
        # Try to open the requested port; if it's busy, try /dev/tty.* fallback and a brief retry loop
        def _try_open(p: str):
            return serial.Serial(port=p, baudrate=self.baud, timeout=0.1)

        last_err = None
        port_try = self.port
        alt_try = None
        if port_try.startswith("/dev/cu."):
            alt_try = port_try.replace("/dev/cu.", "/dev/tty.")
        elif port_try.startswith("/dev/tty."):
            alt_try = port_try.replace("/dev/tty.", "/dev/cu.")

        for attempt in range(6):  # ~3 seconds total
            try:
                self.ser = _try_open(port_try)
                break
            except Exception as e:
                last_err = e
                # On first failure, try the alternate device name once
                if alt_try and port_try != alt_try:
                    port_try, alt_try = alt_try, None
                    time.sleep(0.3)
                    continue
                time.sleep(0.5)
        else:
            raise last_err
        self._log = log
        self._reader_thread: Optional[threading.Thread] = None
        self._running = False
        # Give the ESP32 time to reset when serial opens
        time.sleep(1.8)
        # Clear any boot noise
        try:
            self.ser.reset_input_buffer()
        except Exception:
            pass
        if self._log:
            self._start_reader()

    def send_emotion(self, emotion: str) -> bool:
        try:
            line = f"EMOTION:{emotion}\n".encode("utf-8")
            self.ser.write(line)
            self.ser.flush()
            return True
        except Exception:
            return False

    def send_raw(self, text: str) -> bool:
        try:
            if not text.endswith("\n"):
                text += "\n"
            self.ser.write(text.encode("utf-8"))
            self.ser.flush()
            return True
        except Exception:
            return False

    def _start_reader(self):
        self._running = True
        def _reader():
            buf = bytearray()
            while self._running:
                try:
                    b = self.ser.read(1)
                    if not b:
                        continue
                    if b in (b"\n", b"\r"):
                        if buf:
                            try:
                                line = buf.decode("utf-8", errors="replace").strip()
                                if line:
                                    print(f"[ESP32] {line}")
                            finally:
                                buf.clear()
                    else:
                        buf.extend(b)
                except Exception:
                    # If read fails, keep trying until closed
                    time.sleep(0.05)
        self._reader_thread = threading.Thread(target=_reader, daemon=True)
        self._reader_thread.start()

    def close(self):
        try:
            self._running = False
            if self._reader_thread and self._reader_thread.is_alive():
                self._reader_thread.join(timeout=0.5)
        finally:
            try:
                self.ser.close()
            except Exception:
                pass


class MusicManager:
    """Simple local music player with fade out/in using pygame.mixer.

        Expects files in music_dir with names like:
            0001_happy.mp3 … 0007_surprise.mp3 (one per supported emotion)
    """

    def __init__(self, music_dir: str, fade_ms: int = 800):
        self.music_dir = music_dir
        self.fade_ms = max(0, fade_ms)
        self.enabled = True
        self.current: Optional[str] = None
        self._paused = False
        self.map: Dict[str, str] = {
            "happy": "0001_happy.mp3",
            "sad": "0002_sad.mp3",
            "angry": "0003_angry.mp3",
            "neutral": "0004_neutral.mp3",
            "fear": "0005_fear.mp3",
            "disgust": "0006_disgust.mp3",
            "surprise": "0007_surprise.mp3",
        }
        try:
            import pygame  # type: ignore

            self.pygame = pygame
            pygame.mixer.init()
            pygame.mixer.music.set_volume(1.0)
        except Exception as e:  # pragma: no cover
            print(f"Music disabled: pygame init failed: {e}")
            self.enabled = False

    def _path_for(self, emotion: str) -> Optional[str]:
        name = self.map.get(emotion)
        if name:
            p = os.path.join(self.music_dir, name)
            if os.path.exists(p):
                return p
        # Fallback: search by emotion.* in folder (any common audio extension)
        try:
            import glob as _glob
            for ext in ("mp3", "ogg", "wav", "m4a", "aac"):
                cand = os.path.join(self.music_dir, f"{emotion}.{ext}")
                if os.path.exists(cand):
                    return cand
            # looser match like *emotion*
            cands = _glob.glob(os.path.join(self.music_dir, f"*{emotion}*.*"))
            for c in cands:
                if os.path.isfile(c):
                    return c
        except Exception:
            pass
        return None

    def change_to(self, emotion: str):
        if not self.enabled:
            return
        # If paused and changing to an emotion, resume
        if self._paused:
            try:
                self.pygame.mixer.music.unpause()
            except Exception:
                pass
            self._paused = False
        if emotion == self.current:
            return
        path = self._path_for(emotion)
        if not path:
            print(f"Music file missing for '{emotion}'. Expected under {self.music_dir}")
            self.current = emotion
            return

        try:
            # Fade out current (non-blocking), then load & fade in new
            if self.pygame.mixer.music.get_busy():
                self.pygame.mixer.music.fadeout(self.fade_ms)
                # small wait to avoid abrupt overlap
                time.sleep(self.fade_ms / 1000.0)

            self.pygame.mixer.music.load(path)
            # start with fade-in
            self.pygame.mixer.music.play(loops=-1, fade_ms=self.fade_ms)
            self.current = emotion
            try:
                print(f"Playing: {emotion} ({os.path.basename(path)})")
            except Exception:
                pass
        except Exception as e:
            print(f"Failed to play '{emotion}' ({path}): {e}")

    def pause(self):
        if not self.enabled:
            return
        try:
            if self.pygame.mixer.music.get_busy():
                self.pygame.mixer.music.pause()
                self._paused = True
        except Exception:
            pass

    def set_volume(self, vol: float):
        """Set volume in [0.0, 1.0]."""
        if not self.enabled:
            return
        try:
            v = float(max(0.0, min(1.0, vol)))
            self.pygame.mixer.music.set_volume(v)
        except Exception:
            pass


class SpotifyMusicManager:
    """Spotify backend: starts a playlist/album/track per emotion with fade out/in.

    Requires Spotify Premium and a running Spotify Connect device (e.g., Spotify app on Mac).
    Credentials via SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REDIRECT_URI or CLI flags.
    """

    def __init__(
        self,
        device_query: Optional[str],
        cred: Dict[str, Optional[str]],
        mapping: Dict[str, Optional[str]],
        fade_ms: int = 800,
    ):
        if spotipy is None or SpotifyOAuth is None:
            raise RuntimeError("spotipy is not installed. Run: pip install spotipy")
        self.fade_ms = max(0, fade_ms)
        self.current: Optional[str] = None
        self.map = mapping
        scopes = "user-modify-playback-state user-read-playback-state"
        self.sp = spotipy.Spotify(
            auth_manager=SpotifyOAuth(
                client_id=cred.get("client_id"),
                client_secret=cred.get("client_secret"),
                redirect_uri=cred.get("redirect_uri"),
                scope=scopes,
                open_browser=True,
                cache_path=os.path.join(os.getcwd(), ".moodify_spotify_cache"),
            )
        )
        # Simple rate-limit awareness and throttling
        self._last_api_call = 0.0
        self._min_call_gap = 0.25  # seconds between calls
        self._penalty_until = 0.0  # when 429 told us to wait
        self._last_switch_time = 0.0
        self.min_switch_interval = 5.0  # avoid spamming start_playback
        # Resolve device
        self.device_id = None
        self.device_name = None
        self._select_device(device_query)
        self._paused = False

    def _select_device(self, query: Optional[str]):
        res = self._api_call(self.sp.devices)
        devices = (res or {}).get("devices", [])
        if not devices:
            print("Spotify: No devices found. Open Spotify app and start playback once.")
            return
        if query:
            ql = query.lower()
            for d in devices:
                if ql in d.get("name", "").lower():
                    self.device_id = d.get("id")
                    self.device_name = d.get("name")
                    break
        if self.device_id is None:
            # Fallback: pick active or first
            active = [d for d in devices if d.get("is_active")]
            d = active[0] if active else devices[0]
            self.device_id = d.get("id")
            self.device_name = d.get("name")
        print(f"Spotify target device: {self.device_name}")

    def _sleep_throttle(self):
        now = time.time()
        if now < self._penalty_until:
            time.sleep(max(0.0, self._penalty_until - now))
        gap = now - self._last_api_call
        if gap < self._min_call_gap:
            time.sleep(self._min_call_gap - gap)

    def _api_call(self, func, *args, **kwargs):
        self._sleep_throttle()
        try:
            res = func(*args, **kwargs)
            self._last_api_call = time.time()
            return res
        except Exception as e:
            # Handle 429 if available
            if SpotifyException is not None and isinstance(e, SpotifyException):
                try:
                    if getattr(e, "http_status", None) == 429:
                        retry_after = 1
                        # spotipy error may carry headers or .headers
                        headers = getattr(e, "headers", None) or {}
                        retry_after = int(headers.get("Retry-After", retry_after))
                        self._penalty_until = time.time() + retry_after
                        print(f"Spotify: rate limited, retry after {retry_after}s")
                        time.sleep(retry_after)
                        # one retry
                        try:
                            res = func(*args, **kwargs)
                            self._last_api_call = time.time()
                            return res
                        except Exception:
                            pass
                except Exception:
                    pass
            print(f"Spotify API call failed: {e}")
            return None

    def _fade_volume(self, start: int, end: int, steps: int = 4):
        start = int(max(0, min(100, start)))
        end = int(max(0, min(100, end)))
        if self.device_id is None:
            return
        if steps <= 0 or start == end:
            self._api_call(self.sp.volume, end, device_id=self.device_id)
            return
        delta = (end - start) / float(steps)
        for i in range(steps + 1):
            v = int(round(start + i * delta))
            self._api_call(self.sp.volume, v, device_id=self.device_id)
            time.sleep(max(0.05, self.fade_ms / 1000.0 / max(1, steps)))

    def _normalize_spotify_uri(self, uri: str) -> Optional[str]:
        """Accepts spotify:* URIs or open.spotify.com URLs and returns a spotify:* URI or track URI list.

        Returns a normalized string, or None if the input clearly isn't a supported Spotify context.
        Supported types: track, playlist, album, artist.
        """
        if not uri:
            return None
        u = str(uri).strip()
        if u.startswith("spotify:"):
            return u
        u_low = u.lower()
        if u_low.startswith("http://") or u_low.startswith("https://"):
            # Convert open.spotify.com URLs to spotify:* URIs
            try:
                import urllib.parse as ups

                p = ups.urlparse(u)
                if "open.spotify.com" not in p.netloc:
                    return None
                parts = [pp for pp in p.path.split("/") if pp]
                if len(parts) < 2:
                    return None
                kind, ident = parts[0], parts[1]
                ident = ident.split("?")[0]
                if kind in {"track", "playlist", "album", "artist"} and ident:
                    return f"spotify:{kind}:{ident}"
                return None
            except Exception:
                return None
        # Unknown format
        return None

    def _start_context(self, uri: str):
        if self.device_id is None:
            print("Spotify: No device selected.")
            return
        norm = self._normalize_spotify_uri(uri)
        if not norm:
            print("Spotify: unsupported or empty URI. Use a specific track/playlist/album/artist link or URI.")
            return
        if norm.startswith("spotify:track:"):
            self._api_call(self.sp.start_playback, device_id=self.device_id, uris=[norm])
        else:
            self._api_call(self.sp.start_playback, device_id=self.device_id, context_uri=norm)

    def change_to(self, emotion: str):
        uri = self.map.get(emotion)
        if not uri:
            print(f"Spotify: no URI set for '{emotion}'. Skipping.")
            self.current = emotion
            return
        if emotion == self.current:
            # If we were paused, resume by starting the same context again
            if self._paused:
                try:
                    self._start_context(uri)
                    self._paused = False
                except Exception:
                    pass
            return
        # Avoid spamming Spotify API
        now = time.time()
        if (now - self._last_switch_time) < self.min_switch_interval:
            return
        self._last_switch_time = now

        # Gentle, low-API fade and switch
        target_vol = 70
        self._fade_volume(target_vol, 30, steps=2)
        self._start_context(uri)
        time.sleep(max(0.1, self.fade_ms / 1000.0))
        self._fade_volume(30, target_vol, steps=3)
        self.current = emotion

    def pause(self):
        if self.device_id is None:
            return
        try:
            self._api_call(self.sp.pause_playback, device_id=self.device_id)
            self._paused = True
        except Exception:
            pass

    def set_volume(self, vol: float):
        """Set device volume using percentage 0-100 mapped from [0.0, 1.0]."""
        if self.device_id is None:
            return
        try:
            pct = int(max(0, min(100, round(float(vol) * 100))))
            self._api_call(self.sp.volume, pct, device_id=self.device_id)
        except Exception:
            pass


def main():
    args = parse_args()

    # Initialize stability buffer; if buffer_len <= 0, defer to auto-size with a sane temporary size
    temp_buf_len = args.buffer_len if args.buffer_len and args.buffer_len > 0 else 25
    emotion_buffer: Deque[str] = collections.deque(maxlen=int(temp_buf_len))
    last_sent_emotion: Optional[str] = None
    last_send_time = 0.0
    last_stable_change_time = 0.0
    no_face_first_seen = 0.0
    buf_lock = threading.Lock()

    music_mgr: Optional[object] = None
    predictor = None
    # Decide engine first: custom, deepface, or auto
    engine = args.emotion_engine
    if engine not in ("auto", "deepface", "custom"):
        engine = "auto"

    if engine in ("custom", "auto") and args.custom_classifier:
        try:
            import joblib as _joblib  # type: ignore
            predictor = _joblib.load(args.custom_classifier)
            if not isinstance(predictor, dict) or "pipeline" not in predictor or "labels" not in predictor:
                print("Custom classifier file is invalid; ignoring.")
                predictor = None
            else:
                print(f"Loaded custom classifier with labels: {predictor.get('labels')}")
                # Prefer the embedding backbone stored in the model payload if present
                try:
                    saved_embed = predictor.get("embedding_model") if isinstance(predictor, dict) else None
                    if isinstance(saved_embed, str) and saved_embed:
                        if saved_embed != args.embedding_model:
                            print(f"[info] overriding embedding model: runtime '{args.embedding_model}' -> saved '{saved_embed}'")
                        args.embedding_model = saved_embed
                except Exception:
                    pass
                try:
                    print(f"Using custom classifier: {args.custom_classifier} (embedding {args.embedding_model})")
                except Exception:
                    pass
        except Exception as e:
            print(f"Failed to load custom classifier: {e}")
    # If engine explicitly deepface, ignore predictor
    if engine == "deepface":
        predictor = None
    # Log selected engine
    if predictor is not None:
        print(f"Engine: custom | detector: {args.detector_backend}")
    else:
        print(f"Engine: deepface | detector: {args.detector_backend}")
    if args.spotify:
        try:
            mapping = {
                "happy": args.sp_happy,
                "sad": args.sp_sad,
                "angry": args.sp_angry,
                "neutral": args.sp_neutral,
                "fear": args.sp_fear,
                "disgust": args.sp_disgust,
                "surprise": args.sp_surprise,
            }
            cred = {
                "client_id": args.spotify_client_id,
                "client_secret": args.spotify_client_secret,
                "redirect_uri": args.spotify_redirect_uri,
            }
            music_mgr = SpotifyMusicManager(args.spotify_device, cred, mapping)
        except Exception as e:
            print(f"Spotify disabled: {e}")
            music_mgr = None
    elif not args.no_music:
        music_mgr = MusicManager(args.music_dir)

    ser_mgr: Optional[SerialManager] = None
    if args.serial_port:
        try:
            ser_mgr = SerialManager(args.serial_port, args.baud, log=args.serial_log)
            print(f"Serial connected: {args.serial_port} @ {args.baud}")
        except Exception as e:
            print(f"Failed to open serial '{args.serial_port}': {e}")

    def _open_camera_with_fallback(index: int, width: int, height: int, video_backend: str):
        """Open webcam with OS-aware backend priority and index scan.

        Returns (cap, used_index, used_backend, tried_list)
        where used_backend is a string label, and tried_list is a list of (idx, backend_label).
        """
        import sys as _sys

        tried = []

        # Resolve OpenCV backend constants if present
        _AVF = getattr(cv2, "CAP_AVFOUNDATION", None)
        _DSHOW = getattr(cv2, "CAP_DSHOW", None)
        _MSMF = getattr(cv2, "CAP_MSMF", None)
        _V4L2 = getattr(cv2, "CAP_V4L2", None)

        # Build priority list based on requested backend and OS
        plat = _sys.platform
        backends = []  # list of (label, apiPref or None)

        vb = (video_backend or "auto").lower()
        if vb == "default":
            backends = [("DEFAULT", None)]
        elif vb == "dshow" and _DSHOW is not None:
            backends = [("DSHOW", _DSHOW)]
        elif vb == "msmf" and _MSMF is not None:
            backends = [("MSMF", _MSMF)]
        elif vb == "avfoundation" and _AVF is not None:
            backends = [("AVFOUNDATION", _AVF)]
        elif vb == "v4l2" and _V4L2 is not None:
            backends = [("V4L2", _V4L2)]
        else:
            # auto: choose sensible defaults per OS
            if plat.startswith("win"):
                # Prefer DirectShow on Windows, then MSMF, then default
                if _DSHOW is not None:
                    backends.append(("DSHOW", _DSHOW))
                if _MSMF is not None:
                    backends.append(("MSMF", _MSMF))
                backends.append(("DEFAULT", None))
            elif plat == "darwin":
                if _AVF is not None:
                    backends.append(("AVFOUNDATION", _AVF))
                backends.append(("DEFAULT", None))
            else:
                if _V4L2 is not None:
                    backends.append(("V4L2", _V4L2))
                backends.append(("DEFAULT", None))

        indices = [index] + [i for i in range(0, 6) if i != index]
        for idx in indices:
            for label, be in backends:
                cap = cv2.VideoCapture(idx, be) if be is not None else cv2.VideoCapture(idx)
                if cap.isOpened():
                    # Try to set resolution and test a frame
                    try:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    except Exception:
                        pass
                    ok, _ = cap.read()
                    if ok:
                        return cap, idx, label, tried
                    cap.release()
                tried.append((idx, label))
        return None, None, None, tried

    cap, used_idx, used_be, tried = _open_camera_with_fallback(
        args.camera_index, args.width, args.height, args.video_backend
    )
    if cap is None:
        tried_str = ", ".join([f"idx {i} ({b})" for i, b in tried]) or "none"
        print(f"Error: Could not open webcam. Tried: {tried_str}.")
        print(
            "Hints: \n"
            "- Close any other app using the camera (Teams/Zoom/Discord/Camera app).\n"
            "- On Windows, try --video-backend dshow (DirectShow) or --video-backend msfm, and try --camera-index 0/1.\n"
            "- On Windows, check Privacy → Camera permissions for Python.\n"
            "- On macOS, grant camera permission in System Settings → Privacy & Security → Camera.\n"
            "- If using Windows N edition, install the Media Feature Pack."
        )
        return
    else:
        print(f"Camera opened: index {used_idx} via {used_be}")

    print("Starting Moodify Brain…")
    print("Press 'q' in the preview window to quit." if args.display else "Press Ctrl+C to quit.")

    # Analysis worker: decouple slow DeepFace from capture/render for smooth UI
    frame_queue: "queue.Queue" = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    capture_frames = 0
    analysis_frames = 0
    fps_window_start = time.time()
    auto_buf_enabled = args.buffer_len is None or args.buffer_len <= 0
    auto_buf_applied = not auto_buf_enabled
    # Control file watcher
    control_path = args.control_file
    control_last_mtime = 0.0

    def analysis_worker():
        while not stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                emo = analyze_emotion(frame, predictor=predictor, embedding_model=args.embedding_model, detector_backend=args.detector_backend)
                if emo is not None:
                    with buf_lock:
                        emotion_buffer.append(emo)
                # Count analysis output frames
            except Exception as e:
                print(f"Analysis worker error: {e}")
                time.sleep(0.1)
            finally:
                try:
                    frame_queue.task_done()
                except Exception:
                    pass
            try:
                # Increment analysis counter outside locks
                analysis_frames += 1
            except Exception:
                pass

    worker = threading.Thread(target=analysis_worker, daemon=True)
    worker.start()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Warning: Failed to read frame from webcam. Retrying…")
                time.sleep(0.5)
                continue
            capture_frames += 1

            # Feed most-recent frame to analysis worker without blocking UI
            if frame_queue.empty():
                try:
                    frame_queue.put_nowait(frame)
                except queue.Full:
                    pass

            with buf_lock:
                stable = stable_mode(emotion_buffer, majority_frac=max(0.0, min(1.0, args.majority_frac)))

            overlay_text = "Analyzing…"
            if stable == "no_face":
                overlay_text = "No face detected"
            elif stable:
                overlay_text = f"Stable Emotion: {stable.upper()}"

            now = time.time()

            # Auto-size buffer after ~2s if requested (buffer-len <= 0)
            if not auto_buf_applied and (now - fps_window_start) >= 2.0 and capture_frames > 0:
                fps_est = capture_frames / max(1e-6, (now - fps_window_start))
                new_len = int(max(15, min(200, fps_est * 2.5)))
                with buf_lock:
                    prev = list(emotion_buffer)
                    emotion_buffer = collections.deque(prev[-new_len:], maxlen=new_len)
                auto_buf_applied = True
                print(f"[info] auto buffer_len set to {new_len} (~2.5s @ {fps_est:.1f} FPS)")

            # GUI control file: apply pause/play/volume commands if present
            if control_path:
                try:
                    import os as _os, json as _json
                    mt = _os.path.getmtime(control_path) if _os.path.exists(control_path) else 0.0
                    if mt > control_last_mtime:
                        control_last_mtime = mt
                        with open(control_path, "r") as _f:
                            cfg = _json.load(_f)
                        if isinstance(cfg, dict) and music_mgr is not None:
                            # Volume control
                            if "volume" in cfg:
                                v = float(cfg.get("volume", 1.0))
                                setter = getattr(music_mgr, "set_volume", None)
                                if callable(setter):
                                    setter(v)
                            # Pause
                            if cfg.get("pause") is True:
                                pa = getattr(music_mgr, "pause", None)
                                if callable(pa):
                                    pa()
                            # Play/Resume
                            if cfg.get("play") is True:
                                try:
                                    cur = getattr(music_mgr, "current", None)
                                    if isinstance(cur, str) and cur:
                                        music_mgr.change_to(cur)
                                    else:
                                        music_mgr.change_to(last_sent_emotion or "neutral")
                                except Exception:
                                    pass
                except Exception:
                    pass

            # Handle explicit no_face state by driving to neutral once
            if stable == "no_face" and (now - last_send_time) >= args.min_interval and (last_sent_emotion != ("neutral" if args.no_face_behavior == "neutral" else "no_face")):
                # Grace period before applying no-face behavior
                if no_face_first_seen == 0.0:
                    no_face_first_seen = now
                if (now - no_face_first_seen) < max(0.0, float(args.no_face_grace)):
                    continue
                # LEDs: always drive to neutral; Music: pause or neutral per setting
                target = "neutral"
                sent_serial = False
                if ser_mgr is not None:
                    sent_serial = ser_mgr.send_emotion(target)
                ok_send = True
                status = "SERIAL" if sent_serial else "HTTP"
                if not sent_serial:
                    ok_send = send_command(args.esp32_ip, target)
                    status += ":OK" if ok_send else ":FAILED"
                if music_mgr is not None:
                    try:
                        if args.no_face_behavior == "pause":
                            # Pause if supported; SpotifyMusicManager and MusicManager implement pause()
                            pause_fn = getattr(music_mgr, "pause", None)
                            if callable(pause_fn):
                                pause_fn()
                        elif getattr(music_mgr, "enabled", True):
                            music_mgr.change_to(target)
                    except Exception:
                        pass
                print(f"No face -> LEDs:{target} | Music:{args.no_face_behavior}")
                if ok_send or sent_serial:
                    last_sent_emotion = "no_face" if args.no_face_behavior == "pause" else target
                    last_send_time = now
                    last_stable_change_time = now

            # Normal emotion changes
            elif stable and stable != "no_face" and stable != last_sent_emotion and (now - last_send_time) >= args.min_interval:
                # Face is back; reset no-face timer
                no_face_first_seen = 0.0
                # Enforce hold/cooldown window
                if (now - last_stable_change_time) < max(0.0, float(args.hold_seconds)):
                    pass  # still holding last emotion
                else:
                    # 1) Update ESP32 LEDs (prefer serial if available)
                    sent_serial = False
                    if ser_mgr is not None:
                        sent_serial = ser_mgr.send_emotion(stable)
                    ok_send = True
                    status = "SERIAL" if sent_serial else "HTTP"
                    if not sent_serial:
                        ok_send = send_command(args.esp32_ip, stable)
                        status += ":OK" if ok_send else ":FAILED"
                    # 2) Update music
                    if music_mgr is not None and getattr(music_mgr, "enabled", True):
                        try:
                            music_mgr.change_to(stable)
                        except Exception:
                            pass

                    music_on = bool(music_mgr is not None and getattr(music_mgr, "enabled", True))
                    print(f"New stable emotion: {stable} -> {status} | music {'ON' if music_on else 'OFF'}")
                    if ok_send or sent_serial:
                        last_sent_emotion = stable
                        last_send_time = now
                        last_stable_change_time = now

            if args.display:
                # Overlay
                cv2.putText(
                    frame,
                    overlay_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0) if stable and stable != "no_face" else (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )
                # Show lightweight FPS
                dt = max(1e-6, time.time() - fps_window_start)
                if dt >= 5.0:
                    cap_fps = capture_frames / dt
                    ana_fps = analysis_frames / dt
                    print(f"FPS - Capture: {cap_fps:.1f}, Analysis: {ana_fps:.1f}")
                    capture_frames = 0
                    analysis_frames = 0
                    fps_window_start = time.time()
                try:
                    cap_fps_text = f"FPS C/A: {capture_frames/max(1e-3, dt):.1f}/{analysis_frames/max(1e-3, dt):.1f}"
                    cv2.putText(
                        frame,
                        cap_fps_text,
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (200, 200, 200),
                        1,
                        cv2.LINE_AA,
                    )
                except Exception:
                    pass
                cv2.imshow("Moodify Live", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        try:
            stop_event.set()
        except Exception:
            pass
        try:
            if worker.is_alive():
                worker.join(timeout=0.5)
        except Exception:
            pass
        cap.release()
        if args.display:
            cv2.destroyAllWindows()
        if ser_mgr is not None:
            ser_mgr.close()
        print("Moodify Brain stopped.")


if __name__ == "__main__":
    main()
