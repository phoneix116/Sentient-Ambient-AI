#!/usr/bin/env python3
"""
Moodify Desktop (Qt / PySide6)

A desktop GUI to run Moodify without the Flask web UI or Tkinter.
- Configure serial, camera, and music mode (Local or Spotify)
- Start/Stop mood_detector.py as a subprocess
- Live log viewer
- Helpers for serial testing and Spotify cache clearing

Run:
  source .venv/bin/activate
  pip install -r requirements.txt
  python3 moodify_desktop_qt.py

Notes:
- On macOS, keep Arduino Serial Monitor closed when starting (port exclusive).
- For Spotify, the Redirect URI must match exactly what you added in your Spotify Dashboard.
"""
from __future__ import annotations

import os
import sys
import subprocess
import threading
import time
import shlex
import glob
import re
import json
import io
import base64
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

# Optional Spotify import for preflight auth
try:
    import spotipy  # type: ignore
    from spotipy.oauth2 import SpotifyOAuth  # type: ignore
except Exception:
    spotipy = None
    SpotifyOAuth = None

# Prefer PyQt6 (fewer install issues on macOS), fall back to PySide6. Unify Signal and echo mode.
try:
    from PyQt6 import QtCore, QtGui, QtWidgets  # type: ignore
    from PyQt6.QtCore import pyqtSignal as Signal  # type: ignore
    QT_BACKEND = "PyQt6"
    QLINEEDIT_PASSWORD = QtWidgets.QLineEdit.EchoMode.Password
except Exception:
    from PySide6 import QtCore, QtGui, QtWidgets  # type: ignore
    try:
        Signal = QtCore.Signal  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        from PySide6.QtCore import Signal  # type: ignore
    QT_BACKEND = "PySide6"
    QLINEEDIT_PASSWORD = QtWidgets.QLineEdit.Password

ROOT = os.path.dirname(os.path.abspath(__file__))
PY_EXE = sys.executable or "python3"
SCRIPT = os.path.join(ROOT, "mood_detector.py")
CACHE_FILE = os.path.join(ROOT, ".moodify_spotify_cache")

# Optional matplotlib for charts in Report tab
HAS_MPL = False
try:  # Lazy optional import
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas  # type: ignore
    from matplotlib.figure import Figure  # type: ignore
    HAS_MPL = True
except Exception:
    HAS_MPL = False


class SquareLabel(QtWidgets.QLabel):
    """Label that keeps height in sync with width for a square preview."""

    def hasHeightForWidth(self) -> bool:  # pragma: no cover - Qt hook
        return True

    def heightForWidth(self, width: int) -> int:  # pragma: no cover - Qt hook
        return width

    def sizeHint(self) -> QtCore.QSize:  # pragma: no cover - Qt hook
        hint = super().sizeHint()
        side = max(hint.width(), hint.height(), 360)
        return QtCore.QSize(side, side)

    def minimumSizeHint(self) -> QtCore.QSize:  # pragma: no cover - Qt hook
        return QtCore.QSize(360, 360)


class LogReader(QtCore.QThread):
    line = Signal(str)
    exited = Signal(int)

    def __init__(self, proc: subprocess.Popen):
        super().__init__()
        self._proc = proc
        self._stop = False

    def run(self):
        if not self._proc.stdout:
            return
        for raw in self._proc.stdout:
            if self._stop:
                break
            try:
                self.line.emit(raw.rstrip("\n"))
            except Exception:
                pass
        try:
            rc = self._proc.wait(timeout=0.5)
            self.exited.emit(rc)
        except Exception:
            pass

    def stop(self):
        self._stop = True


class SessionRecorder:
    """Collects session analytics by parsing stdout lines and UI events.

    Captured items:
    - stable emotion changes ("New stable emotion: <emo> -> ...")
    - music play events ("Playing: <emo> (<file>)")
    - ESP32 lines ("[ESP32] ...")
    - errors ("[error]"/"Failed to play")
    - session metadata at start
    """

    EMOTIONS = ["happy","sad","angry","neutral","fear","disgust","surprise","no_face"]

    def __init__(self):
        self.reset()

    def reset(self):
        self.start_ts: Optional[float] = None
        self.end_ts: Optional[float] = None
        self.events: List[Dict[str, Any]] = []
        self.stable_changes: List[Dict[str, Any]] = []  # {ts, emotion}
        self.play_events: List[Dict[str, Any]] = []     # {ts, emotion, track}
        self.meta: Dict[str, Any] = {}

    # --- Lifecycle ---
    def session_start(self, meta: Dict[str, Any]):
        self.reset()
        self.start_ts = time.time()
        self.meta = dict(meta or {})

    def session_stop(self):
        if self.end_ts is None:
            self.end_ts = time.time()

    # --- Parsing ---
    # Be liberal: allow prefixes, varying cases, optional filename for play
    _re_stable = re.compile(r"new\s+stable\s+emotion:\s*(?P<emo>[a-z_]+)", re.IGNORECASE)
    _re_play = re.compile(r"playing:\s*(?P<emo>[a-z_]+)(?:\s*\((?P<file>[^)]*)\))?", re.IGNORECASE)

    def on_line(self, line: str):
        ts = time.time()
        line = line.strip()
        if not line:
            return
        self.events.append({"ts": ts, "line": line})
        m = self._re_stable.search(line)
        if m:
            emo = m.group("emo").lower()
            self.stable_changes.append({"ts": ts, "emotion": emo})
            return
        m = self._re_play.search(line)
        if m:
            self.play_events.append({"ts": ts, "emotion": m.group("emo").lower(), "track": m.group("file")})
            return
        if line.startswith("[ESP32]"):
            self.events[-1]["esp32"] = True
        if "[error]" in line.lower() or line.lower().startswith("failed to play"):
            self.events[-1]["error"] = True

    # --- Summaries ---
    def _session_window(self) -> Tuple[float, float]:
        start = self.start_ts or time.time()
        end = self.end_ts or time.time()
        return start, max(end, start)

    def durations_by_emotion(self) -> Dict[str, float]:
        """Approximate time spent in each stable emotion using change timestamps."""
        start, end = self._session_window()
        # Build segments between changes
        changes = list(self.stable_changes)
        if not changes:
            return {k: 0.0 for k in self.EMOTIONS}
        durations = {k: 0.0 for k in self.EMOTIONS}
        for i, cur in enumerate(changes):
            t0 = cur["ts"]
            t1 = changes[i+1]["ts"] if i+1 < len(changes) else end
            emo = cur["emotion"]
            durations[emo] = durations.get(emo, 0.0) + max(0.0, t1 - t0)
        return durations

    def counts_by_emotion(self) -> Dict[str, int]:
        counts = {k: 0 for k in self.EMOTIONS}
        for e in self.stable_changes:
            counts[e["emotion"]] = counts.get(e["emotion"], 0) + 1
        return counts

    def summary(self) -> Dict[str, Any]:
        start, end = self._session_window()
        dur = end - start
        counts = self.counts_by_emotion()
        durs = self.durations_by_emotion()
        # Compute top emotion more robustly: prefer duration max, else last stable, else none
        top: Optional[str] = None
        if any(v > 0 for v in durs.values()):
            top = max(durs.items(), key=lambda kv: kv[1])[0]
        elif self.stable_changes:
            top = self.stable_changes[-1]["emotion"]
        elif any(v > 0 for v in counts.values()):
            top = max(counts.items(), key=lambda kv: kv[1])[0]
        return {
            "started": datetime.fromtimestamp(start).isoformat(timespec="seconds"),
            "ended": datetime.fromtimestamp(end).isoformat(timespec="seconds"),
            "duration_sec": round(dur, 2),
            "changes": len(self.stable_changes),
            "top_emotion": top,
            "counts": counts,
            "durations": {k: round(v, 2) for k, v in durs.items()},
            "plays": len(self.play_events),
            "meta": self.meta,
        }

    # --- Export ---
    def export_csv(self, path: str):
        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["timestamp_iso", "type", "emotion", "detail"]) 
            for e in self.stable_changes:
                w.writerow([datetime.fromtimestamp(e["ts"]).isoformat(), "stable", e["emotion"], ""]) 
            for e in self.play_events:
                w.writerow([datetime.fromtimestamp(e["ts"]).isoformat(), "playing", e["emotion"], e.get("track","")])
            for e in self.events:
                if e.get("error"):
                    w.writerow([datetime.fromtimestamp(e["ts"]).isoformat(), "error", "", e["line"]])

    def export_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.summary(), f, indent=2)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Moodify Desktop ({QT_BACKEND})")
        self.resize(980, 720)
        self.setMinimumSize(800, 600)

        # --- Global palette & stylesheet (modern dark theme with accent) ---
        try:
            accent = "#00C2A8"  # teal accent
            bg0 = "#0f1115"; bg1 = "#151922"; bg2 = "#1B2130"; bg3 = "#242C3D"
            fg0 = "#E6EDF3"; fg1 = "#C2CAD3"; subtle = "#9AA5B1"; border = "#2B354A"
            self.setStyleSheet(
                f"""
                QWidget {{ background: {bg1}; color: {fg0}; font-family: 'Segoe UI', 'Inter', sans-serif; font-size: 12.5px; }}
                QGroupBox {{ border: 1px solid {border}; border-radius: 8px; margin-top: 12px; background: {bg2}; }}
                QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {fg1}; }}
                QPushButton {{ background: {bg3}; color: {fg0}; border: 1px solid {border}; border-radius: 6px; padding: 6px 10px; }}
                QPushButton:hover {{ border-color: {accent}; }}
                QPushButton:pressed {{ background: {bg2}; }}
                QPushButton#Primary {{ background: {accent}; color: #061116; border: none; }}
                QPushButton#Ghost {{ background: transparent; border: 1px solid {border}; color: {fg1}; }}
                QPushButton#Ghost:hover {{ border-color: {accent}; color: {fg0}; }}
                QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{ background: {bg0}; border: 1px solid {border}; border-radius: 6px; padding: 6px; selection-background-color: {accent}; selection-color: #061116; }}
                QTabWidget::pane {{ border: 1px solid {border}; border-radius: 8px; background: {bg2}; }}
                QTabBar::tab {{ background: {bg3}; padding: 6px 12px; border-top-left-radius: 6px; border-top-right-radius: 6px; margin-right: 4px; }}
                QTabBar::tab:selected {{ background: {bg2}; color: {fg0}; }}
                QLabel#Hint {{ color: {subtle}; font-size: 11px; }}
                QLabel#Kpi {{ font-size: 28px; font-weight: 600; color: {fg0}; }}
                QLabel#KpiSub {{ color: {subtle}; font-size: 12px; letter-spacing: .3px; text-transform: uppercase; }}
                QLabel#Pill {{ background: {bg3}; border: 1px solid {border}; border-radius: 10px; padding: 3px 8px; color: {fg1}; }}
                QFrame#Card {{ background: {bg3}; border: 1px solid {border}; border-radius: 10px; }}
                QHeaderView::section {{ background: {bg3}; color: {fg1}; border: 1px solid {border}; padding: 6px; }}
                QTableWidget {{ gridline-color: {border}; }}
                QSlider::groove:horizontal {{ height: 6px; background: {border}; border-radius: 3px; }}
                QSlider::handle:horizontal {{ width: 16px; background: {accent}; border-radius: 8px; margin: -5px 0; }}
                QStatusBar {{ background: {bg2}; border-top: 1px solid {border}; color: {fg1}; }}
                """
            )
        except Exception:
            pass

        # Basic View menu for resizing
        try:
            mb = self.menuBar()
            view = mb.addMenu("View")
            act_zoom = QtGui.QAction("Zoom", self)
            act_zoom.triggered.connect(lambda: self.showNormal() if self.isMaximized() else self.showMaximized())
            view.addAction(act_zoom)
            act_fs = QtGui.QAction("Enter Full Screen", self)
            act_fs.setShortcut("Ctrl+Meta+F")
            def _toggle_fs():
                self.showNormal() if self.isFullScreen() else self.showFullScreen()
            act_fs.triggered.connect(_toggle_fs)
            view.addAction(act_fs)
        except Exception:
            pass

        self.proc: Optional[subprocess.Popen] = None
        self.reader: Optional[LogReader] = None
        self._trainProc: Optional[subprocess.Popen] = None
        self._trainReader: Optional[LogReader] = None

        # --- Main Layout & Tabs Setup ---
        centralWidget = QtWidgets.QWidget()
        self.setCentralWidget(centralWidget)
        mainLayout = QtWidgets.QVBoxLayout(centralWidget)
        self.tabs = QtWidgets.QTabWidget()
        mainLayout.addWidget(self.tabs)

        runPage = QtWidgets.QWidget()
        trainPage = QtWidgets.QWidget()
        reportPage = QtWidgets.QWidget()
        logsPage = QtWidgets.QWidget()

        runLayout = QtWidgets.QVBoxLayout(runPage)
        trainLayout = QtWidgets.QVBoxLayout(trainPage)
        reportLayout = QtWidgets.QVBoxLayout(reportPage)
        logsLayout = QtWidgets.QVBoxLayout(logsPage)

        # --- Build Run Tab ---
        form = QtWidgets.QFormLayout()
        runLayout.addLayout(form)
        self.serialPort = QtWidgets.QComboBox(); self.serialPort.setEditable(True)
        self.baud = QtWidgets.QLineEdit("115200")
        self.serialLog = QtWidgets.QCheckBox("Show ESP32 logs"); self.serialLog.setChecked(True)
        serialBox = QtWidgets.QHBoxLayout(); serialBox.addWidget(self.serialPort); serialBox.addWidget(QtWidgets.QLabel("Baud:")); serialBox.addWidget(self.baud); serialBox.addWidget(self.serialLog)
        spw = QtWidgets.QWidget(); spw.setLayout(serialBox)
        form.addRow("Serial Port", spw)
        self.display = QtWidgets.QCheckBox("Show preview"); self.display.setChecked(True)
        self.cameraIndex = QtWidgets.QSpinBox(); self.cameraIndex.setRange(0, 8); self.cameraIndex.setValue(0)
        self.minInterval = QtWidgets.QDoubleSpinBox(); self.minInterval.setRange(0.1, 10.0); self.minInterval.setSingleStep(0.1); self.minInterval.setValue(1.0)
        self.pauseNoFace = QtWidgets.QCheckBox("Pause music when no face")
        self.videoBackend = QtWidgets.QComboBox(); self.videoBackend.addItems(["auto", "dshow", "msmf", "default"])
        if sys.platform.startswith("win"): self.videoBackend.setCurrentText("dshow")
        else: self.videoBackend.setCurrentText("auto")
        vbox = QtWidgets.QHBoxLayout(); vbox.addWidget(self.display); vbox.addWidget(QtWidgets.QLabel("Camera index:")); vbox.addWidget(self.cameraIndex); vbox.addWidget(QtWidgets.QLabel("Min interval (s):")); vbox.addWidget(self.minInterval); vbox.addWidget(QtWidgets.QLabel("Video backend:")); vbox.addWidget(self.videoBackend); vbox.addWidget(self.pauseNoFace)
        vw = QtWidgets.QWidget(); vw.setLayout(vbox)
        form.addRow("Vision", vw)
        self.previewGroup = QtWidgets.QGroupBox("Live Camera Preview (idle only)")
        pv = QtWidgets.QVBoxLayout(); self.previewLabel = SquareLabel(); self.previewLabel.setMinimumSize(200, 200)
        self.previewLabel.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding); self.previewLabel.setStyleSheet("background:#111;border:1px solid #333"); self.previewLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        hint = QtWidgets.QLabel("Preview runs when not started. Close other camera apps if blank."); hint.setObjectName("Hint")
        ph = QtWidgets.QHBoxLayout(); self.previewStartBtn = QtWidgets.QPushButton("Start Preview"); self.previewStopBtn = QtWidgets.QPushButton("Stop Preview")
        ph.addWidget(self.previewStartBtn); ph.addWidget(self.previewStopBtn); ph.addStretch(1)
        pv.addWidget(self.previewLabel, alignment=QtCore.Qt.AlignmentFlag.AlignCenter); pv.addLayout(ph); pv.addWidget(hint)
        self.previewGroup.setLayout(pv)
        self.musicCtrlGroup = QtWidgets.QGroupBox("Music Controls")
        mc = QtWidgets.QVBoxLayout(); mc.setContentsMargins(8, 8, 8, 8); mc.setSpacing(8)
        self.nowPlaying = QtWidgets.QLabel("-"); self.nowPlaying.setVisible(False)
        volLabel = QtWidgets.QLabel("Volume"); volLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        mc.addWidget(volLabel)
        self.volumeSlider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal); self.volumeSlider.setRange(0, 100); self.volumeSlider.setValue(100); self.volumeSlider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.volumeSlider.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed); self.volumeSlider.setMinimumHeight(24)
        mc.addWidget(self.volumeSlider)
        row = QtWidgets.QHBoxLayout(); row.addStretch(1); self.pausePlayBtn = QtWidgets.QPushButton("Pause"); self.pausePlayBtn.setMinimumWidth(80); row.addWidget(self.pausePlayBtn); mc.addLayout(row)
        self.musicCtrlGroup.setLayout(mc); self.musicCtrlGroup.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        # Use a splitter for better cross-platform resizing of preview vs. music controls
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(self.previewGroup)
        splitter.addWidget(self.musicCtrlGroup)
        try:
            splitter.setChildrenCollapsible(False)
        except Exception:
            pass
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        runLayout.addWidget(splitter, 1)
        self.musicMode = QtWidgets.QComboBox(); self.musicMode.addItems(["Local", "Spotify"])
        form.addRow("Music Mode", self.musicMode)
        self.musicDir = QtWidgets.QLineEdit(os.path.join(ROOT, "media", "mp3"))
        browseBtn = QtWidgets.QPushButton("Browse…")
        lhb = QtWidgets.QHBoxLayout(); lhb.addWidget(QtWidgets.QLabel("Music folder")); lhb.addWidget(self.musicDir, 1); lhb.addWidget(browseBtn)
        self.localGroup = QtWidgets.QGroupBox("Local music"); lw = QtWidgets.QWidget(); lw.setLayout(lhb); v_local = QtWidgets.QVBoxLayout(); v_local.addWidget(lw); self.localGroup.setLayout(v_local)
        form.addRow(self.localGroup)
        self.spDevice = QtWidgets.QLineEdit("MacBook"); self.spId = QtWidgets.QLineEdit(); self.spSecret = QtWidgets.QLineEdit(); self.spSecret.setEchoMode(QLINEEDIT_PASSWORD); self.spRedirect = QtWidgets.QLineEdit("http://127.0.0.1:8888/callback")
        self.spHappy = QtWidgets.QLineEdit(); self.spSad = QtWidgets.QLineEdit(); self.spAngry = QtWidgets.QLineEdit(); self.spNeutral = QtWidgets.QLineEdit(); self.spFear = QtWidgets.QLineEdit(); self.spDisgust = QtWidgets.QLineEdit(); self.spSurprise = QtWidgets.QLineEdit()
        self.spGroup = QtWidgets.QGroupBox("Spotify settings"); spForm = QtWidgets.QFormLayout(); spForm.addRow("Spotify device contains", self.spDevice); spForm.addRow("Spotify client id", self.spId); spForm.addRow("Spotify client secret", self.spSecret); spForm.addRow("Spotify redirect URI", self.spRedirect)
        spForm.addRow("URI happy", self.spHappy); spForm.addRow("URI sad", self.spSad); spForm.addRow("URI angry", self.spAngry); spForm.addRow("URI neutral", self.spNeutral); spForm.addRow("URI fear", self.spFear); spForm.addRow("URI disgust", self.spDisgust); spForm.addRow("URI surprise", self.spSurprise)
        self.spGroup.setLayout(spForm); form.addRow(self.spGroup)
        modelGrp = QtWidgets.QGroupBox("Model"); mf = QtWidgets.QFormLayout(); self.runUseModel = QtWidgets.QCheckBox("Use custom model"); mf.addRow(self.runUseModel)
        mh = QtWidgets.QHBoxLayout(); self.runModelCombo = QtWidgets.QComboBox(); self.runModelCombo.setEditable(True); self.runModelCombo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.runPickModelBtn = QtWidgets.QPushButton("Pick…"); mh.addWidget(self.runModelCombo, 1); mh.addWidget(self.runPickModelBtn); mw = QtWidgets.QWidget(); mw.setLayout(mh); mf.addRow("Model file", mw); modelGrp.setLayout(mf)
        detectGrp = QtWidgets.QGroupBox("Detection"); df = QtWidgets.QFormLayout(); self.engineCombo = QtWidgets.QComboBox(); self.engineCombo.addItems(["Auto", "DeepFace CNN", "Custom model"]); self.engineCombo.setCurrentText("Auto")
        self.detectorCombo = QtWidgets.QComboBox(); self.detectorCombo.addItems(["opencv", "retinaface", "mediapipe", "mtcnn", "ssd", "dlib"]); self.detectorCombo.setCurrentText("opencv"); df.addRow("Emotion engine", self.engineCombo); df.addRow("Detector backend", self.detectorCombo); detectGrp.setLayout(df)
        modelDetectRow = QtWidgets.QHBoxLayout(); modelDetectRow.addWidget(modelGrp, 1); modelDetectRow.addWidget(detectGrp, 1); runLayout.addLayout(modelDetectRow)
        btns = QtWidgets.QHBoxLayout(); self.startBtn = QtWidgets.QPushButton("Start"); self.startBtn.setObjectName("Primary"); self.stopBtn = QtWidgets.QPushButton("Stop"); self.refreshBtn = QtWidgets.QPushButton("Refresh Ports"); self.clearCacheBtn = QtWidgets.QPushButton("Clear Spotify Cache"); self.testSerialBtn = QtWidgets.QPushButton("Send TEST"); self.spAuthBtn = QtWidgets.QPushButton("Authorize Spotify…"); self.openLogsBtn = QtWidgets.QPushButton("Open logs…")
        btns.addWidget(self.startBtn); btns.addWidget(self.stopBtn); btns.addStretch(1); btns.addWidget(self.refreshBtn); btns.addWidget(self.testSerialBtn); btns.addWidget(self.spAuthBtn); btns.addWidget(self.openLogsBtn); btns.addWidget(self.clearCacheBtn); runLayout.addLayout(btns)

        # --- Build Training Tab ---
        trainGroup = QtWidgets.QGroupBox("Custom Training (Capture & Train)"); trainLayout.addWidget(trainGroup); tform = QtWidgets.QFormLayout(trainGroup)
        self.dsRoot = QtWidgets.QLineEdit(os.path.join(ROOT, "dataset")); dsBrowse = QtWidgets.QPushButton("Browse…")
        dsh = QtWidgets.QHBoxLayout(); dsh.addWidget(self.dsRoot, 1); dsh.addWidget(dsBrowse); dsw = QtWidgets.QWidget(); dsw.setLayout(dsh); tform.addRow("Dataset folder", dsw)
        self.modelOut = QtWidgets.QLineEdit(os.path.join(ROOT, "custom_emotions.pkl")); self.modelSaveBtn = QtWidgets.QPushButton("Save As…"); self.modelPickBtn = QtWidgets.QPushButton("Pick existing…")
        self.modelSaveBtn.setToolTip("Choose where to save the trained model (name the .pkl file)"); self.modelPickBtn.setToolTip("Select an existing trained model (.pkl) to use")
        moh = QtWidgets.QHBoxLayout(); moh.addWidget(self.modelOut, 1); moh.addWidget(self.modelSaveBtn); moh.addWidget(self.modelPickBtn); mow = QtWidgets.QWidget(); mow.setLayout(moh); tform.addRow("Model output", mow)
        self.embModel = QtWidgets.QComboBox(); self.embModel.addItems(["Facenet512", "VGG-Face", "ArcFace"]); self.embModel.setCurrentText("Facenet512")
        self.algCombo = QtWidgets.QComboBox(); self.algCombo.addItems(["logreg", "svm"]); self.algCombo.setCurrentText("logreg")
        self.trainDetectorCombo = QtWidgets.QComboBox(); self.trainDetectorCombo.addItems(["opencv", "retinaface", "mediapipe", "mtcnn", "ssd", "dlib"]); self.trainDetectorCombo.setCurrentText("opencv")
        eah = QtWidgets.QHBoxLayout(); eah.addWidget(QtWidgets.QLabel("Embedding:")); eah.addWidget(self.embModel); eah.addSpacing(16); eah.addWidget(QtWidgets.QLabel("Algo:")); eah.addWidget(self.algCombo); eah.addSpacing(16); eah.addWidget(QtWidgets.QLabel("Detector:")); eah.addWidget(self.trainDetectorCombo); eaw = QtWidgets.QWidget(); eaw.setLayout(eah); tform.addRow("Training options", eaw)
        ash = QtWidgets.QHBoxLayout(); self.autoSearch = QtWidgets.QCheckBox("Auto pick best (grid search)"); self.maxPerClass = QtWidgets.QSpinBox(); self.maxPerClass.setRange(0, 10000); self.maxPerClass.setValue(0)
        self.maxPerClass.setToolTip("Limit images per class during search/training (0 = all)"); ash.addWidget(self.autoSearch); ash.addSpacing(12); ash.addWidget(QtWidgets.QLabel("Max imgs/class:")); ash.addWidget(self.maxPerClass); asw = QtWidgets.QWidget(); asw.setLayout(ash); tform.addRow("Auto optimize", asw)
        capLayout = QtWidgets.QGridLayout(); self._counts = {k: QtWidgets.QLabel("0") for k in ["happy","sad","angry","neutral","fear","disgust","surprise"]}; self._capBtns = {}; self._clrBtns = {}
        def mk_row(row: int, label: str):
            btn = QtWidgets.QPushButton(f"Capture {label.title()}"); btn.setMinimumWidth(150); btn.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
            btn.clicked.connect(lambda _=None, l=label: self._capture_label(l)); capLayout.addWidget(btn, row, 0); capLayout.addWidget(QtWidgets.QLabel("Count:"), row, 1, QtCore.Qt.AlignmentFlag.AlignRight); capLayout.addWidget(self._counts[label], row, 2)
            clr = QtWidgets.QPushButton("Clear"); clr.setToolTip(f"Delete all {label} pictures in the dataset"); clr.clicked.connect(lambda _=None, l=label: self._clear_label(l)); capLayout.addWidget(clr, row, 3)
            self._capBtns[label] = btn; self._clrBtns[label] = clr
        for i, lab in enumerate(["happy","sad","angry","neutral","fear","disgust","surprise"]): mk_row(i, lab)
        capLayout.setColumnStretch(0, 1); capw = QtWidgets.QWidget(); capw.setLayout(capLayout); tform.addRow("Capture", capw)
        self.trainBtn = QtWidgets.QPushButton("Train model"); self.useCustomModel = QtWidgets.QCheckBox("Use this model when starting"); self.previewBtn = QtWidgets.QPushButton("Preview dataset…"); self.clearAllBtn = QtWidgets.QPushButton("Clear all pics")
        tuh = QtWidgets.QHBoxLayout(); tuh.addWidget(self.trainBtn); tuh.addWidget(self.previewBtn); tuh.addWidget(self.clearAllBtn); tuw = QtWidgets.QWidget(); tuw.setLayout(tuh); tform.addRow("Actions", tuw); tform.addRow(self.useCustomModel)
        self.trainProgress = QtWidgets.QProgressBar(); self.trainProgress.setTextVisible(True); self.trainProgress.setVisible(False); tform.addRow("Progress", self.trainProgress)
        self.lastPreview = QtWidgets.QLabel(); self.lastPreview.setFixedSize(200, 150); self.lastPreview.setStyleSheet("background:#222; border:1px solid #444"); tform.addRow("Last capture", self.lastPreview)

        # --- Build Report Tab ---
        self.recorder = SessionRecorder()
        repCards = QtWidgets.QHBoxLayout()
        def _kpi_card(title: str) -> Tuple[QtWidgets.QFrame, QtWidgets.QLabel]:
            card = QtWidgets.QFrame(); card.setObjectName("Card"); vl = QtWidgets.QVBoxLayout(card)
            lbl = QtWidgets.QLabel("0"); lbl.setObjectName("Kpi"); sub = QtWidgets.QLabel(title); sub.setObjectName("KpiSub")
            vl.addWidget(lbl, alignment=QtCore.Qt.AlignmentFlag.AlignRight); vl.addWidget(sub, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
            return card, lbl
        c, self.repDurationLbl = _kpi_card("DURATION"); repCards.addWidget(c)
        c, self.repChangesLbl = _kpi_card("EMOTION CHANGES"); repCards.addWidget(c)
        c, self.repPlaysLbl = _kpi_card("SONGS PLAYED"); repCards.addWidget(c)
        c, self.repTopLbl = _kpi_card("TOP EMOTION"); repCards.addWidget(c)
        reportLayout.addLayout(repCards)
        repMain = QtWidgets.QHBoxLayout()
        repCharts = QtWidgets.QVBoxLayout()
        if HAS_MPL:
            # Use palette colors for charts
            palette = self.palette()
            bg2_color = palette.color(QtGui.QPalette.ColorRole.Window).name()
            fg0_color = palette.color(QtGui.QPalette.ColorRole.WindowText).name()
            fg1_color = palette.color(QtGui.QPalette.ColorRole.Text).name()
            border_color = palette.color(QtGui.QPalette.ColorRole.Mid).name()
            accent_color = "#00C2A8" # Re-declare for safety
            self._fig = Figure(figsize=(6, 6), dpi=100, facecolor=bg2_color, edgecolor=fg0_color)
            self._fig.set_tight_layout(True)
            self._ax1 = self._fig.add_subplot(2, 1, 1)
            self._ax2 = self._fig.add_subplot(2, 1, 2)
            for ax in (self._ax1, self._ax2):
                ax.set_facecolor(bg2_color)
                ax.tick_params(axis='x', colors=fg1_color); ax.tick_params(axis='y', colors=fg1_color)
                ax.spines['bottom'].set_color(border_color); ax.spines['top'].set_color(border_color); ax.spines['left'].set_color(border_color); ax.spines['right'].set_color(border_color)
                ax.yaxis.label.set_color(fg1_color); ax.xaxis.label.set_color(fg1_color); ax.title.set_color(fg0_color)
            self._canvas = FigureCanvas(self._fig)
            repCharts.addWidget(self._canvas)
        else:
            mpl_missing = QtWidgets.QLabel("Matplotlib not found. Charts disabled.\nInstall with: pip install matplotlib")
            mpl_missing.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter); repCharts.addWidget(mpl_missing)
        repMain.addLayout(repCharts, 2)
        repDetails = QtWidgets.QVBoxLayout(); self.repTable = QtWidgets.QTableWidget(); self.repTable.setColumnCount(4)
        self.repTable.setHorizontalHeaderLabels(["Emotion", "Changes", "Duration (s)", "Share (%)"]); self.repTable.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch); self.repTable.verticalHeader().setVisible(False)
        repDetails.addWidget(self.repTable)
        metaGroup = QtWidgets.QGroupBox("Session Info"); metaForm = QtWidgets.QFormLayout(metaGroup)
        self.repStartLbl = QtWidgets.QLabel("-"); self.repEndLbl = QtWidgets.QLabel("-"); self.repEngineLbl = QtWidgets.QLabel("-"); self.repDetectorLbl = QtWidgets.QLabel("-"); self.repModelLbl = QtWidgets.QLabel("-"); self.repMusicModeLbl = QtWidgets.QLabel("-")
        metaForm.addRow("Started:", self.repStartLbl); metaForm.addRow("Ended:", self.repEndLbl); metaForm.addRow("Engine:", self.repEngineLbl); metaForm.addRow("Detector:", self.repDetectorLbl); metaForm.addRow("Model:", self.repModelLbl); metaForm.addRow("Music Mode:", self.repMusicModeLbl)
        repDetails.addWidget(metaGroup); repMain.addLayout(repDetails, 1); reportLayout.addLayout(repMain)
        repButtons = QtWidgets.QHBoxLayout(); self.repExportCsv = QtWidgets.QPushButton("Export CSV"); self.repExportJson = QtWidgets.QPushButton("Export JSON"); self.repExportPng = QtWidgets.QPushButton("Export Chart PNG"); self.repExportHtml = QtWidgets.QPushButton("Export HTML"); self.repReset = QtWidgets.QPushButton("Reset Session Data"); self.repReset.setObjectName("Ghost")
        repButtons.addStretch(1); repButtons.addWidget(self.repExportCsv); repButtons.addWidget(self.repExportJson); repButtons.addWidget(self.repExportPng); repButtons.addWidget(self.repExportHtml); repButtons.addStretch(1); repButtons.addWidget(self.repReset); reportLayout.addLayout(repButtons)

        # --- Build Logs Tab ---
        self.statusLabel = QtWidgets.QLabel("Status: stopped")
        logsLayout.addWidget(self.statusLabel)
        self.logBox = QtWidgets.QPlainTextEdit()
        self.logBox.setReadOnly(True)
        logsLayout.addWidget(self.logBox)

        # --- Add Tabs to Widget ---
        self.tabs.addTab(runPage, "Run")
        self.tabs.addTab(trainPage, "Training")
        self.tabs.addTab(reportPage, "Report")
        self.tabs.addTab(logsPage, "Logs")

        # --- Connect Signals ---
        self.startBtn.clicked.connect(self.on_start)
        self.stopBtn.clicked.connect(self.on_stop)
        self.refreshBtn.clicked.connect(self._refresh_ports)
        browseBtn.clicked.connect(self._browse_music)
        self.clearCacheBtn.clicked.connect(self._clear_cache)
        self.testSerialBtn.clicked.connect(self._send_test)
        self.spAuthBtn.clicked.connect(self._spotify_auth)
        self.openLogsBtn.clicked.connect(self._open_logs_dialog)
        self.musicMode.currentTextChanged.connect(self._toggle_music_mode)
        self.trainBtn.clicked.connect(self._train_model)
        self.previewBtn.clicked.connect(self._preview_dataset)
        self.clearAllBtn.clicked.connect(self._clear_all_labels)
        self.modelSaveBtn.clicked.connect(self._browse_model_out_save)
        self.modelPickBtn.clicked.connect(self._browse_model_out_pick)
        self.repExportCsv.clicked.connect(self._export_report_csv)
        self.repExportJson.clicked.connect(self._export_report_json)
        self.repExportPng.clicked.connect(self._export_report_png)
        self.repExportHtml.clicked.connect(self._export_report_html)
        self.repReset.clicked.connect(self._reset_session)
        def _pick_run_model():
            self._browse_model_out_pick()
            path = self.modelOut.text().strip()
            if path:
                abs_path = os.path.abspath(os.path.expanduser(path))
                self.modelOut.setText(abs_path)
                self._refresh_model_choices(abs_path)
                self.runModelCombo.setEditText(abs_path)
        self.runPickModelBtn.clicked.connect(_pick_run_model)
        self._controlPath = os.path.join(ROOT, ".moodify_control.json")
        self._controlState = {"volume": 1.0}
        try:
            with open(self._controlPath, "w") as f: json.dump(self._controlState, f)
        except Exception: pass
        self._previewTimer = QtCore.QTimer(self); self._previewTimer.setInterval(40); self._previewTimer.timeout.connect(self._on_preview_timer)
        self._previewCap = None
        self.previewStartBtn.clicked.connect(self._start_idle_preview)
        self.previewStopBtn.clicked.connect(self._stop_idle_preview)
        self._paused = False
        self.pausePlayBtn.clicked.connect(self._toggle_pause_play)
        self.volumeSlider.valueChanged.connect(self._on_volume_changed)
        self._repTimer = QtCore.QTimer(self); self._repTimer.setSingleShot(True); self._repTimer.setInterval(250); self._repTimer.timeout.connect(self._update_report_ui)

        # --- Final Initialization ---
        self._load_settings()
        self._toggle_music_mode(self.musicMode.currentText())
        self._refresh_model_choices(self.runModelCombo.currentText())
        self._refresh_counts()
        self._start_idle_preview()
        self.log("Welcome to Moodify")
        if self.useCustomModel.isChecked(): self.log(f"Using custom model: {self.modelOut.text()}")
        else: self.log("Using default built-in model")

    def log(self, s: str):
        self.logBox.appendPlainText(s)
        self.logBox.verticalScrollBar().setValue(self.logBox.verticalScrollBar().maximum())
        try:
            if hasattr(self, "_logView") and self._logView is not None:
                self._logView.appendPlainText(s)
                self._logView.verticalScrollBar().setValue(self._logView.verticalScrollBar().maximum())
        except Exception:
            pass
        try:
            txt = str(s)
            if hasattr(self, "recorder") and self.recorder is not None:
                self.recorder.on_line(txt)
                self._schedule_report_update()
            if txt.startswith("Playing:"):
                self.nowPlaying.setText(txt.replace("Playing:", "").strip())
            elif txt.startswith("New stable emotion:"):
                part = txt.split("New stable emotion:", 1)[1].strip()
                emo = part.split(" ", 1)[0].strip()
                self.nowPlaying.setText(f"{emo}")
        except Exception:
            pass

    def _append(self, s: str):
        self.log(s)
        
    def _open_logs_dialog(self):
        try:
            if getattr(self, "_logDialog", None) is None:
                dlg = QtWidgets.QDialog(self)
                dlg.setWindowTitle("Moodify Logs")
                dlg.resize(800, 600)
                v = QtWidgets.QVBoxLayout(dlg)
                view = QtWidgets.QPlainTextEdit()
                view.setReadOnly(True)
                view.setPlainText(self.logBox.toPlainText())
                v.addWidget(view)
                self._logDialog = dlg
                self._logView = view
            self._logDialog.show()
            self._logDialog.raise_()
            self._logDialog.activateWindow()
        except Exception as e:
            self.log(f"[desktop] Failed to open logs dialog: {e}")

    # --- Report helpers ---
    def _schedule_report_update(self):
        if hasattr(self, "_repTimer"):
            self._repTimer.start()

    def _update_report_ui(self):
        try:
            s = self.recorder.summary()
            self.repStartLbl.setText(str(s.get("started", "-")))
            self.repEndLbl.setText(str(s.get("ended", "-")))
            self.repDurationLbl.setText(f"{s.get('duration_sec', 0)}s")
            self.repChangesLbl.setText(str(s.get("changes", 0)))
            self.repPlaysLbl.setText(str(s.get("plays", 0)))
            self.repTopLbl.setText(str(s.get("top_emotion", "-")))
            meta = s.get("meta", {})
            self.repEngineLbl.setText(str(meta.get("engine", "-")))
            self.repDetectorLbl.setText(str(meta.get("detector", "-")))
            self.repModelLbl.setText(str(meta.get("model", "-")))
            self.repMusicModeLbl.setText(str(meta.get("music_mode", "-")))
            counts = s.get("counts", {})
            durs = s.get("durations", {})
            total = float(s.get("duration_sec", 1.0)) or 1.0
            emos = SessionRecorder.EMOTIONS
            self.repTable.setRowCount(len(emos))
            for row, emo in enumerate(emos):
                cnt = int(counts.get(emo, 0))
                dur = float(durs.get(emo, 0.0))
                share = 100.0 * (dur / total)
                for col, val in enumerate([emo, str(cnt), f"{dur:.1f}", f"{share:.1f}"]):
                    item = QtWidgets.QTableWidgetItem(val)
                    if col > 0:
                        item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
                    self.repTable.setItem(row, col, item)
            if HAS_MPL:
                xs = list(range(len(emos)))
                ys = [float(durs.get(e, 0.0)) for e in emos]
                self._ax2.clear()
                self._ax2.bar(xs, ys, color="#00C2A8")
                self._ax2.set_xticks(xs)
                self._ax2.set_xticklabels(emos, rotation=0)
                self._ax2.set_ylabel("seconds")
                self._ax2.set_title("Time by emotion")
                self._ax1.clear()
                if self.recorder.stable_changes:
                    t0 = self.recorder.start_ts
                    times = [t0] + [e["ts"] for e in self.recorder.stable_changes]
                    emos_idx = {e:i for i, e in enumerate(emos)}
                    vals = [emos_idx[self.recorder.stable_changes[0]["emotion"]]]
                    for e in self.recorder.stable_changes:
                        vals.append(emos_idx.get(e["emotion"], 0))
                    times.append(self.recorder.end_ts or time.time())
                    vals.append(vals[-1])
                    times = [(t - t0) for t in times]
                    self._ax1.step(times, vals, where='post')
                    self._ax1.set_yticks(xs)
                    self._ax1.set_yticklabels(emos)
                    self._ax1.set_xlabel("seconds since start")
                    self._ax1.set_title("Stable emotion over time")
                self._fig.tight_layout()
                self._canvas.draw_idle()
        except Exception:
            pass

    def _export_report_csv(self):
        try:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export CSV", os.path.join(ROOT, "session.csv"), "CSV (*.csv)")
            if path:
                self.recorder.export_csv(path)
                self.log(f"[report] CSV exported: {path}")
        except Exception as e:
            self.log(f"[report] CSV export failed: {e}")

    def _export_report_json(self):
        try:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export JSON", os.path.join(ROOT, "session.json"), "JSON (*.json)")
            if path:
                self.recorder.export_json(path)
                self.log(f"[report] JSON exported: {path}")
        except Exception as e:
            self.log(f"[report] JSON export failed: {e}")

    def _export_report_png(self):
        if not HAS_MPL:
            return
        try:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export PNG", os.path.join(ROOT, "session.png"), "PNG (*.png)")
            if path:
                self._fig.savefig(path)
                self.log(f"[report] PNG exported: {path}")
        except Exception as e:
            self.log(f"[report] PNG export failed: {e}")

    def _export_report_html(self):
        """Export a self-contained HTML report with summary, table, and optional embedded chart image."""
        try:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export HTML", os.path.join(ROOT, "session_report.html"), "HTML (*.html)")
            if not path:
                return
            s = self.recorder.summary()
            counts = s.get("counts", {})
            durs = s.get("durations", {})
            total = float(s.get("duration_sec", 0.0)) or 0.0
            started = s.get("started", "-")
            ended = s.get("ended", "-")
            # Table rows
            emos = SessionRecorder.EMOTIONS
            rows = []
            for emo in emos:
                cnt = int(counts.get(emo, 0))
                dur = float(durs.get(emo, 0.0))
                pct = 0.0 if total <= 0 else (100.0 * dur / total)
                bar = f"<div style='background:#2b2b2b;width:100%;height:10px;border-radius:6px;overflow:hidden'><div style='background:#00C2A8;height:10px;width:{pct:.1f}%;'></div></div>"
                rows.append(f"<tr><td>{emo}</td><td style='text-align:right'>{cnt}</td><td style='text-align:right'>{dur:.1f}s</td><td style='text-align:right'>{pct:.1f}%</td><td>{bar}</td></tr>")
            # Optional embedded chart image from matplotlib
            img_data = ""
            if HAS_MPL:
                try:
                    buf = io.BytesIO()
                    self._fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
                    buf.seek(0)
                    b64 = base64.b64encode(buf.read()).decode("ascii")
                    img_data = f"<img alt='Report chart' style='max-width:100%;border:1px solid #2b2b2b;border-radius:8px' src='data:image/png;base64,{b64}'/>"
                except Exception:
                    img_data = ""
            # Precompute optional chart section to avoid backslashes inside f-string expressions
            chart_html = ("<div class='card'><h2>Charts</h2>" + img_data + "</div>") if img_data else ""
            html = f"""
<!doctype html>
<html><head><meta charset='utf-8'/>
<title>Moodify Session Report</title>
<style>
 body{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;background:#111;color:#eee;}}
 h1,h2{{font-weight:600}}
 table{{width:100%;border-collapse:collapse;margin:10px 0;}}
 th,td{{padding:8px;border-bottom:1px solid #333;text-align:left}}
 .meta{{color:#bbb}}
 .container{{max-width:980px;margin:24px auto;padding:0 16px}}
 .card{{background:#1a1a1a;border:1px solid #2b2b2b;border-radius:8px;padding:16px;margin:12px 0}}
 .small{{font-size:12px;color:#aaa}}
 .kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}}
 .kpi{{background:#161a22;border:1px solid #2b2b2b;border-radius:8px;padding:12px}}
 .kpi .v{{font-size:24px;font-weight:700}}
 .kpi .t{{font-size:12px;color:#9aa5b1;text-transform:uppercase;letter-spacing:.3px}}
 a.button{{display:inline-block;background:#00C2A8;color:#061116;text-decoration:none;padding:8px 12px;border-radius:6px}}
 </style></head>
<body><div class='container'>
  <h1>Moodify Session Report</h1>
  <div class='meta'>Started: {started} &nbsp;|&nbsp; Ended: {ended} &nbsp;|&nbsp; Duration: {total:.1f}s</div>
  <div class='kpis'>
    <div class='kpi'><div class='v'>{total:.1f}s</div><div class='t'>Duration</div></div>
    <div class='kpi'><div class='v'>{int(s.get('changes',0))}</div><div class='t'>Emotion changes</div></div>
    <div class='kpi'><div class='v'>{int(s.get('plays',0))}</div><div class='t'>Songs played</div></div>
    <div class='kpi'><div class='v'>{s.get('top_emotion','-')}</div><div class='t'>Top emotion</div></div>
  </div>
  <div class='card'>
    <h2>Summary</h2>
    <table><thead><tr><th>Emotion</th><th style='text-align:right'>Changes</th><th style='text-align:right'>Time</th><th style='text-align:right'>Share</th><th>Distribution</th></tr></thead>
    <tbody>{''.join(rows) if rows else '<tr><td colspan=4 class="small">No data.</td></tr>'}</tbody></table>
  </div>
    {chart_html}
  <div class='small'>Generated by Moodify Desktop</div>
</div></body></html>
"""
            with open(path, "w", encoding="utf-8") as f:
                f.write(html)
            self.log(f"[report] HTML exported: {path}")
        except Exception as e:
            self.log(f"[report] HTML export failed: {e}")

    def _reset_session(self):
        try:
            self.recorder.reset()
            self._update_report_ui()
            self.log("[report] Session cleared.")
        except Exception as e:
            self.log(f"[report] Reset failed: {e}")

    def _browse_music(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose music folder", self.musicDir.text())
        if d:
            self.musicDir.setText(d)

    def _browse_model_out_save(self):
        try:
            start_path = self.modelOut.text().strip() or os.path.join(ROOT, "custom_emotions.pkl")
            start_dir = os.path.dirname(start_path) or ROOT
            start_name = os.path.basename(start_path) or "custom_emotions.pkl"
            fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save trained model as…", os.path.join(start_dir, start_name), "Model files (*.pkl);;All Files (*)")
            if fname:
                if not fname.lower().endswith(".pkl"): fname += ".pkl"
                self.modelOut.setText(fname)
        except Exception as e:
            self.log(f"[desktop] Save model dialog failed: {e}")

    def _browse_model_out_pick(self):
        try:
            start_path = self.modelOut.text().strip() or os.path.join(ROOT, "custom_emotions.pkl")
            start_dir = os.path.dirname(start_path) or ROOT
            fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose existing model…", start_dir, "Model files (*.pkl);;All Files (*)")
            if fname:
                self.modelOut.setText(fname)
        except Exception as e:
            self.log(f"[desktop] Pick model dialog failed: {e}")

    def _clear_cache(self):
        try:
            if os.path.exists(CACHE_FILE):
                os.remove(CACHE_FILE)
                self.log("[desktop] Cleared .moodify_spotify_cache")
            else:
                self.log("[desktop] No cache file to remove")
        except Exception as e:
            self.log(f"[desktop] Failed to remove cache: {e}")

    def _refresh_ports(self):
        current = self.serialPort.currentText() if hasattr(self, "serialPort") else ""
        ports: List[str] = []
        try:
            if os.name == "nt":
                from serial.tools import list_ports
                ports = [p.device for p in list_ports.comports()]
            else:
                for pattern in ("/dev/cu.*", "/dev/tty.*"): ports.extend(glob.glob(pattern))
        except Exception:
            ports = [f"COM{i}" for i in range(1, 21)] if os.name == "nt" else []
        ports = sorted(set(ports))
        if hasattr(self, "serialPort"):
            self.serialPort.clear()
            if ports: self.serialPort.addItems(ports)
            pref = "COM3" if os.name == "nt" else "/dev/cu.SLAB_USBtoUART"
            if pref in ports:
                idx = self.serialPort.findText(pref)
                if idx >= 0: self.serialPort.setCurrentIndex(idx)
            elif current:
                self.serialPort.setEditText(current)

    def _refresh_model_choices(self, ensure: Optional[str] = None):
        if not hasattr(self, "runModelCombo"): return
        try: current = self.runModelCombo.currentText().strip()
        except Exception: current = ""
        def _add(path: Optional[str], bucket: List[str]):
            if path:
                norm = os.path.abspath(os.path.expanduser(path))
                if norm not in bucket: bucket.append(norm)
        choices: List[str] = []
        _add(ensure, choices); _add(current, choices)
        if hasattr(self, "modelOut"):
            try: _add(self.modelOut.text().strip(), choices)
            except Exception: pass
        for pattern in ("custom_emotions*.pkl", "*.pkl", "*.joblib"):
            for path in glob.glob(os.path.join(ROOT, pattern)): _add(path, choices)
        combo = self.runModelCombo
        try: combo.blockSignals(True)
        except Exception: pass
        combo.clear()
        for path in choices: combo.addItem(path)
        target = ensure or current or (choices[0] if choices else "")
        if target: combo.setEditText(os.path.abspath(os.path.expanduser(target)))
        try: combo.blockSignals(False)
        except Exception: pass

    def _toggle_music_mode(self, text: str):
        is_spotify = text.lower() == "spotify"
        try:
            self.localGroup.setVisible(not is_spotify)
            self.spGroup.setVisible(is_spotify)
            self.spAuthBtn.setVisible(is_spotify)
            self.clearCacheBtn.setVisible(is_spotify)
        except Exception: pass

    def _refresh_counts(self):
        root = self.dsRoot.text().strip()
        for k, lab in self._counts.items():
            try:
                labdir = os.path.join(root, k)
                n = sum(len(glob.glob(os.path.join(labdir, ext))) for ext in ("*.jpg","*.jpeg","*.png","*.bmp"))
                lab.setText(str(n))
            except Exception: lab.setText("0")

    def _capture_label(self, label: str):
        if self.proc and self.proc.poll() is None:
            self.log("[training] Stop running session before capturing.")
            return
        try:
            import cv2, time, os
            idx = int(self.cameraIndex.value())
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                self.log(f"[training] Failed to open camera index {idx}"); return
            for _ in range(3): cap.read()
            ok, frame = cap.read(); cap.release()
            if not ok or frame is None:
                self.log("[training] Failed to capture frame"); return
            root = self.dsRoot.text().strip(); labdir = os.path.join(root, label); os.makedirs(labdir, exist_ok=True)
            ts = int(time.time()*1000); outp = os.path.join(labdir, f"{ts}.jpg"); cv2.imwrite(outp, frame)
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); h, w, ch = rgb.shape
                qfmt = QtGui.QImage.Format.Format_RGB888
                qimg = QtGui.QImage(rgb.data, w, h, ch*w, qfmt)
                pix = QtGui.QPixmap.fromImage(qimg).scaled(self.lastPreview.width(), self.lastPreview.height(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
                self.lastPreview.setPixmap(pix)
            except Exception: pass
            self.log(f"[training] Captured {label}: {outp}"); self._refresh_counts()
        except Exception as e:
            self.log(f"[training] Capture failed: {e}")

    def _clear_label(self, label: str):
        root = self.dsRoot.text().strip(); labdir = os.path.join(root, label)
        if not os.path.isdir(labdir):
            self.log(f"[training] No folder to clear for {label}"); return
        mb = QtWidgets.QMessageBox(self); mb.setIcon(QtWidgets.QMessageBox.Icon.Warning); mb.setWindowTitle("Confirm clear")
        mb.setText(f"Delete all images in '{label}'?"); mb.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
        if mb.exec() != QtWidgets.QMessageBox.StandardButton.Yes: return
        removed = 0
        try:
            for ext in ("*.jpg","*.jpeg","*.png","*.bmp"):
                for p in glob.glob(os.path.join(labdir, ext)):
                    try: os.remove(p); removed += 1
                    except Exception: pass
            self.log(f"[training] Cleared {removed} files from {labdir}")
        finally: self._refresh_counts()

    def _clear_all_labels(self):
        for lab in ["happy","sad","angry","neutral","fear","disgust","surprise"]: self._clear_label(lab)

    def _preview_dataset(self):
        dlg = QtWidgets.QDialog(self); dlg.setWindowTitle("Dataset Preview"); dlg.resize(800, 600)
        v = QtWidgets.QVBoxLayout(dlg); toolbar = QtWidgets.QHBoxLayout(); openBtn = QtWidgets.QPushButton("Open folder…"); refreshBtn = QtWidgets.QPushButton("Refresh")
        toolbar.addWidget(openBtn); toolbar.addWidget(refreshBtn); toolbar.addStretch(1); v.addLayout(toolbar)
        scroll = QtWidgets.QScrollArea(); scroll.setWidgetResizable(True); container = QtWidgets.QWidget(); grid = QtWidgets.QGridLayout(container); scroll.setWidget(container); v.addWidget(scroll, 1)
        root = self.dsRoot.text().strip(); labels = ["happy","sad","angry","neutral","fear","disgust","surprise"]
        def load_grid():
            while grid.count():
                item = grid.takeAt(0)
                if item.widget(): item.widget().deleteLater()
            row = 0
            for lab in labels:
                gb = QtWidgets.QGroupBox(lab.title()); gl = QtWidgets.QGridLayout(gb)
                imgs = [p for ext in ("*.jpg","*.jpeg","*.png","*.bmp") for p in glob.glob(os.path.join(root, lab, ext))]
                for i, p in enumerate(sorted(imgs)[-30:]):
                    pix = QtGui.QPixmap(p)
                    if not pix.isNull():
                        thumb = pix.scaled(120, 90, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
                        lbl = QtWidgets.QLabel(); lbl.setPixmap(thumb); lbl.setToolTip(p); gl.addWidget(lbl, i // 6, i % 6)
                grid.addWidget(gb, row, 0); row += 1
        def open_folder():
            try: QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(root))
            except Exception: pass
        openBtn.clicked.connect(open_folder); refreshBtn.clicked.connect(load_grid); load_grid(); dlg.exec()

    def _train_model(self):
        if self.proc and self.proc.poll() is None: self.log("[training] Stop running session before training."); return
        if self._trainProc and self._trainProc.poll() is None: self.log("[training] Training already in progress."); return
        ds = self.dsRoot.text().strip(); model_out = os.path.abspath(os.path.expanduser(self.modelOut.text().strip())); self.modelOut.setText(model_out)
        emb = self.embModel.currentText().strip(); algo = self.algCombo.currentText().strip()
        args = [PY_EXE, os.path.join(ROOT, "custom_emotion_trainer.py"), "--data-dir", ds, "--model-out", model_out]
        if self.autoSearch.isChecked():
            args += ["--auto-search"]
            if (mpc := int(self.maxPerClass.value())) > 0: args += ["--max-per-class", str(mpc)]
        else:
            args += ["--embedding-model", emb, "--algo", algo, "--detector-backend", self.trainDetectorCombo.currentText().strip()]
        self.log("[training] Launching: " + shlex.join(args))
        try:
            self._trainProc = subprocess.Popen(args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            self._trainReader = LogReader(self._trainProc)
            self._trainReader.line.connect(self.log)
            def _done(rc: int):
                self.log(f"[training] Training finished with code {rc}")
                if rc == 0 and self.useCustomModel.isChecked(): self.log(f"[training] Model ready: {model_out}")
                self._refresh_model_choices(self.modelOut.text())
                self._set_training_ui(False)
                time.sleep(0.05)
                self._trainReader = None; self._trainProc = None
            self._trainReader.exited.connect(_done); self._trainReader.start(); self._set_training_ui(True)
            self.trainProgress.setRange(0, 0); self.trainProgress.setFormat("Training…"); self.trainProgress.setVisible(True); self.statusLabel.setText("Status: training")
        except Exception as e:
            self.log(f"[training] Failed to start trainer: {e}")

    def _warn(self, title: str, text: str):
        try:
            mb = QtWidgets.QMessageBox(self); mb.setIcon(QtWidgets.QMessageBox.Icon.Warning); mb.setWindowTitle(title); mb.setText(text); mb.exec()
        except Exception:
            self.log(f"[warn] {title}: {text}")

    def _set_running_ui(self, running: bool):
        self.startBtn.setEnabled(not running)
        self.stopBtn.setEnabled(running)
        widgets_to_toggle = [
            self.serialPort, self.baud, self.serialLog, self.display, self.cameraIndex, self.minInterval, self.pauseNoFace,
            self.musicMode, self.musicDir, self.spDevice, self.spId, self.spSecret, self.spRedirect, self.spHappy, self.spSad,
            self.spAngry, self.spNeutral, self.spFear, self.spDisgust, self.spSurprise, self.runUseModel, self.runModelCombo,
            self.runPickModelBtn, self.dsRoot, self.modelOut, self.embModel, self.algCombo, self.trainBtn, self.engineCombo, self.detectorCombo
        ]
        for w in widgets_to_toggle: w.setEnabled(not running)
        self.previewGroup.setEnabled(not running)
        if running: self._stop_idle_preview()
        elif not self._previewTimer.isActive(): self._start_idle_preview()

    def _set_training_ui(self, running: bool):
        widgets_to_toggle = [self.dsRoot, self.modelOut, self.embModel, self.algCombo, self.trainBtn, self.previewBtn, self.clearAllBtn, self.autoSearch, self.maxPerClass, self.trainDetectorCombo]
        for w in widgets_to_toggle: w.setEnabled(not running)
        for d in (self._capBtns, self._clrBtns):
            for btn in d.values(): btn.setEnabled(not running)
        self.trainProgress.setVisible(running)
        if not running:
            self.trainProgress.setRange(0, 1); self.trainProgress.setValue(0)
            if not (self.proc and self.proc.poll() is None): self.statusLabel.setText("Status: stopped")

    def on_start(self):
        if self.proc and self.proc.poll() is None: self.log("[desktop] Already running"); return
        try:
            if self.runUseModel.isChecked():
                self.useCustomModel.setChecked(True)
                if rp := self.runModelCombo.currentText().strip():
                    abs_path = os.path.abspath(os.path.expanduser(rp))
                    self.runModelCombo.setEditText(abs_path); self.modelOut.setText(abs_path)
                    self._refresh_model_choices(abs_path)
        except Exception: pass
        if not (port := self.serialPort.currentText().strip()):
            self._warn("Missing serial port", f"Please select a serial port (e.g., {'COM3' if os.name == 'nt' else '/dev/cu.SLAB_USBtoUART'})."); return
        if self.musicMode.currentText().lower() == "spotify" and (not self.spId.text().strip() or not self.spSecret.text().strip() or not self.spRedirect.text().strip()):
            self._warn("Spotify settings incomplete", "Client ID, Client Secret, and Redirect URI are required for Spotify mode."); return
        args = [PY_EXE, SCRIPT, "--serial-port", port, "--baud", self.baud.text().strip() or "115200"]
        if self.serialLog.isChecked(): args.append("--serial-log")
        if self.display.isChecked(): args.append("--display")
        args += ["--camera-index", str(self.cameraIndex.value()), "--min-interval", f"{self.minInterval.value():.2f}", "--video-backend", self.videoBackend.currentText().strip().lower() or "auto"]
        if self.pauseNoFace.isChecked(): args += ["--no-face-behavior", "pause"]
        engine_map = {"Auto": "auto", "DeepFace CNN": "deepface", "Custom model": "custom"}
        args += ["--emotion-engine", engine_map.get(self.engineCombo.currentText(), "auto"), "--detector-backend", self.detectorCombo.currentText().strip()]
        if self.musicMode.currentText().lower() == "spotify":
            args.append("--spotify")
            if dev := self.spDevice.text().strip(): args += ["--spotify-device", dev]
            if cid := self.spId.text().strip(): args += ["--spotify-client-id", cid]
            if sec := self.spSecret.text().strip(): args += ["--spotify-client-secret", sec]
            if ru := self.spRedirect.text().strip(): args += ["--spotify-redirect-uri", ru]; self.log(f"[desktop] Using Spotify redirect: {ru}")
            for key, widget in (("happy", self.spHappy), ("sad", self.spSad), ("angry", self.spAngry), ("neutral", self.spNeutral), ("fear", self.spFear), ("disgust", self.spDisgust), ("surprise", self.spSurprise)):
                if val := widget.text().strip(): args += [f"--sp-{key}", val]
        else:
            args += ["--music-dir", self.musicDir.text().strip()]
        if self.useCustomModel.isChecked():
            if not (model_path := self.modelOut.text().strip()):
                self._warn("Custom model", "Please choose a model file (or disable 'Use this model')."); return
            if not os.path.isfile(model_path):
                self._warn("Custom model", f"Model file not found:\n{model_path}"); return
            args += ["--custom-classifier", model_path, "--embedding-model", self.embModel.currentText().strip()]
        args += ["--control-file", self._controlPath]
        self.log("[desktop] Launching: " + shlex.join(args))
        try:
            self.proc = subprocess.Popen(args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            self.reader = LogReader(self.proc); self.reader.line.connect(self.log); self.reader.exited.connect(self._on_exited); self.reader.start()
            self.statusLabel.setText("Status: running"); self._set_running_ui(True)
            meta = {"serial_port": port, "baud": self.baud.text().strip() or "115200", "video_backend": self.videoBackend.currentText().strip(), "engine": self.engineCombo.currentText(), "detector": self.detectorCombo.currentText(), "music_mode": self.musicMode.currentText(), "model": (self.modelOut.text().strip() if self.useCustomModel.isChecked() else 'DeepFace CNN')}
            self.recorder.session_start(meta); self._schedule_report_update()
        except Exception as e:
            self.log(f"[desktop] Failed to start: {e}")

    def _on_exited(self, rc: int):
        self.statusLabel.setText(f"Status: stopped (code {rc})")
        self._set_running_ui(False)
        if hasattr(self, 'recorder'):
            self.recorder.session_stop()
            self._schedule_report_update()

    def on_stop(self):
        if not self.proc or self.proc.poll() is not None: self.log("[desktop] Not running"); return
        try:
            if self.reader: self.reader.stop()
            self.proc.terminate()
            try: self.proc.wait(timeout=2)
            except Exception: self.proc.kill()
            self.log("[desktop] Stopped.")
        except Exception as e:
            self.log(f"[desktop] Stop failed: {e}")
        finally:
            self.statusLabel.setText("Status: stopped"); self._set_running_ui(False)
            if hasattr(self, 'recorder'):
                self.recorder.session_stop()
                self._schedule_report_update()

    def _start_idle_preview(self):
        if self.proc and self.proc.poll() is None: return
        try:
            import cv2
            self._stop_idle_preview()
            idx = int(self.cameraIndex.value())
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened(): self.log(f"[preview] Failed to open camera index {idx}"); return
            self._previewCap = cap; self._previewTimer.start()
        except Exception as e:
            self.log(f"[preview] Start failed: {e}")

    def _stop_idle_preview(self):
        if self._previewTimer.isActive(): self._previewTimer.stop()
        if self._previewCap is not None: self._previewCap.release()
        self._previewCap = None

    def _on_preview_timer(self):
        if self._previewCap is None: return
        try:
            import cv2
            ok, frame = self._previewCap.read()
            if not ok or frame is None: return
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); h, w, ch = rgb.shape
            qimg = QtGui.QImage(rgb.data, w, h, ch*w, QtGui.QImage.Format.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(qimg).scaled(self.previewLabel.width(), self.previewLabel.height(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
            self.previewLabel.setPixmap(pix)
        except Exception: pass

    def _write_control(self, update: dict):
        try:
            self._controlState.update(update)
            tmp = self._controlPath + ".tmp"
            with open(tmp, "w") as f: json.dump(self._controlState, f)
            os.replace(tmp, self._controlPath)
        except Exception as e:
            self.log(f"[desktop] Failed to write control file: {e}")

    def _toggle_pause_play(self):
        self._paused = not self._paused
        self._write_control({"pause": self._paused, "play": not self._paused})
        self.pausePlayBtn.setText("Play" if self._paused else "Pause")

    def _on_volume_changed(self, val: int):
        self._write_control({"volume": max(0.0, min(1.0, float(val) / 100.0))})

    def _send_test(self):
        import serial
        if not (port := self.serialPort.currentText().strip()): self.log("[desktop] No serial port selected"); return
        try:
            with serial.Serial(port=port, baudrate=int(self.baud.text().strip() or 115200), timeout=0.2) as ser:
                time.sleep(1.6)
                try: ser.reset_input_buffer()
                except Exception: pass
                ser.write(b"TEST\n"); ser.flush(); self.log("[desktop] Sent TEST")
        except Exception as e:
            self.log(f"[desktop] Serial TEST failed: {e}")

    def _spotify_auth(self):
        if SpotifyOAuth is None: self._warn("Spotify", "spotipy is not installed in this environment."); return
        if not (cid := self.spId.text().strip()) or not (secret := self.spSecret.text().strip()) or not (redirect := self.spRedirect.text().strip()):
            self._warn("Spotify", "Please fill Client ID, Client Secret, and Redirect URI."); return
        try:
            SpotifyOAuth(client_id=cid, client_secret=secret, redirect_uri=redirect, scope="user-modify-playback-state user-read-playback-state", open_browser=True, cache_path=CACHE_FILE)
            self.log("[desktop] Opened Spotify auth in browser. Complete the flow and return.")
        except Exception as e:
            self.log(f"[desktop] Spotify auth init failed: {e}")

    def closeEvent(self, ev: QtGui.QCloseEvent):
        try: self._save_settings()
        except Exception: pass
        try: self.on_stop()
        except Exception: pass
        try: self._stop_training()
        except Exception: pass
        try: self._stop_idle_preview()
        except Exception: pass
        ev.accept()

    def _stop_training(self):
        if self._trainProc and self._trainProc.poll() is None:
            try:
                self.log("[training] Stopping training…"); self._trainProc.terminate()
                try: self._trainProc.wait(timeout=2)
                except Exception: self._trainProc.kill()
            except Exception: pass
        if self._trainReader and self._trainReader.isRunning():
            try: self._trainReader.stop(); self._trainReader.wait(500)
            except Exception: pass
        self._set_training_ui(False)

    def _load_settings(self):
        s = QtCore.QSettings("Moodify", "Desktop")
        def get(key, default=""): return s.value(key, default)
        self.serialPort.setEditText(str(get("serial/port", self.serialPort.currentText())))
        self.baud.setText(str(get("serial/baud", "115200")))
        self.serialLog.setChecked(bool(get("serial/log", True)))
        self.display.setChecked(bool(get("vision/display", True)))
        self.cameraIndex.setValue(int(get("vision/camera", self.cameraIndex.value())))
        self.minInterval.setValue(float(get("vision/min_interval", self.minInterval.value())))
        self.pauseNoFace.setChecked(bool(get("vision/pause_no_face", False)))
        self.musicMode.setCurrentText(str(get("music/mode", self.musicMode.currentText())))
        self.musicDir.setText(str(get("music/dir", self.musicDir.text())))
        self.spDevice.setText(str(get("spotify/device", self.spDevice.text())))
        self.spId.setText(str(get("spotify/id", self.spId.text())))
        self.spRedirect.setText(str(get("spotify/redirect", self.spRedirect.text())))
        self.spHappy.setText(str(get("spotify/uri_happy", ""))); self.spSad.setText(str(get("spotify/uri_sad", ""))); self.spAngry.setText(str(get("spotify/uri_angry", ""))); self.spNeutral.setText(str(get("spotify/uri_neutral", ""))); self.spFear.setText(str(get("spotify/uri_fear", ""))); self.spDisgust.setText(str(get("spotify/uri_disgust", ""))); self.spSurprise.setText(str(get("spotify/uri_surprise", "")))
        if val := str(get("run/model_file", self.runModelCombo.currentText())): self._refresh_model_choices(val)
        self.runUseModel.setChecked(bool(get("run/use_model", self.runUseModel.isChecked())))
        self.dsRoot.setText(str(get("train/dataset", self.dsRoot.text())))
        self.modelOut.setText(str(get("train/model_out", self.modelOut.text())))
        self.embModel.setCurrentText(str(get("train/emb_model", self.embModel.currentText())))
        self.algCombo.setCurrentText(str(get("train/algo", self.algCombo.currentText())))
        self.useCustomModel.setChecked(bool(get("train/use_model", False)))
        self.engineCombo.setCurrentText(str(get("detect/engine", self.engineCombo.currentText())))
        self.detectorCombo.setCurrentText(str(get("detect/detector", self.detectorCombo.currentText())))
        self.trainDetectorCombo.setCurrentText(str(get("train/detector", self.trainDetectorCombo.currentText())))

    def _save_settings(self):
        s = QtCore.QSettings("Moodify", "Desktop")
        s.setValue("serial/port", self.serialPort.currentText()); s.setValue("serial/baud", self.baud.text()); s.setValue("serial/log", self.serialLog.isChecked())
        s.setValue("vision/display", self.display.isChecked()); s.setValue("vision/camera", self.cameraIndex.value()); s.setValue("vision/min_interval", self.minInterval.value()); s.setValue("vision/pause_no_face", self.pauseNoFace.isChecked())
        s.setValue("music/mode", self.musicMode.currentText()); s.setValue("music/dir", self.musicDir.text())
        s.setValue("spotify/device", self.spDevice.text()); s.setValue("spotify/id", self.spId.text()); s.setValue("spotify/redirect", self.spRedirect.text())
        s.setValue("spotify/uri_happy", self.spHappy.text()); s.setValue("spotify/uri_sad", self.spSad.text()); s.setValue("spotify/uri_angry", self.spAngry.text()); s.setValue("spotify/uri_neutral", self.spNeutral.text()); s.setValue("spotify/uri_fear", self.spFear.text()); s.setValue("spotify/uri_disgust", self.spDisgust.text()); s.setValue("spotify/uri_surprise", self.spSurprise.text())
        s.setValue("run/model_file", self.runModelCombo.currentText()); s.setValue("run/use_model", self.runUseModel.isChecked())
        s.setValue("train/dataset", self.dsRoot.text()); s.setValue("train/model_out", self.modelOut.text()); s.setValue("train/emb_model", self.embModel.currentText()); s.setValue("train/algo", self.algCombo.currentText()); s.setValue("train/use_model", self.useCustomModel.isChecked())
        s.setValue("detect/engine", self.engineCombo.currentText()); s.setValue("detect/detector", self.detectorCombo.currentText()); s.setValue("train/detector", self.trainDetectorCombo.currentText())

if __name__ == "__main__":
    try:
        # Enable HiDPI scaling for crisp UI on macOS/Windows
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_EnableHighDpiScaling)
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)
    except Exception:
        pass
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec())
