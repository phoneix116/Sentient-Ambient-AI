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
from typing import Optional, List

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


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Moodify Desktop ({QT_BACKEND})")
        self.resize(980, 720)
        self.setMinimumSize(800, 600)

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
        # Training subprocess/reader
        self._trainProc: Optional[subprocess.Popen] = None
        self._trainReader: Optional[LogReader] = None

        cw = QtWidgets.QWidget()
        try:
            cw.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        except Exception:
            pass
        self.setCentralWidget(cw)
        layout = QtWidgets.QVBoxLayout(cw)
        # Tabs container
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs, 1)
        # Run page (main controls)
        runPage = QtWidgets.QWidget(); runLayout = QtWidgets.QVBoxLayout(runPage)
        # Training page
        trainPage = QtWidgets.QWidget(); trainLayout = QtWidgets.QVBoxLayout(trainPage)
        # Logs page
        logsPage = QtWidgets.QWidget(); logsLayout = QtWidgets.QVBoxLayout(logsPage)

        # Top form
        form = QtWidgets.QFormLayout()
        runLayout.addLayout(form)

        # Serial
        self.serialPort = QtWidgets.QComboBox()
        self.serialPort.setEditable(True)
        self._refresh_ports()
        self.baud = QtWidgets.QLineEdit("115200")
        self.serialLog = QtWidgets.QCheckBox("Show ESP32 logs")
        self.serialLog.setChecked(True)
        serialBox = QtWidgets.QHBoxLayout()
        serialBox.addWidget(self.serialPort)
        serialBox.addWidget(QtWidgets.QLabel("Baud:"))
        serialBox.addWidget(self.baud)
        serialBox.addWidget(self.serialLog)
        spw = QtWidgets.QWidget(); spw.setLayout(serialBox)
        form.addRow("Serial Port", spw)

        # Vision
        self.display = QtWidgets.QCheckBox("Show preview")
        self.display.setChecked(True)
        self.cameraIndex = QtWidgets.QSpinBox(); self.cameraIndex.setRange(0, 8); self.cameraIndex.setValue(0)
        self.minInterval = QtWidgets.QDoubleSpinBox(); self.minInterval.setRange(0.1, 10.0); self.minInterval.setSingleStep(0.1); self.minInterval.setValue(1.0)
        self.pauseNoFace = QtWidgets.QCheckBox("Pause music when no face")
        vbox = QtWidgets.QHBoxLayout()
        vbox.addWidget(self.display)
        vbox.addWidget(QtWidgets.QLabel("Camera index:"))
        vbox.addWidget(self.cameraIndex)
        vbox.addWidget(QtWidgets.QLabel("Min interval (s):"))
        vbox.addWidget(self.minInterval)
        vbox.addWidget(self.pauseNoFace)
        vw = QtWidgets.QWidget(); vw.setLayout(vbox)
        form.addRow("Vision", vw)

        # Live Preview (idle only)
        self.previewGroup = QtWidgets.QGroupBox("Live Camera Preview (idle only)")
        pv = QtWidgets.QVBoxLayout()
        self.previewLabel = QtWidgets.QLabel()
        self.previewLabel.setFixedSize(480, 360)
        self.previewLabel.setStyleSheet("background:#111;border:1px solid #333")
        hint = QtWidgets.QLabel("Preview runs when not started. Close other camera apps if blank.")
        hint.setStyleSheet("color:#888;font-size:11px")
        ph = QtWidgets.QHBoxLayout()
        self.previewStartBtn = QtWidgets.QPushButton("Start Preview")
        self.previewStopBtn = QtWidgets.QPushButton("Stop Preview")
        ph.addWidget(self.previewStartBtn)
        ph.addWidget(self.previewStopBtn)
        ph.addStretch(1)
        pv.addWidget(self.previewLabel, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)
        pv.addLayout(ph)
        pv.addWidget(hint)
        self.previewGroup.setLayout(pv)
        runLayout.addWidget(self.previewGroup)

        # Music mode
        self.musicMode = QtWidgets.QComboBox()
        self.musicMode.addItems(["Local", "Spotify"])
        form.addRow("Music Mode", self.musicMode)

        # Local music group (shown only when mode = Local)
        self.musicDir = QtWidgets.QLineEdit(os.path.join(ROOT, "media", "mp3"))
        browseBtn = QtWidgets.QPushButton("Browse…")
        browseBtn.clicked.connect(self._browse_music)
        lhb = QtWidgets.QHBoxLayout(); lhb.addWidget(QtWidgets.QLabel("Music folder")); lhb.addWidget(self.musicDir, 1); lhb.addWidget(browseBtn)
        self.localGroup = QtWidgets.QGroupBox("Local music")
        lw = QtWidgets.QWidget(); lw.setLayout(lhb)
        v_local = QtWidgets.QVBoxLayout(); v_local.addWidget(lw)
        self.localGroup.setLayout(v_local)
        form.addRow(self.localGroup)

        # Spotify fields (group, shown only when mode = Spotify)
        self.spDevice = QtWidgets.QLineEdit("MacBook")
        self.spId = QtWidgets.QLineEdit()
        self.spSecret = QtWidgets.QLineEdit(); self.spSecret.setEchoMode(QLINEEDIT_PASSWORD)
        self.spRedirect = QtWidgets.QLineEdit("http://127.0.0.1:8888/callback")
        self.spHappy = QtWidgets.QLineEdit()
        self.spSad = QtWidgets.QLineEdit()
        self.spAngry = QtWidgets.QLineEdit()
        self.spNeutral = QtWidgets.QLineEdit()
        self.spFear = QtWidgets.QLineEdit()

        self.spGroup = QtWidgets.QGroupBox("Spotify settings")
        spForm = QtWidgets.QFormLayout()
        spForm.addRow("Spotify device contains", self.spDevice)
        spForm.addRow("Spotify client id", self.spId)
        spForm.addRow("Spotify client secret", self.spSecret)
        spForm.addRow("Spotify redirect URI", self.spRedirect)
        spForm.addRow("URI happy", self.spHappy)
        spForm.addRow("URI sad", self.spSad)
        spForm.addRow("URI angry", self.spAngry)
        spForm.addRow("URI neutral", self.spNeutral)
        spForm.addRow("URI fear", self.spFear)
        self.spGroup.setLayout(spForm)
        form.addRow(self.spGroup)

        # Quick model chooser on Run tab
        modelGrp = QtWidgets.QGroupBox("Model")
        mf = QtWidgets.QFormLayout()
        self.runUseModel = QtWidgets.QCheckBox("Use custom model")
        mf.addRow(self.runUseModel)
        mh = QtWidgets.QHBoxLayout()
        self.runModelPath = QtWidgets.QLineEdit(os.path.join(ROOT, "custom_emotions.pkl"))
        self.runPickModelBtn = QtWidgets.QPushButton("Pick…")
        mh.addWidget(self.runModelPath, 1)
        mh.addWidget(self.runPickModelBtn)
        mw = QtWidgets.QWidget(); mw.setLayout(mh)
        mf.addRow("Model file", mw)
        modelGrp.setLayout(mf)
        runLayout.addWidget(modelGrp)

        # Detection options on Run tab
        detectGrp = QtWidgets.QGroupBox("Detection")
        df = QtWidgets.QFormLayout()
        self.engineCombo = QtWidgets.QComboBox()
        self.engineCombo.addItems(["Auto", "DeepFace CNN", "Custom model"])
        self.engineCombo.setCurrentText("Auto")
        self.detectorCombo = QtWidgets.QComboBox()
        self.detectorCombo.addItems(["opencv", "retinaface", "mediapipe", "mtcnn", "ssd", "dlib"]) 
        self.detectorCombo.setCurrentText("opencv")
        df.addRow("Emotion engine", self.engineCombo)
        df.addRow("Detector backend", self.detectorCombo)
        detectGrp.setLayout(df)
        runLayout.addWidget(detectGrp)

        # Buttons row
        btns = QtWidgets.QHBoxLayout()
        self.startBtn = QtWidgets.QPushButton("Start")
        self.stopBtn = QtWidgets.QPushButton("Stop")
        self.refreshBtn = QtWidgets.QPushButton("Refresh Ports")
        self.clearCacheBtn = QtWidgets.QPushButton("Clear Spotify Cache")
        self.testSerialBtn = QtWidgets.QPushButton("Send TEST")
        self.spAuthBtn = QtWidgets.QPushButton("Authorize Spotify…")
        self.openLogsBtn = QtWidgets.QPushButton("Open logs…")
        btns.addWidget(self.startBtn)
        btns.addWidget(self.stopBtn)
        btns.addStretch(1)
        btns.addWidget(self.refreshBtn)
        btns.addWidget(self.testSerialBtn)
        btns.addWidget(self.spAuthBtn)
        btns.addWidget(self.openLogsBtn)
        btns.addWidget(self.clearCacheBtn)
        runLayout.addLayout(btns)

    # (Brightness controls removed per request)

        # --- Custom Training Section ---
        trainGroup = QtWidgets.QGroupBox("Custom Training (Capture & Train)")
        tform = QtWidgets.QFormLayout()
        # Dataset dir
        self.dsRoot = QtWidgets.QLineEdit(os.path.join(ROOT, "dataset"))
        dsBrowse = QtWidgets.QPushButton("Browse…")
        def _b():
            d = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose dataset folder", self.dsRoot.text())
            if d:
                self.dsRoot.setText(d)
                self._refresh_counts()
        dsBrowse.clicked.connect(_b)
        dsh = QtWidgets.QHBoxLayout(); dsh.addWidget(self.dsRoot, 1); dsh.addWidget(dsBrowse)
        dsw = QtWidgets.QWidget(); dsw.setLayout(dsh)
        tform.addRow("Dataset folder", dsw)
        # Model out + browse buttons
        self.modelOut = QtWidgets.QLineEdit(os.path.join(ROOT, "custom_emotions.pkl"))
        self.modelSaveBtn = QtWidgets.QPushButton("Save As…")
        self.modelPickBtn = QtWidgets.QPushButton("Pick existing…")
        self.modelSaveBtn.setToolTip("Choose where to save the trained model (name the .pkl file)")
        self.modelPickBtn.setToolTip("Select an existing trained model (.pkl) to use")
        self.modelSaveBtn.clicked.connect(self._browse_model_out_save)
        self.modelPickBtn.clicked.connect(self._browse_model_out_pick)
        moh = QtWidgets.QHBoxLayout(); moh.addWidget(self.modelOut, 1); moh.addWidget(self.modelSaveBtn); moh.addWidget(self.modelPickBtn)
        mow = QtWidgets.QWidget(); mow.setLayout(moh)
        tform.addRow("Model output", mow)
    # Embedding, Algo & Detector for training
        self.embModel = QtWidgets.QComboBox(); self.embModel.addItems(["Facenet512", "VGG-Face", "ArcFace"]) ; self.embModel.setCurrentText("Facenet512")
        self.algCombo = QtWidgets.QComboBox(); self.algCombo.addItems(["logreg", "svm"]) ; self.algCombo.setCurrentText("logreg")
        self.trainDetectorCombo = QtWidgets.QComboBox(); self.trainDetectorCombo.addItems(["opencv", "retinaface", "mediapipe", "mtcnn", "ssd", "dlib"]) ; self.trainDetectorCombo.setCurrentText("opencv")
        eah = QtWidgets.QHBoxLayout()
        eah.addWidget(QtWidgets.QLabel("Embedding:")); eah.addWidget(self.embModel)
        eah.addSpacing(16)
        eah.addWidget(QtWidgets.QLabel("Algo:")); eah.addWidget(self.algCombo)
        eah.addSpacing(16)
        eah.addWidget(QtWidgets.QLabel("Detector:")); eah.addWidget(self.trainDetectorCombo)
        eaw = QtWidgets.QWidget(); eaw.setLayout(eah)
        tform.addRow("Training options", eaw)
        # Auto search option
        ash = QtWidgets.QHBoxLayout()
        self.autoSearch = QtWidgets.QCheckBox("Auto pick best (grid search)")
        self.maxPerClass = QtWidgets.QSpinBox(); self.maxPerClass.setRange(0, 10000); self.maxPerClass.setValue(0)
        self.maxPerClass.setToolTip("Limit images per class during search/training (0 = all)")
        ash.addWidget(self.autoSearch)
        ash.addSpacing(12)
        ash.addWidget(QtWidgets.QLabel("Max imgs/class:"))
        ash.addWidget(self.maxPerClass)
        asw = QtWidgets.QWidget(); asw.setLayout(ash)
        tform.addRow("Auto optimize", asw)
        # Capture row
        capLayout = QtWidgets.QGridLayout()
        self._counts = {k: QtWidgets.QLabel("0") for k in ["happy","sad","angry","neutral","fear"]}
        self._capBtns = {}
        self._clrBtns = {}
        def mk_row(row: int, label: str):
            btn = QtWidgets.QPushButton(f"Capture {label.title()}")
            btn.clicked.connect(lambda _=None, l=label: self._capture_label(l))
            capLayout.addWidget(btn, row, 0)
            capLayout.addWidget(QtWidgets.QLabel("Count:"), row, 1)
            capLayout.addWidget(self._counts[label], row, 2)
            clr = QtWidgets.QPushButton("Clear")
            clr.setToolTip(f"Delete all {label} pictures in the dataset")
            clr.clicked.connect(lambda _=None, l=label: self._clear_label(l))
            capLayout.addWidget(clr, row, 3)
            self._capBtns[label] = btn
            self._clrBtns[label] = clr
        for i, lab in enumerate(["happy","sad","angry","neutral","fear"]):
            mk_row(i, lab)
        capw = QtWidgets.QWidget(); capw.setLayout(capLayout)
        tform.addRow("Capture", capw)
        # Train & Use
        tuh = QtWidgets.QHBoxLayout()
        self.trainBtn = QtWidgets.QPushButton("Train model")
        self.useCustomModel = QtWidgets.QCheckBox("Use this model when starting")
        self.previewBtn = QtWidgets.QPushButton("Preview dataset…")
        self.clearAllBtn = QtWidgets.QPushButton("Clear all pics")
        tuh.addWidget(self.trainBtn)
        tuh.addWidget(self.previewBtn)
        tuh.addWidget(self.clearAllBtn)
        tuh.addWidget(self.useCustomModel)
        tuw = QtWidgets.QWidget(); tuw.setLayout(tuh)
        tform.addRow("Actions", tuw)
        # Training progress (indeterminate)
        self.trainProgress = QtWidgets.QProgressBar()
        self.trainProgress.setTextVisible(True)
        self.trainProgress.setVisible(False)
        tform.addRow("Progress", self.trainProgress)
        # Optional last preview
        self.lastPreview = QtWidgets.QLabel()
        self.lastPreview.setFixedSize(200, 150)
        self.lastPreview.setStyleSheet("background:#222; border:1px solid #444")
        tform.addRow("Last capture", self.lastPreview)
        trainGroup.setLayout(tform)
        trainLayout.addWidget(trainGroup)

        # Music controls
        self.musicCtrlGroup = QtWidgets.QGroupBox("Music Controls")
        mc = QtWidgets.QGridLayout()
        self.nowPlaying = QtWidgets.QLabel("Now Playing: -")
        self.nowPlaying.setStyleSheet("font-weight:500")
        self.volumeSlider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.volumeSlider.setRange(0, 100)
        self.volumeSlider.setValue(100)
        self.pausePlayBtn = QtWidgets.QPushButton("Pause")
        mc.addWidget(QtWidgets.QLabel("Now Playing:"), 0, 0)
        mc.addWidget(self.nowPlaying, 0, 1, 1, 2)
        mc.addWidget(QtWidgets.QLabel("Volume"), 1, 0)
        mc.addWidget(self.volumeSlider, 1, 1)
        mc.addWidget(self.pausePlayBtn, 1, 2)
        self.musicCtrlGroup.setLayout(mc)
        runLayout.addWidget(self.musicCtrlGroup)

        # Status + log moved to Logs tab
        self.statusLabel = QtWidgets.QLabel("Status: stopped")
        logsLayout.addWidget(self.statusLabel)
        # Model indicator
        self.modelLabel = QtWidgets.QLabel("Model: DeepFace built-in")
        logsLayout.addWidget(self.modelLabel)
        self.log = QtWidgets.QPlainTextEdit(); self.log.setReadOnly(True)
        logsLayout.addWidget(self.log, 1)

        # Assemble tabs
        self.tabs.addTab(runPage, "Run")
        self.tabs.addTab(trainPage, "Training")
        self.tabs.addTab(logsPage, "Logs")

        # Wiring signals
        self.startBtn.clicked.connect(self.on_start)
        self.stopBtn.clicked.connect(self.on_stop)
        self.refreshBtn.clicked.connect(self._refresh_ports)
        self.clearCacheBtn.clicked.connect(self._clear_cache)
        self.testSerialBtn.clicked.connect(self._send_test)
        # (Send BRIGHT button removed)
        self.spAuthBtn.clicked.connect(self._spotify_auth)
        self.openLogsBtn.clicked.connect(self._open_logs_dialog)
        self.musicMode.currentTextChanged.connect(self._toggle_music_mode)
        self._toggle_music_mode(self.musicMode.currentText())
        self.trainBtn.clicked.connect(self._train_model)
        self.previewBtn.clicked.connect(self._preview_dataset)
        self.clearAllBtn.clicked.connect(self._clear_all_labels)
        # Run tab model picker mirrors training picker result
        def _pick_run_model():
            try:
                self._browse_model_out_pick()
                self.runModelPath.setText(self.modelOut.text())
            except Exception:
                pass
        self.runPickModelBtn.clicked.connect(_pick_run_model)
        # initial counts
        self._refresh_counts()

        # Control file path/state used for music commands from GUI
        self._controlPath = os.path.join(ROOT, ".moodify_control.json")
        self._controlState = {"volume": 1.0}
        try:
            import json
            with open(self._controlPath, "w") as f:
                json.dump(self._controlState, f)
        except Exception:
            pass

        # Preview machinery
        self._previewTimer = QtCore.QTimer(self)
        self._previewTimer.setInterval(40)  # ~25 FPS target
        self._previewTimer.timeout.connect(self._on_preview_timer)
        self._previewCap = None
        self.previewStartBtn.clicked.connect(self._start_idle_preview)
        self.previewStopBtn.clicked.connect(self._stop_idle_preview)
        # Start preview by default
        self._start_idle_preview()

        # Music control wiring
        self._paused = False
        self.pausePlayBtn.clicked.connect(self._toggle_pause_play)
        self.volumeSlider.valueChanged.connect(self._on_volume_changed)

    # UI helpers
    def _append(self, s: str):
        self.log.appendPlainText(s)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())
        try:
            if hasattr(self, "_logView") and self._logView is not None:
                self._logView.appendPlainText(s)
                self._logView.verticalScrollBar().setValue(self._logView.verticalScrollBar().maximum())
        except Exception:
            pass
        # Lightweight parse for Now Playing / stable emotion lines
        try:
            txt = str(s)
            if txt.startswith("Playing:"):
                # Format: Playing: <emotion> (<file>)
                self.nowPlaying.setText(txt.replace("Playing:", "").strip())
            elif txt.startswith("New stable emotion:"):
                # New stable emotion: happy -> ...
                part = txt.split("New stable emotion:", 1)[1].strip()
                emo = part.split(" ", 1)[0].strip()
                self.nowPlaying.setText(f"{emo}")
        except Exception:
            pass

    def _open_logs_dialog(self):
        try:
            if getattr(self, "_logDialog", None) is None:
                dlg = QtWidgets.QDialog(self)
                dlg.setWindowTitle("Moodify Logs")
                dlg.resize(800, 600)
                v = QtWidgets.QVBoxLayout(dlg)
                view = QtWidgets.QPlainTextEdit()
                view.setReadOnly(True)
                view.setPlainText(self.log.toPlainText())
                v.addWidget(view)
                self._logDialog = dlg
                self._logView = view
            self._logDialog.show()
            self._logDialog.raise_()
            self._logDialog.activateWindow()
        except Exception as e:
            self._append(f"[desktop] Failed to open logs dialog: {e}")

    def _browse_music(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose music folder", self.musicDir.text())
        if d:
            self.musicDir.setText(d)

    def _browse_model_out_save(self):
        """Let user name and choose where to save the trained model (.pkl)."""
        try:
            start_path = self.modelOut.text().strip() or os.path.join(ROOT, "custom_emotions.pkl")
            start_dir = os.path.dirname(start_path) or ROOT
            start_name = os.path.basename(start_path) or "custom_emotions.pkl"
            fname, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save trained model as…",
                os.path.join(start_dir, start_name),
                "Model files (*.pkl);;All Files (*)",
            )
            if fname:
                if not fname.lower().endswith(".pkl"):
                    fname += ".pkl"
                self.modelOut.setText(fname)
        except Exception as e:
            self._append(f"[desktop] Save model dialog failed: {e}")

    def _browse_model_out_pick(self):
        """Pick an existing trained model (.pkl) to use."""
        try:
            start_path = self.modelOut.text().strip() or os.path.join(ROOT, "custom_emotions.pkl")
            start_dir = os.path.dirname(start_path) or ROOT
            fname, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Choose existing model…",
                start_dir,
                "Model files (*.pkl);;All Files (*)",
            )
            if fname:
                self.modelOut.setText(fname)
        except Exception as e:
            self._append(f"[desktop] Pick model dialog failed: {e}")

    def _clear_cache(self):
        try:
            if os.path.exists(CACHE_FILE):
                os.remove(CACHE_FILE)
                self._append("[desktop] Cleared .moodify_spotify_cache")
            else:
                self._append("[desktop] No cache file to remove")
        except Exception as e:
            self._append(f"[desktop] Failed to remove cache: {e}")

    def _refresh_ports(self):
        """Populate serial port combo with macOS-style cu.* / tty.* devices."""
        current = self.serialPort.currentText() if hasattr(self, "serialPort") else ""
        ports: List[str] = []
        for pattern in ("/dev/cu.*", "/dev/tty.*"):
            ports.extend(glob.glob(pattern))
        ports = sorted(set(ports))
        if hasattr(self, "serialPort"):
            self.serialPort.clear()
            if ports:
                self.serialPort.addItems(ports)
            # Suggest common default if present
            pref = "/dev/cu.SLAB_USBtoUART"
            if pref in ports:
                idx = self.serialPort.findText(pref)
                if idx >= 0:
                    self.serialPort.setCurrentIndex(idx)
            elif current:
                self.serialPort.setEditText(current)

    def _toggle_music_mode(self, text: str):
        is_spotify = text.lower() == "spotify"
        # Show/hide sections for Local vs Spotify
        try:
            self.localGroup.setVisible(not is_spotify)
            self.spGroup.setVisible(is_spotify)
        except Exception:
            pass
        # Spotify-specific buttons visible only in Spotify mode
        try:
            self.spAuthBtn.setVisible(is_spotify)
            self.clearCacheBtn.setVisible(is_spotify)
        except Exception:
            pass
        # Keep enabling logic for safety
        for w in (self.musicDir,):
            try:
                w.setEnabled(not is_spotify)
            except Exception:
                pass
        for w in (self.spDevice, self.spId, self.spSecret, self.spRedirect,
                  self.spHappy, self.spSad, self.spAngry, self.spNeutral, self.spFear):
            try:
                w.setEnabled(is_spotify)
            except Exception:
                pass

    # Training helpers
    def _refresh_counts(self):
        import os, glob
        root = self.dsRoot.text().strip()
        for k, lab in self._counts.items():
            try:
                labdir = os.path.join(root, k)
                n = 0
                for ext in ("*.jpg","*.jpeg","*.png","*.bmp"):
                    n += len(glob.glob(os.path.join(labdir, ext)))
                lab.setText(str(n))
            except Exception:
                lab.setText("0")

    def _capture_label(self, label: str):
        # Avoid conflicts if process is running
        if self.proc and self.proc.poll() is None:
            self._append("[training] Stop running session before capturing.")
            return
        try:
            import cv2, time, os
            idx = int(self.cameraIndex.value())
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                self._append(f"[training] Failed to open camera index {idx}")
                return
            # grab a couple frames for auto exposure
            for _ in range(3):
                cap.read()
            ok, frame = cap.read()
            cap.release()
            if not ok or frame is None:
                self._append("[training] Failed to capture frame")
                return
            # save
            root = self.dsRoot.text().strip()
            labdir = os.path.join(root, label)
            os.makedirs(labdir, exist_ok=True)
            ts = int(time.time()*1000)
            outp = os.path.join(labdir, f"{ts}.jpg")
            cv2.imwrite(outp, frame)
            # preview
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                # Determine QImage format enum across PyQt6/PySide6
                try:
                    qfmt = QtGui.QImage.Format.Format_RGB888
                except AttributeError:
                    qfmt = QtGui.QImage.Format_RGB888
                qimg = QtGui.QImage(rgb.data, w, h, ch*w, qfmt)
                pix = QtGui.QPixmap.fromImage(qimg).scaled(
                    self.lastPreview.width(), self.lastPreview.height(),
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation,
                )
                self.lastPreview.setPixmap(pix)
            except Exception:
                pass
            self._append(f"[training] Captured {label}: {outp}")
            self._refresh_counts()
        except Exception as e:
            self._append(f"[training] Capture failed: {e}")

    def _clear_label(self, label: str):
        import os, glob, shutil
        root = self.dsRoot.text().strip()
        labdir = os.path.join(root, label)
        if not os.path.isdir(labdir):
            self._append(f"[training] No folder to clear for {label}")
            return
        mb = QtWidgets.QMessageBox(self)
        mb.setIcon(QtWidgets.QMessageBox.Icon.Warning if hasattr(QtWidgets.QMessageBox, 'Icon') else QtWidgets.QMessageBox.Warning)
        mb.setWindowTitle("Confirm clear")
        mb.setText(f"Delete all images in '{label}'?")
        mb.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
        r = mb.exec()
        if r != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        removed = 0
        try:
            for ext in ("*.jpg","*.jpeg","*.png","*.bmp"):
                for p in glob.glob(os.path.join(labdir, ext)):
                    try:
                        os.remove(p)
                        removed += 1
                    except Exception:
                        pass
            self._append(f"[training] Cleared {removed} files from {labdir}")
        finally:
            self._refresh_counts()

    def _clear_all_labels(self):
        for lab in ["happy","sad","angry","neutral","fear"]:
            self._clear_label(lab)

    def _preview_dataset(self):
        # Build a simple dialog with thumbnails grouped by emotion
        import os, glob
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Dataset Preview")
        dlg.resize(800, 600)
        v = QtWidgets.QVBoxLayout(dlg)
        toolbar = QtWidgets.QHBoxLayout()
        openBtn = QtWidgets.QPushButton("Open folder…")
        refreshBtn = QtWidgets.QPushButton("Refresh")
        toolbar.addWidget(openBtn)
        toolbar.addWidget(refreshBtn)
        toolbar.addStretch(1)
        v.addLayout(toolbar)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        container = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(container)
        scroll.setWidget(container)
        v.addWidget(scroll, 1)

        root = self.dsRoot.text().strip()
        labels = ["happy","sad","angry","neutral","fear"]

        def load_grid():
            # Clear previous
            while grid.count():
                item = grid.takeAt(0)
                w = item.widget()
                if w:
                    w.deleteLater()
            row = 0
            for lab in labels:
                gb = QtWidgets.QGroupBox(lab.title())
                gl = QtWidgets.QGridLayout(gb)
                imgs = []
                for ext in ("*.jpg","*.jpeg","*.png","*.bmp"):
                    imgs.extend(glob.glob(os.path.join(root, lab, ext)))
                imgs = sorted(imgs)[-30:]  # show up to 30 most recent
                col = 0; r = 0
                for p in imgs:
                    pix = QtGui.QPixmap(p)
                    if not pix.isNull():
                        thumb = pix.scaled(120, 90, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
                        lbl = QtWidgets.QLabel()
                        lbl.setPixmap(thumb)
                        lbl.setToolTip(p)
                        gl.addWidget(lbl, r, col)
                        col += 1
                        if col >= 6:
                            col = 0; r += 1
                grid.addWidget(gb, row, 0)
                row += 1

        def open_folder():
            try:
                QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(root))
            except Exception:
                pass

        openBtn.clicked.connect(open_folder)
        refreshBtn.clicked.connect(load_grid)
        load_grid()
        dlg.exec()

    def _train_model(self):
        # Avoid conflicts if process is running
        if self.proc and self.proc.poll() is None:
            self._append("[training] Stop running session before training.")
            return
        if self._trainProc and self._trainProc.poll() is None:
            self._append("[training] Training already in progress.")
            return
        ds = self.dsRoot.text().strip()
        model_out = os.path.abspath(os.path.expanduser(self.modelOut.text().strip()))
        # Normalize field so user sees the real path
        try:
            self.modelOut.setText(model_out)
        except Exception:
            pass
        emb = self.embModel.currentText().strip()
        algo = self.algCombo.currentText().strip()
        args = [PY_EXE, os.path.join(ROOT, "custom_emotion_trainer.py"),
                "--data-dir", ds, "--model-out", model_out]
        if self.autoSearch.isChecked():
            args += ["--auto-search"]
            mpc = int(self.maxPerClass.value())
            if mpc > 0:
                args += ["--max-per-class", str(mpc)]
        else:
            args += ["--embedding-model", emb, "--algo", algo,
                     "--detector-backend", self.trainDetectorCombo.currentText().strip()]
        self._append("[training] Launching: " + shlex.join(args))
        try:
            self._trainProc = subprocess.Popen(args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            self._trainReader = LogReader(self._trainProc)
            self._trainReader.line.connect(self._append)
            def _done(rc: int):
                self._append(f"[training] Training finished with code {rc}")
                if rc == 0 and self.useCustomModel.isChecked():
                    self._append(f"[training] Model ready: {model_out}")
                # Re-enable UI and clear refs
                self._set_training_ui(False)
                try:
                    # small delay to ensure thread exit signals processed
                    time.sleep(0.05)
                except Exception:
                    pass
                self._trainReader = None
                self._trainProc = None
            self._trainReader.exited.connect(_done)
            self._trainReader.start()
            self._set_training_ui(True)
            try:
                # Indeterminate spinner
                self.trainProgress.setRange(0, 0)
                self.trainProgress.setFormat("Training…")
                self.trainProgress.setVisible(True)
                self.statusLabel.setText("Status: training")
            except Exception:
                pass
        except Exception as e:
            self._append(f"[training] Failed to start trainer: {e}")

    def _warn(self, title: str, text: str):
        try:
            mb = QtWidgets.QMessageBox(self)
            if hasattr(QtWidgets.QMessageBox, "Icon"):
                mb.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            else:
                mb.setIcon(QtWidgets.QMessageBox.Warning)
            mb.setWindowTitle(title)
            mb.setText(text)
            mb.exec()
        except Exception:
            self._append(f"[warn] {title}: {text}")

    def _set_running_ui(self, running: bool):
        self.startBtn.setEnabled(not running)  # enable when not running
        self.stopBtn.setEnabled(running)
        for w in (
            self.serialPort, self.baud, self.serialLog,
            self.display, self.cameraIndex, self.minInterval, self.pauseNoFace,
            self.musicMode, self.musicDir, self.spDevice, self.spId, self.spSecret,
            self.spRedirect, self.spHappy, self.spSad, self.spAngry, self.spNeutral, self.spFear,
            self.dsRoot, self.modelOut, self.embModel, self.algCombo, self.trainBtn, self.engineCombo, self.detectorCombo,
        ):
            try:
                w.setEnabled(not running if w is not self.spSecret else (not running and self.musicMode.currentText().lower() == "spotify"))
            except Exception:
                pass
        # Preview controls disabled when running
        try:
            self.previewGroup.setEnabled(not running)
            if running:
                self._stop_idle_preview()
            elif not self._previewTimer.isActive():
                self._start_idle_preview()
        except Exception:
            pass

    def _set_training_ui(self, running: bool):
        for w in (self.dsRoot, self.modelOut, self.embModel, self.algCombo, self.trainBtn, self.previewBtn, self.clearAllBtn, self.autoSearch, self.maxPerClass, self.trainDetectorCombo):
            try:
                w.setEnabled(not running)
            except Exception:
                pass
        # Disable capture/clear buttons during training
        for d in (self._capBtns, self._clrBtns):
            for _, btn in d.items():
                try:
                    btn.setEnabled(not running)
                except Exception:
                    pass
        # Progress bar visibility
        try:
            if not running:
                self.trainProgress.setVisible(False)
                self.trainProgress.setRange(0, 1)
                self.trainProgress.setValue(0)
                # Restore status only if not running main process
                if not (self.proc and self.proc.poll() is None):
                    self.statusLabel.setText("Status: stopped")
        except Exception:
            pass

    # Process control
    def on_start(self):
        if self.proc and self.proc.poll() is None:
            self._append("[desktop] Already running")
            return
        # If Run tab has 'Use custom model' checked, sync into training fields
        try:
            if hasattr(self, "runUseModel") and self.runUseModel.isChecked():
                self.useCustomModel.setChecked(True)
                rp = self.runModelPath.text().strip()
                if rp:
                    self.modelOut.setText(os.path.abspath(os.path.expanduser(rp)))
        except Exception:
            pass
        # Basic validation
        port = self.serialPort.currentText().strip()
        if not port:
            self._warn("Missing serial port", "Please select a serial port (e.g., /dev/cu.SLAB_USBtoUART).")
            return
        if self.musicMode.currentText().lower() == "spotify":
            if not self.spId.text().strip() or not self.spSecret.text().strip() or not self.spRedirect.text().strip():
                self._warn("Spotify settings incomplete", "Client ID, Client Secret, and Redirect URI are required for Spotify mode.")
                return
        args = [PY_EXE, SCRIPT,
                "--serial-port", port,
                "--baud", self.baud.text().strip() or "115200"]
        if self.serialLog.isChecked():
            args.append("--serial-log")
        if self.display.isChecked():
            args.append("--display")
        args += ["--camera-index", str(self.cameraIndex.value()),
                 "--min-interval", f"{self.minInterval.value():.2f}"]
        if self.pauseNoFace.isChecked():
            args += ["--no-face-behavior", "pause"]

        # Detection engine and face detector backend
        try:
            engine_map = {
                "Auto": "auto",
                "DeepFace CNN": "deepface",
                "Custom model": "custom",
            }
            eng_val = engine_map.get(self.engineCombo.currentText(), "auto")
            args += ["--emotion-engine", eng_val, "--detector-backend", self.detectorCombo.currentText().strip()]
        except Exception:
            pass

        if self.musicMode.currentText().lower() == "spotify":
            args.append("--spotify")
            if self.spDevice.text().strip():
                args += ["--spotify-device", self.spDevice.text().strip()]
            if self.spId.text().strip():
                args += ["--spotify-client-id", self.spId.text().strip()]
            if self.spSecret.text().strip():
                args += ["--spotify-client-secret", self.spSecret.text().strip()]
            if self.spRedirect.text().strip():
                # Trim to avoid hidden spaces causing INVALID_CLIENT
                ru = self.spRedirect.text().strip()
                args += ["--spotify-redirect-uri", ru]
                self._append(f"[desktop] Using Spotify redirect: {ru}")
            for key, widget in ("happy", self.spHappy), ("sad", self.spSad), ("angry", self.spAngry), ("neutral", self.spNeutral), ("fear", self.spFear):
                val = widget.text().strip()
                if val:
                    args += [f"--sp-{key}", val]
        else:
            args += ["--music-dir", self.musicDir.text().strip()]

        # Use trained classifier if requested
        if self.useCustomModel.isChecked():
            model_path = self.modelOut.text().strip()
            if not model_path:
                self._warn("Custom model", "Please choose a model file (or disable 'Use this model').")
                return
            if not os.path.isfile(model_path):
                self._warn("Custom model", f"Model file not found:\n{model_path}")
                return
            args += ["--custom-classifier", model_path, "--embedding-model", self.embModel.currentText().strip()]

        # Update model label pre-launch
        try:
            eng = getattr(self, "engineCombo", None).currentText() if hasattr(self, "engineCombo") else "Auto"
            if eng == "DeepFace CNN":
                self.modelLabel.setText("Model: DeepFace CNN (built-in)")
            elif eng == "Custom model" or self.useCustomModel.isChecked():
                self.modelLabel.setText(f"Model: {os.path.basename(self.modelOut.text().strip())} ({self.embModel.currentText().strip()})")
            else:
                # Auto: reflect checkbox
                if self.useCustomModel.isChecked():
                    self.modelLabel.setText(f"Model: {os.path.basename(self.modelOut.text().strip())} ({self.embModel.currentText().strip()})")
                else:
                    self.modelLabel.setText("Model: DeepFace CNN (built-in)")
        except Exception:
            pass
        # Pass control-file path for music commands
        try:
            args += ["--control-file", self._controlPath]
        except Exception:
            pass
        self._append("[desktop] Launching: " + shlex.join(args))
        try:
            self.proc = subprocess.Popen(args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            self.reader = LogReader(self.proc)
            self.reader.line.connect(self._append)
            self.reader.exited.connect(self._on_exited)
            self.reader.start()
            self.statusLabel.setText("Status: running")
            self._set_running_ui(True)
        except Exception as e:
            self._append(f"[desktop] Failed to start: {e}")

    def _on_exited(self, rc: int):
        self.statusLabel.setText(f"Status: stopped (code {rc})")
        self._set_running_ui(False)

    def on_stop(self):
        if not self.proc or self.proc.poll() is not None:
            self._append("[desktop] Not running")
            return
        try:
            if self.reader:
                self.reader.stop()
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except Exception:
                self.proc.kill()
            self._append("[desktop] Stopped.")
        except Exception as e:
            self._append(f"[desktop] Stop failed: {e}")
        finally:
            self.statusLabel.setText("Status: stopped")
            self._set_running_ui(False)

    # --- Live Preview (idle only) ---
    def _start_idle_preview(self):
        # Don't start while running the main process to avoid camera contention
        if self.proc and self.proc.poll() is None:
            return
        try:
            import cv2
            # Close previous
            self._stop_idle_preview()
            idx = int(self.cameraIndex.value())
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                self._append(f"[preview] Failed to open camera index {idx}")
                return
            self._previewCap = cap
            self._previewTimer.start()
        except Exception as e:
            self._append(f"[preview] Start failed: {e}")

    def _stop_idle_preview(self):
        try:
            if self._previewTimer.isActive():
                self._previewTimer.stop()
        except Exception:
            pass
        try:
            if self._previewCap is not None:
                self._previewCap.release()
        except Exception:
            pass
        self._previewCap = None

    def _on_preview_timer(self):
        if self._previewCap is None:
            return
        try:
            import cv2
            ok, frame = self._previewCap.read()
            if not ok or frame is None:
                return
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            try:
                qfmt = QtGui.QImage.Format.Format_RGB888
            except AttributeError:
                qfmt = QtGui.QImage.Format_RGB888
            qimg = QtGui.QImage(rgb.data, w, h, ch*w, qfmt)
            pix = QtGui.QPixmap.fromImage(qimg).scaled(
                self.previewLabel.width(), self.previewLabel.height(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            self.previewLabel.setPixmap(pix)
        except Exception:
            pass

    # --- Music control helpers ---
    def _write_control(self, update: dict):
        try:
            import json
            # merge
            self._controlState.update(update)
            tmp = self._controlPath + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self._controlState, f)
            os.replace(tmp, self._controlPath)
        except Exception as e:
            self._append(f"[desktop] Failed to write control file: {e}")

    def _toggle_pause_play(self):
        self._paused = not self._paused
        if self._paused:
            self._write_control({"pause": True, "play": False})
            self.pausePlayBtn.setText("Play")
        else:
            self._write_control({"play": True, "pause": False})
            self.pausePlayBtn.setText("Pause")

    def _on_volume_changed(self, val: int):
        # Map 0-100 to 0.0-1.0
        v = max(0.0, min(1.0, float(val) / 100.0))
        self._write_control({"volume": v})

    def _send_test(self):
        # Try a quick one-shot TEST over serial without touching the running process
        import serial  # type: ignore
        port = self.serialPort.currentText().strip()
        if not port:
            self._append("[desktop] No serial port selected")
            return
        try:
            with serial.Serial(port=port, baudrate=int(self.baud.text().strip() or 115200), timeout=0.2) as ser:
                # Allow reset
                time.sleep(1.6)
                try:
                    ser.reset_input_buffer()
                except Exception:
                    pass
                ser.write(b"TEST\n"); ser.flush()
                self._append("[desktop] Sent TEST")
        except Exception as e:
            self._append(f"[desktop] Serial TEST failed: {e}")

    # _send_bright removed

    def _spotify_auth(self):
        if SpotifyOAuth is None:
            self._warn("Spotify", "spotipy is not installed in this environment.")
            return
        cid = self.spId.text().strip(); secret = self.spSecret.text().strip(); redirect = self.spRedirect.text().strip()
        if not cid or not secret or not redirect:
            self._warn("Spotify", "Please fill Client ID, Client Secret, and Redirect URI.")
            return
        scopes = "user-modify-playback-state user-read-playback-state"
        try:
            SpotifyOAuth(client_id=cid, client_secret=secret, redirect_uri=redirect, scope=scopes,
                         open_browser=True, cache_path=CACHE_FILE)
            self._append("[desktop] Opened Spotify auth in browser. Complete the flow and return.")
        except Exception as e:
            self._append(f"[desktop] Spotify auth init failed: {e}")

    def closeEvent(self, ev: QtGui.QCloseEvent):
        try:
            self._save_settings()
        except Exception:
            pass
        try:
            self.on_stop()
        except Exception:
            pass
        try:
            self._stop_training()
        except Exception:
            pass
        try:
            self._stop_idle_preview()
        except Exception:
            pass
        ev.accept()

    def _stop_training(self):
        if self._trainProc and self._trainProc.poll() is None:
            try:
                self._append("[training] Stopping training…")
                self._trainProc.terminate()
                try:
                    self._trainProc.wait(timeout=2)
                except Exception:
                    self._trainProc.kill()
            except Exception:
                pass
        try:
            if self._trainReader and self._trainReader.isRunning():
                self._trainReader.stop()
                self._trainReader.wait(500)
        except Exception:
            pass
        self._set_training_ui(False)

    # Settings persistence
    def _load_settings(self):
        s = QtCore.QSettings("Moodify", "Desktop")
        def get(key, default=""):
            return s.value(key, default)
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
        self.spHappy.setText(str(get("spotify/uri_happy", "")))
        self.spSad.setText(str(get("spotify/uri_sad", "")))
        self.spAngry.setText(str(get("spotify/uri_angry", "")))
        self.spNeutral.setText(str(get("spotify/uri_neutral", "")))
        self.spFear.setText(str(get("spotify/uri_fear", "")))
        # training
        self.dsRoot.setText(str(get("train/dataset", self.dsRoot.text())))
        self.modelOut.setText(str(get("train/model_out", self.modelOut.text())))
        self.embModel.setCurrentText(str(get("train/emb_model", self.embModel.currentText())))
        self.algCombo.setCurrentText(str(get("train/algo", self.algCombo.currentText())))
        self.useCustomModel.setChecked(bool(get("train/use_model", False)))
        # detection
        try:
            self.engineCombo.setCurrentText(str(get("detect/engine", self.engineCombo.currentText())))
            self.detectorCombo.setCurrentText(str(get("detect/detector", self.detectorCombo.currentText())))
            self.trainDetectorCombo.setCurrentText(str(get("train/detector", self.trainDetectorCombo.currentText())))
        except Exception:
            pass
    # (Brightness setting removed)

    def _save_settings(self):
        s = QtCore.QSettings("Moodify", "Desktop")
        s.setValue("serial/port", self.serialPort.currentText())
        s.setValue("serial/baud", self.baud.text())
        s.setValue("serial/log", self.serialLog.isChecked())
        s.setValue("vision/display", self.display.isChecked())
        s.setValue("vision/camera", self.cameraIndex.value())
        s.setValue("vision/min_interval", self.minInterval.value())
        s.setValue("vision/pause_no_face", self.pauseNoFace.isChecked())
        s.setValue("music/mode", self.musicMode.currentText())
        s.setValue("music/dir", self.musicDir.text())
        s.setValue("spotify/device", self.spDevice.text())
        s.setValue("spotify/id", self.spId.text())
        s.setValue("spotify/redirect", self.spRedirect.text())
        s.setValue("spotify/uri_happy", self.spHappy.text())
        s.setValue("spotify/uri_sad", self.spSad.text())
        s.setValue("spotify/uri_angry", self.spAngry.text())
        s.setValue("spotify/uri_neutral", self.spNeutral.text())
        s.setValue("spotify/uri_fear", self.spFear.text())
        # training
        s.setValue("train/dataset", self.dsRoot.text())
        s.setValue("train/model_out", self.modelOut.text())
        s.setValue("train/emb_model", self.embModel.currentText())
        s.setValue("train/algo", self.algCombo.currentText())
        s.setValue("train/use_model", self.useCustomModel.isChecked())
        # detection
        try:
            s.setValue("detect/engine", self.engineCombo.currentText())
            s.setValue("detect/detector", self.detectorCombo.currentText())
            s.setValue("train/detector", self.trainDetectorCombo.currentText())
        except Exception:
            pass
    # (Brightness setting removed)


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
