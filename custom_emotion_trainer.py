#!/usr/bin/env python3
"""
Custom Emotion Trainer

Train a lightweight classifier on top of DeepFace embeddings using a
user-provided image dataset organized as:

dataset/
  happy/
    img001.jpg
    img002.png
  sad/
    ...
  angry/
    ...
  neutral/
    ...
  fear/
    ...

This script computes face embeddings for each image (using a fixed backbone
like Facenet512), trains a scikit-learn classifier (LogisticRegression by
default), evaluates it, and saves the trained pipeline to disk.

The saved model can be used by mood_detector.py via --custom-classifier.
"""
from __future__ import annotations

import os
import cv2
import glob
import json
import argparse
from typing import List, Tuple

import numpy as np

try:
    from deepface import DeepFace  # type: ignore
except Exception as e:
    raise SystemExit("DeepFace is required. pip install deepface") from e

try:
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.svm import SVC  # type: ignore
    from sklearn.model_selection import train_test_split  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore
    from sklearn.pipeline import Pipeline  # type: ignore
    from sklearn.metrics import classification_report, fbeta_score  # type: ignore
    import joblib  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("scikit-learn is required. pip install scikit-learn") from e


CANON_EMOTIONS = {"happy", "sad", "angry", "neutral", "fear"}


def _list_images(root: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for label in sorted(os.listdir(root)):
        d = os.path.join(root, label)
        if not os.path.isdir(d):
            continue
        # Only allow canonical labels to ensure runtime compatibility
        if label not in CANON_EMOTIONS:
            print(f"[warn] Skipping non-canonical label '{label}'. Allowed: {sorted(CANON_EMOTIONS)}")
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            for p in glob.glob(os.path.join(d, ext)):
                pairs.append((p, label))
    return pairs


def _embed_bgr(img_bgr, model_name: str = "Facenet512", detector_backend: str = "opencv") -> np.ndarray | None:
    try:
        reps = DeepFace.represent(
            img_path=img_bgr,
            model_name=model_name,
            enforce_detection=False,
            detector_backend=detector_backend,
            align=True,
        )
        if isinstance(reps, list) and reps:
            emb = reps[0].get("embedding")
        elif isinstance(reps, dict):
            emb = reps.get("embedding")
        else:
            return None
        if emb is None:
            return None
        return np.asarray(emb, dtype=np.float32)
    except Exception:
        return None


def load_dataset(root: str, model_name: str, detector_backend: str) -> tuple[np.ndarray, np.ndarray, List[str]]:
    pairs = _list_images(root)
    if not pairs:
        raise SystemExit(f"No images found under '{root}'. Organize as <root>/<emotion>/*.jpg")
    X: List[np.ndarray] = []
    y: List[int] = []
    labels = sorted(sorted({lbl for _, lbl in pairs}))
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}

    for path, lbl in pairs:
        img = cv2.imread(path)
        if img is None:
            print(f"[warn] Failed to read image: {path}")
            continue
        emb = _embed_bgr(img, model_name=model_name, detector_backend=detector_backend)
        if emb is None:
            print(f"[warn] No embedding for: {path}")
            continue
        X.append(emb)
        y.append(label_to_idx[lbl])

    if not X:
        raise SystemExit("No embeddings computed. Check images and model.")
    return np.vstack(X), np.asarray(y, dtype=np.int64), labels


def train_classifier(X: np.ndarray, y: np.ndarray, algo: str = "logreg") -> Pipeline:
    if algo == "svm":
        clf = SVC(probability=True, kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")
    else:
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])
    pipe.fit(X, y)
    return pipe


def main():
    ap = argparse.ArgumentParser(description="Train a custom emotion classifier on user images.")
    ap.add_argument("--data-dir", required=True, help="Dataset root with per-emotion folders")
    ap.add_argument("--model-out", default="custom_emotions.pkl", help="Output path for trained model")
    ap.add_argument("--embedding-model", default="Facenet512", help="DeepFace embedding backbone (e.g., Facenet512, VGG-Face)")
    ap.add_argument("--algo", choices=["logreg", "svm"], default="logreg")
    ap.add_argument("--detector-backend", choices=["opencv", "retinaface", "mediapipe", "mtcnn", "ssd", "dlib"], default="opencv")
    ap.add_argument("--test-split", type=float, default=0.2)
    args = ap.parse_args()

    print(f"Loading dataset from {args.data_dir}… (detector: {args.detector_backend})")
    X, y, labels = load_dataset(args.data_dir, model_name=args.embedding_model, detector_backend=args.detector_backend)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_split, stratify=y, random_state=42)

    print(f"Training {args.algo} classifier on {X_train.shape[0]} samples…")
    pipe = train_classifier(X_train, y_train, algo=args.algo)

    # Evaluate (skip gracefully if too few samples)
    try:
        print("Evaluating…")
        if X_test.shape[0] > 0:
            y_pred = pipe.predict(X_test)
            print(classification_report(y_test, y_pred, target_names=labels, digits=3))
            try:
                # F2 prioritizes recall (missed detections) 4x over precision
                f2_macro = fbeta_score(y_test, y_pred, beta=2, average="macro", zero_division=0)
                f2_per_class = fbeta_score(y_test, y_pred, beta=2, average=None, zero_division=0)
                print(f"Macro F2: {f2_macro:.3f}")
                # Show per-class F2 in label order
                for lbl, val in zip(labels, f2_per_class):
                    print(f"  F2[{lbl}]: {val:.3f}")
            except Exception as e:
                print(f"[warn] Failed to compute F2 scores: {e}")
        else:
            print("[warn] Test split has 0 samples; skipping evaluation.")
    except Exception as e:
        print(f"[warn] Evaluation failed: {e}")

    payload = {
        "pipeline": pipe,
        "labels": labels,
        "embedding_model": args.embedding_model,
    }
    # Normalize and ensure parent directory exists
    out_path = os.path.abspath(os.path.expanduser(args.model_out))
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    print(f"Saving model to {out_path} …")
    try:
        joblib.dump(payload, out_path)
    except Exception as e:
        raise SystemExit(f"Failed to save model: {e}")

    meta = {
        "labels": labels,
        "embedding_model": args.embedding_model,
        "algo": args.algo,
        "detector_backend": args.detector_backend,
        "samples": int(X.shape[0]),
    }
    try:
        with open(out_path + ".json", "w") as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        print(f"[warn] Failed to write metadata JSON: {e}")
    print(f"Saved model to {out_path} and metadata to {out_path}.json")


if __name__ == "__main__":
    main()
