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
    disgust/
        ...
    surprise/
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
from typing import List, Tuple, Dict, Any

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


CANON_EMOTIONS = {"happy", "sad", "angry", "neutral", "fear", "disgust", "surprise"}


def _resolve_detector_backend(detector_backend: str) -> str:
    """Return a compatible detector backend, falling back when mediapipe conflicts.

    mediapipe 0.10.x is incompatible with protobuf >=4/5 (required by TF 2.20+).
    When detected, silently switch to 'opencv' and log a warning.
    """
    backend = detector_backend
    if backend == "mediapipe":
        try:
            from google.protobuf import __version__ as pb_ver  # type: ignore
            major = int(str(pb_ver).split(".")[0])
            if major >= 4:
                print("[warn] mediapipe detector incompatible with protobuf>=4 (TF 2.20 uses protobuf>=5). Falling back to opencv.")
                backend = "opencv"
            else:
                __import__("mediapipe")
        except Exception as e:
            print(f"[warn] mediapipe detector unavailable ({e}). Falling back to opencv.")
            backend = "opencv"
    return backend


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
        arr = np.asarray(emb, dtype=np.float32).reshape(-1)
        return arr
    except Exception:
        return None


def _augment_image(img: np.ndarray) -> np.ndarray:
    """Apply a random small augmentation: rotate, flip, brightness/contrast jitter, or zoom.

    Returns a new BGR image of the same size.
    """
    h, w = img.shape[:2]
    choice = int(np.random.randint(0, 4))
    out = img.copy()
    if choice == 0:
        # small rotation
        angle = float(np.random.uniform(-10, 10))
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        out = cv2.warpAffine(out, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    elif choice == 1:
        # horizontal flip
        out = cv2.flip(out, 1)
    elif choice == 2:
        # brightness/contrast jitter: new = alpha*img + beta
        alpha = float(np.random.uniform(0.9, 1.1))
        beta = float(np.random.uniform(-15, 15))
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
    else:
        # zoom 0.9-1.1 with crop/pad
        scale = float(np.random.uniform(0.9, 1.1))
        nz = cv2.resize(out, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        nh, nw = nz.shape[:2]
        if scale >= 1.0:
            # center-crop back to original size
            y0 = (nh - h) // 2
            x0 = (nw - w) // 2
            out = nz[y0:y0 + h, x0:x0 + w]
        else:
            # place on canvas
            canvas = np.zeros_like(out)
            y0 = (h - nh) // 2
            x0 = (w - nw) // 2
            canvas[y0:y0 + nh, x0:x0 + nw] = nz
            out = canvas
    return out


def load_dataset(root: str, model_name: str, detector_backend: str, max_per_class: int = 0, augment: int = 0) -> tuple[np.ndarray, np.ndarray, List[str]]:
    detector_backend = _resolve_detector_backend(detector_backend)
    pairs = _list_images(root)
    if not pairs:
        raise SystemExit(f"No images found under '{root}'. Organize as <root>/<emotion>/*.jpg")
    X: List[np.ndarray] = []
    y: List[int] = []
    labels = sorted(sorted({lbl for _, lbl in pairs}))
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    per_class_counts: Dict[str, int] = {lbl: 0 for lbl in labels}

    for path, lbl in pairs:
        if max_per_class and per_class_counts.get(lbl, 0) >= max_per_class:
            continue
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
        per_class_counts[lbl] = per_class_counts.get(lbl, 0) + 1
        # Generate augmented variants
        for _ in range(max(0, int(augment))):
            try:
                aug = _augment_image(img)
                aemb = _embed_bgr(aug, model_name=model_name, detector_backend=detector_backend)
                if aemb is not None:
                    X.append(aemb)
                    y.append(label_to_idx[lbl])
            except Exception:
                pass

    if not X:
        raise SystemExit("No embeddings computed. Check images and model.")
    X_arr = np.vstack(X).astype(np.float32)
    y_arr = np.asarray(y, dtype=np.int64)
    return X_arr, y_arr, labels


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


def _auto_search(
    data_dir: str,
    embeddings: List[str],
    detectors: List[str],
    algos: List[str],
    test_split: float,
    max_per_class: int,
) -> Dict[str, Any]:
    """Run a simple grid search across embeddings, detectors, and algos.

    Returns a dict with keys: best, results. 'best' includes the trained Pipeline.
    """
    results: List[Dict[str, Any]] = []
    best: Dict[str, Any] | None = None
    for emb in embeddings:
        for det in detectors:
            det_eff = _resolve_detector_backend(det)
            try:
                print(f"[search] Embedding={emb} | Detector={det} -> using {det_eff}")
                X, y, labels = load_dataset(data_dir, model_name=emb, detector_backend=det_eff, max_per_class=max_per_class)
                embed_dim = int(X.shape[1])
            except SystemExit as e:
                print(f"[search] Skipping {emb}/{det_eff}: {e}")
                continue
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, stratify=y, random_state=42)
            for algo in algos:
                try:
                    pipe = train_classifier(X_train, y_train, algo=algo)
                    y_pred = pipe.predict(X_test)
                    f2_macro = fbeta_score(y_test, y_pred, beta=2, average="macro", zero_division=0)
                    rec: Dict[str, Any] = {
                        "embedding_model": emb,
                        "detector_backend": det_eff,
                        "algo": algo,
                        "embedding_dim": embed_dim,
                        "f2_macro": float(f2_macro),
                        "n_train": int(X_train.shape[0]),
                        "n_test": int(X_test.shape[0]),
                        "labels": labels,
                    }
                    results.append(rec)
                    print(f"[search] {emb}/{det_eff}/{algo} -> Macro F2={f2_macro:.3f}")
                    if best is None or f2_macro > best["f2_macro"]:
                        best = {
                            **rec,
                            "pipeline": pipe,
                        }
                except Exception as e:
                    print(f"[search] {emb}/{det_eff}/{algo} failed: {e}")
    if best is None:
        raise SystemExit("Auto search failed: no valid configuration produced a model.")
    return {"best": best, "results": results}


def main():
    ap = argparse.ArgumentParser(description="Train a custom emotion classifier on user images.")
    ap.add_argument("--data-dir", required=True, help="Dataset root with per-emotion folders")
    ap.add_argument("--model-out", default="custom_emotions.pkl", help="Output path for trained model")
    ap.add_argument("--embedding-model", default="Facenet512", help="DeepFace embedding backbone (e.g., Facenet512, VGG-Face)")
    ap.add_argument("--algo", choices=["logreg", "svm"], default="logreg")
    ap.add_argument("--detector-backend", choices=["opencv", "retinaface", "mediapipe", "mtcnn", "ssd", "dlib"], default="opencv")
    ap.add_argument("--test-split", type=float, default=0.2)
    ap.add_argument("--auto-search", action="store_true", help="Try multiple embeddings/detectors/algos and pick the best")
    ap.add_argument("--search-embeddings", default="Facenet512,VGG-Face,ArcFace", help="Comma-separated embedding models to try when --auto-search")
    ap.add_argument("--search-detectors", default="opencv,retinaface,mtcnn,dlib,ssd,mediapipe", help="Comma-separated detectors to try when --auto-search")
    ap.add_argument("--search-algos", default="logreg,svm", help="Comma-separated algos to try when --auto-search")
    ap.add_argument("--max-per-class", type=int, default=0, help="Cap images per class during search/training (0=all)")
    ap.add_argument("--augment", type=int, default=0, help="Number of augmented variants to generate per image during training")
    args = ap.parse_args()
    if args.auto_search:
        emb_list = [s.strip() for s in args.search_embeddings.split(",") if s.strip()]
        det_list = [s.strip() for s in args.search_detectors.split(",") if s.strip()]
        algo_list = [s.strip() for s in args.search_algos.split(",") if s.strip()]
        print("Starting auto search…")
        res = _auto_search(
            data_dir=args.data_dir,
            embeddings=emb_list,
            detectors=det_list,
            algos=algo_list,
            test_split=args.test_split,
            max_per_class=max(0, int(args.max_per_class)),
        )
        best = res["best"]
        labels = best["labels"]
        pipe = best["pipeline"]
        out_path = os.path.abspath(os.path.expanduser(args.model_out))
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        print(f"Best config: {best['embedding_model']}/{best['detector_backend']}/{best['algo']} -> Macro F2={best['f2_macro']:.3f}")
        print(f"Saving best model to {out_path} …")
        try:
            joblib.dump({
                "pipeline": pipe,
                "labels": labels,
                "embedding_model": best["embedding_model"],
                "embedding_dim": int(best.get("embedding_dim", 0)),
            }, out_path)
        except Exception as e:
            raise SystemExit(f"Failed to save model: {e}")
        meta = {
            "labels": labels,
            "embedding_model": best["embedding_model"],
            "algo": best["algo"],
            "detector_backend": best["detector_backend"],
            "samples": int(sum(1 for _ in _list_images(args.data_dir))),
            "embedding_dim": int(best.get("embedding_dim", 0)),
            "score": {"f2_macro": float(best["f2_macro"])},
        }
        try:
            with open(out_path + ".json", "w") as f:
                json.dump(meta, f, indent=2)
            with open(out_path + ".search.json", "w") as f:
                json.dump({"results": res["results"]}, f, indent=2)
        except Exception as e:
            print(f"[warn] Failed to write metadata JSON: {e}")
        print(f"Saved model to {out_path}, metadata to {out_path}.json, and search results to {out_path}.search.json")
        return
    else:
        print(f"Loading dataset from {args.data_dir}… (detector: {args.detector_backend}), augment={args.augment}")
        X, y, labels = load_dataset(
            args.data_dir,
            model_name=args.embedding_model,
            detector_backend=args.detector_backend,
            max_per_class=max(0, int(args.max_per_class)),
            augment=max(0, int(args.augment)),
        )
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
                    f2_macro = fbeta_score(y_test, y_pred, beta=2, average="macro", zero_division=0)
                    f2_per_class = fbeta_score(y_test, y_pred, beta=2, average=None, zero_division=0)
                    print(f"Macro F2: {f2_macro:.3f}")
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
            "embedding_dim": int(X.shape[1]),
        }
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
            "embedding_dim": int(X.shape[1]),
        }
        try:
            with open(out_path + ".json", "w") as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            print(f"[warn] Failed to write metadata JSON: {e}")
        print(f"Saved model to {out_path} and metadata to {out_path}.json")


if __name__ == "__main__":
    main()
