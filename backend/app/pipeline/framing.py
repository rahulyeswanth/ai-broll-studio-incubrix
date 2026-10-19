"""Subject-aware crop positioning. When the source aspect differs from the target (e.g. landscape -> 9:16)
we estimate where the visual interest is (faces if detected, otherwise edge/contrast energy) rather than always
cropping the exact centre. Falls back to the centre when OpenCV/NumPy are unavailable."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

from .media import video_info

log = logging.getLogger(__name__)


def estimate_focus(video_path: Path, samples: int = 5) -> Tuple[float, float, str]:
    """Return (cx, cy, method) with cx, cy in 0..1 (fractions of source width/height)."""
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError:
        return 0.5, 0.5, "center_default(opencv_unavailable)"
    cap = cv2.VideoCapture(str(video_path))
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if n <= 0:
            return 0.5, 0.5, "center_default(no_frames)"
        face_x, face_y, face_w = [], [], []
        energy_x, energy_y = None, None
        cascade = None
        try:
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            if cascade.empty():
                cascade = None
        except Exception:  # noqa: BLE001
            cascade = None
        used = 0
        for k in range(samples):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * (k + 0.5) / samples))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            used += 1
            h, w = frame.shape[:2]
            scale = 320.0 / max(w, 1)
            small = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if cascade is not None:
                faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(24, 24))
                for (x, y, fw, fh) in faces:
                    face_x.append((x + fw / 2) / gray.shape[1])
                    face_y.append((y + fh / 2) / gray.shape[0])
                    face_w.append(fw * fh)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
            mag = np.sqrt(gx * gx + gy * gy) + 0.25 * cv2.Laplacian(gray, cv2.CV_32F).__abs__()
            ex = mag.sum(axis=0)
            ey = mag.sum(axis=1)
            energy_x = ex if energy_x is None else energy_x + ex
            energy_y = ey if energy_y is None else energy_y + ey
        if used == 0:
            return 0.5, 0.5, "center_default(decode_failed)"
        if face_x:
            wts = np.array(face_w, dtype=float)
            return float(np.average(face_x, weights=wts)), float(np.average(face_y, weights=wts)), "face_detection"
        if energy_x is not None and energy_x.sum() > 0:
            def centroid(v):
                idx = np.arange(len(v), dtype=float)
                # emphasise the strongest region: use the energy^2 centroid to avoid washing out to centre
                p = v ** 2
                return float((idx * p).sum() / p.sum() / max(len(v) - 1, 1))
            return centroid(energy_x), centroid(energy_y), "edge_energy_centroid"
        return 0.5, 0.5, "center_default(flat_frames)"
    finally:
        cap.release()


def crop_offsets(src_w: int, src_h: int, dst_w: int, dst_h: int, cx: float, cy: float) -> Tuple[int, int, int, int]:
    """Scale-to-cover then crop. Returns (scaled_w, scaled_h, x_off, y_off) so the focus point stays in frame."""
    scale = max(dst_w / src_w, dst_h / src_h)
    sw, sh = int(round(src_w * scale)), int(round(src_h * scale))
    sw, sh = max(sw, dst_w), max(sh, dst_h)
    x = int(round(cx * sw - dst_w / 2))
    y = int(round(cy * sh - dst_h / 2))
    x = max(0, min(sw - dst_w, x))
    y = max(0, min(sh - dst_h, y))
    return sw, sh, x - (x % 2), y - (y % 2)
