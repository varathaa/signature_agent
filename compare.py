import cv2
import numpy as np

import config

_W, _H = 256, 128


def preprocess(img: np.ndarray) -> np.ndarray | None:
    """Return a normalised binary ink mask (ink=255) or None if no ink is found."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    ys, xs = np.where(ink > 0)
    if len(xs) < 20:
        return None
    ink = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = ink.shape
    scale = min((_W - 16) / w, (_H - 16) / h)
    ink = cv2.resize(ink, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((_H, _W), np.uint8)
    oy, ox = (_H - ink.shape[0]) // 2, (_W - ink.shape[1]) // 2
    canvas[oy:oy + ink.shape[0], ox:ox + ink.shape[1]] = ink
    return (canvas > 60).astype(np.uint8) * 255


def _chamfer_score(a: np.ndarray, b: np.ndarray) -> float:
    # Symmetric mean distance from each ink pixel to the nearest ink pixel of the other image.
    dt_a = cv2.distanceTransform(255 - a, cv2.DIST_L2, 3)
    dt_b = cv2.distanceTransform(255 - b, cv2.DIST_L2, 3)
    d = (dt_b[a > 0].mean() + dt_a[b > 0].mean()) / 2
    return float(np.exp(-d / 6.0))


def _overlap_score(a: np.ndarray, b: np.ndarray) -> float:
    # Dice overlap after thickening strokes so small shifts are tolerated.
    k = np.ones((5, 5), np.uint8)
    da, db = cv2.dilate(a, k) > 0, cv2.dilate(b, k) > 0
    inter = np.logical_and(da, db).sum()
    return float(2 * inter / (da.sum() + db.sum() + 1e-9))


def similarity(master: np.ndarray, letter: np.ndarray) -> float | None:
    a, b = preprocess(master), preprocess(letter)
    if a is None or b is None:
        return None
    return round(0.6 * _chamfer_score(a, b) + 0.4 * _overlap_score(a, b), 4)


def verdict(score: float | None) -> str:
    if score is None:
        return "REVIEW"
    if score >= config.MATCH_THRESHOLD:
        return "MATCH"
    if score >= config.MATCH_THRESHOLD - config.REVIEW_BAND:
        return "REVIEW"
    return "MISMATCH"
