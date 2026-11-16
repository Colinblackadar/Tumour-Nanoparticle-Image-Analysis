#!/usr/bin/env python3
"""Stabilize a multichannel intravital time-lapse using manually placed Fiji ROIs.

The method uses the vessel channel to estimate translation, then applies the
same framewise translation to both the vessel and nanoparticle channels.

The registration method reproduces the supplied analysis script:
  1. CLAHE-enhanced bidirectional Lucas-Kanade tracking (31-pixel window,
     3 pyramid levels, 3x3 grid), constrained by manual point ROI anchors.
  2. Local vessel-channel template refinement (25x25-pixel patch, +/-8-pixel
     search radius, and smoothness lambda = 0.08).

The output is a multichannel TIFF (time, channel, y, x). Intensities are
resampled by first-order interpolation, with zero padding at image borders.
No quantification or statistical analysis is performed.

Example:
    python intravital_roi_stabilization.py \
        --video /path/to/recording.tif \
        --rois /path/to/anchors.zip

See README_intravital_stabilization.md for instructions.
"""

from __future__ import annotations

import argparse
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import shift as ndi_shift
from skimage import filters

try:
    from roifile import roiread
except ImportError as exc:
    raise SystemExit("Missing dependency: roifile. Install with: pip install -r requirements_intravital_stabilization.txt") from exc

# Fixed analysis parameters (unchanged from the original processing script).
VESSEL_CHANNEL = 0  # First channel (index 0) is used for registration.
LK_WIN_SIZE = 31
LK_MAX_LEVEL = 3
LK_MAX_ITER = 40
LK_EPS = 0.01
GRID_RADIUS_PX = 3
GRID_SIDE = 3
POINT_FB_ERROR_MAX_PX = 1.5
MIN_GOOD_GRID_POINTS = 4
MAX_ONE_FRAME_DISPLACEMENT_PX = 8.0
SNAP_TEMPLATE_HALF_WIDTH = 12
SNAP_SEARCH_RADIUS = 8
SNAP_SMOOTH_LAMBDA = 0.08
ABS_BAD_MATCH_SCORE = 0.15
BAD_SCORE_MAD_MULTIPLIER = 5.0
ABS_QUANT_BAD_MATCH_SCORE = 0.08
CATASTROPHIC_SHARPNESS_RATIO = 0.28
SHARPNESS_LOCAL_RADIUS = 5
LARGE_RESIDUAL_STEP_PX = 2.5
LARGE_STEP_EXTRA_PENALTY = 1.5
TRACKING_SUSPECT_FB_ERROR = 0.9
INTERPOLATION_ORDER = 1
TIFF_COMPRESSION = "zlib"
METHOD_NAME = "roi_anchored_lk_and_template_refinement"

# Generic Fiji ROI naming for frame association (1-based frame numbers).
FRAME_RE = re.compile(r"^[Ff]0*(\d+)$")
FLEX_FRAME_RE = re.compile(r"(?:frame|frm|f|t)[_\-\s]*0*(\d+)", re.IGNORECASE)


# ========================================================================
# Input TIFF and Fiji ROI anchors
# ========================================================================

@dataclass
class Anchor:
    frame: int
    x: float
    y: float
    name: str


def normalize_to_tcyx(arr: np.ndarray, axes: str) -> np.ndarray:
    axes = axes.upper()
    arr = np.asarray(arr)
    if len(axes) != arr.ndim:
        raise ValueError(f"Axes '{axes}' do not match ndim={arr.ndim}.")
    if 'Z' in axes:
        zi = axes.index('Z')
        if arr.shape[zi] != 1:
            raise ValueError(f'Input contains Z={arr.shape[zi]} planes. Make an explicit 2D time-lapse/projection first.')
        arr = np.take(arr, 0, axis=zi)
        axes = axes[:zi] + axes[zi + 1:]
    if 'S' in axes and 'C' not in axes:
        axes = axes.replace('S', 'C')
    removable = [i for i, ax in enumerate(axes) if ax not in 'TCYX' and arr.shape[i] == 1]
    for i in reversed(removable):
        arr = np.take(arr, 0, axis=i)
        axes = axes[:i] + axes[i + 1:]
    if any((ax not in 'TCYX' for ax in axes)):
        raise ValueError(f'Unsupported TIFF axes: {axes}')
    if 'T' not in axes:
        arr = np.expand_dims(arr, 0)
        axes = 'T' + axes
    if 'C' not in axes:
        arr = np.expand_dims(arr, 1)
        axes = axes[0] + 'C' + axes[1:]
    order = [axes.index(a) for a in 'TCYX']
    arr = np.transpose(arr, order)
    if arr.shape[1] != 2:
        raise ValueError(f'Expected exactly 2 channels after normalization; got shape {arr.shape}.')
    return arr


def load_tiff(path: Path) -> Tuple[np.ndarray, str]:
    with tifffile.TiffFile(path) as tf:
        series = tf.series[0]
        axes = series.axes
        arr = series.asarray()
    return (normalize_to_tcyx(arr, axes), axes)


def infer_frame_name_first(roi, name: str) -> Optional[int]:
    m = FRAME_RE.match(name.strip())
    if m:
        return int(m.group(1)) - 1
    m = FLEX_FRAME_RE.search(name)
    if m:
        return int(m.group(1)) - 1
    tpos = int(getattr(roi, 't_position', 0) or 0)
    if tpos > 0:
        return tpos - 1
    pos = int(getattr(roi, 'position', 0) or 0)
    if pos > 0:
        return pos - 1
    return None


def load_anchors(path: Path, n_frames: int) -> List[Anchor]:
    rois = roiread(str(path))
    if not isinstance(rois, (list, tuple)):
        rois = [rois]
    raw: List[Anchor] = []
    for i, roi in enumerate(rois):
        name = str(getattr(roi, 'name', '') or f'roi_{i + 1}')
        fr = infer_frame_name_first(roi, name)
        if fr is None:
            warnings.warn(f"Skipping ROI '{name}': no frame.")
            continue
        if not 0 <= fr < n_frames:
            warnings.warn(f"Skipping ROI '{name}': frame {fr + 1} outside movie.")
            continue
        coords = np.asarray(roi.coordinates(), dtype=float)
        if coords.ndim != 2 or len(coords) == 0:
            warnings.warn(f"Skipping ROI '{name}': invalid point.")
            continue
        raw.append(Anchor(frame=fr, x=float(np.mean(coords[:, 0])), y=float(np.mean(coords[:, 1])), name=name))
    if len(raw) < 4:
        raise ValueError(f'{path.name}: need at least 4 valid point anchors.')
    by_frame: Dict[int, List[Anchor]] = {}
    for a in raw:
        by_frame.setdefault(a.frame, []).append(a)
    anchors: List[Anchor] = []
    for fr in sorted(by_frame):
        items = by_frame[fr]
        anchors.append(Anchor(frame=fr, x=float(np.mean([a.x for a in items])), y=float(np.mean([a.y for a in items])), name=' + '.join((a.name for a in items))))
    return anchors


def choose_target_anchor(anchors: Sequence[Anchor]) -> Anchor:
    return anchors[len(anchors) // 2]



# ========================================================================
# Image preprocessing
# ========================================================================

def robust_norm_float(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img, dtype=np.float32)
    vals = x[np.isfinite(x)]
    if len(vals) == 0:
        return np.zeros_like(x)
    lo, hi = np.percentile(vals, [1, 99])
    if hi <= lo:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0, 1).astype(np.float32)


def robust_u8(img: np.ndarray) -> np.ndarray:
    return np.rint(255 * robust_norm_float(img)).astype(np.uint8)


def preprocess_for_lk(img: np.ndarray) -> np.ndarray:
    u = robust_u8(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(u)


def preprocess_match(img: np.ndarray) -> np.ndarray:
    return robust_norm_float(img)


def extract_patch(img: np.ndarray, cx: float, cy: float, half_width: int) -> np.ndarray:
    size = 2 * half_width + 1
    h, w = img.shape
    x0 = int(round(cx)) - half_width
    y0 = int(round(cy)) - half_width
    x1 = x0 + size
    y1 = y0 + size
    left = max(0, -x0)
    top = max(0, -y0)
    right = max(0, x1 - w)
    bottom = max(0, y1 - h)
    padded = np.pad(img, ((top, bottom), (left, right)), mode='reflect')
    return padded[y0 + top:y0 + top + size, x0 + left:x0 + left + size].astype(np.float32)


def local_sharpness(frame: np.ndarray, x: float, y: float) -> float:
    p = extract_patch(frame, x, y, 12)
    return float(np.mean(filters.sobel(robust_norm_float(p))))


def local_median_ratio(values: np.ndarray, radius: int) -> np.ndarray:
    x = np.asarray(values, float)
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        idx = [j for j in range(max(0, i - radius), min(len(x), i + radius + 1)) if j != i and np.isfinite(x[j])]
        if len(idx) < 2 or not np.isfinite(x[i]):
            continue
        med = np.median(x[idx])
        if med > 0:
            out[i] = x[i] / med
    return out



# ========================================================================
# Stage 1: bidirectional grid optical flow
# ========================================================================

def grid_offsets() -> np.ndarray:
    vals = np.linspace(-GRID_RADIUS_PX, GRID_RADIUS_PX, GRID_SIDE, dtype=np.float32)
    return np.array([(dx, dy) for dy in vals for dx in vals], dtype=np.float32)


GRID_OFFSETS = grid_offsets()


def build_grid(x: float, y: float) -> np.ndarray:
    center = np.array([x, y], dtype=np.float32)
    return (GRID_OFFSETS + center[None, :]).reshape(-1, 1, 2)


def lk_params() -> dict:
    return dict(winSize=(LK_WIN_SIZE, LK_WIN_SIZE), maxLevel=LK_MAX_LEVEL, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, LK_MAX_ITER, LK_EPS), flags=0, minEigThreshold=0.0001)


def track_grid_one_step(prev_img: np.ndarray, next_img: np.ndarray, x: float, y: float) -> Dict[str, float]:
    p0 = build_grid(x, y)
    params = lk_params()
    try:
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev_img, next_img, p0, None, **params)
        if p1 is None or st1 is None:
            raise RuntimeError
        p0b, st2, _ = cv2.calcOpticalFlowPyrLK(next_img, prev_img, p1, None, **params)
        if p0b is None or st2 is None:
            raise RuntimeError
        a = p0.reshape(-1, 2)
        b = p1.reshape(-1, 2)
        back = p0b.reshape(-1, 2)
        status = st1.reshape(-1).astype(bool) & st2.reshape(-1).astype(bool)
        fb = np.sqrt(np.sum((back - a) ** 2, axis=1))
        finite = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1) & np.isfinite(back).all(axis=1) & np.isfinite(fb)
        good = status & finite & (fb <= POINT_FB_ERROR_MAX_PX)
        n_good = int(np.sum(good))
        if n_good < MIN_GOOD_GRID_POINTS:
            return {'accepted': False, 'dx': 0.0, 'dy': 0.0, 'n_good': n_good, 'median_fb_error': float(np.nanmedian(fb[finite])) if np.any(finite) else np.nan}
        d = b[good] - a[good]
        dx = float(np.median(d[:, 0]))
        dy = float(np.median(d[:, 1]))
        if math.hypot(dx, dy) > MAX_ONE_FRAME_DISPLACEMENT_PX:
            return {'accepted': False, 'dx': 0.0, 'dy': 0.0, 'n_good': n_good, 'median_fb_error': float(np.median(fb[good]))}
        return {'accepted': True, 'dx': dx, 'dy': dy, 'n_good': n_good, 'median_fb_error': float(np.median(fb[good]))}
    except Exception:
        return {'accepted': False, 'dx': 0.0, 'dy': 0.0, 'n_good': 0, 'median_fb_error': np.nan}


def linear_anchor_position(a: Anchor, b: Anchor, frame: int) -> np.ndarray:
    alpha = (frame - a.frame) / max(1, b.frame - a.frame)
    return np.array([(1 - alpha) * a.x + alpha * b.x, (1 - alpha) * a.y + alpha * b.y])


def coarse_forward(prep: Sequence[np.ndarray], a: Anchor, b: Anchor) -> Tuple[np.ndarray, pd.DataFrame]:
    n = b.frame - a.frame + 1
    centers = np.zeros((n, 2), float)
    centers[0] = [a.x, a.y]
    rows = []
    for j in range(n - 1):
        t0 = a.frame + j
        t1 = t0 + 1
        cur = centers[j]
        step = track_grid_one_step(prep[t0], prep[t1], cur[0], cur[1])
        if step['accepted']:
            nxt = cur + [step['dx'], step['dy']]
            fallback = False
        else:
            nxt = cur + (linear_anchor_position(a, b, t1) - linear_anchor_position(a, b, t0))
            fallback = True
        centers[j + 1] = nxt
        rows.append({'direction': 'forward', 'from_frame_1based': t0 + 1, 'to_frame_1based': t1 + 1, 'fallback': fallback, **step})
    err = np.array([b.x, b.y]) - centers[-1]
    alpha = np.linspace(0, 1, n)[:, None]
    centers = centers + alpha * err
    return (centers, pd.DataFrame(rows))


def coarse_backward(prep: Sequence[np.ndarray], a: Anchor, b: Anchor) -> Tuple[np.ndarray, pd.DataFrame]:
    n = b.frame - a.frame + 1
    centers = np.zeros((n, 2), float)
    centers[-1] = [b.x, b.y]
    rows = []
    for j in range(n - 1, 0, -1):
        t0 = a.frame + j
        t1 = t0 - 1
        cur = centers[j]
        step = track_grid_one_step(prep[t0], prep[t1], cur[0], cur[1])
        if step['accepted']:
            prv = cur + [step['dx'], step['dy']]
            fallback = False
        else:
            prv = cur + (linear_anchor_position(a, b, t1) - linear_anchor_position(a, b, t0))
            fallback = True
        centers[j - 1] = prv
        rows.append({'direction': 'backward', 'from_frame_1based': t0 + 1, 'to_frame_1based': t1 + 1, 'fallback': fallback, **step})
    err = np.array([a.x, a.y]) - centers[0]
    alpha = np.linspace(1, 0, n)[:, None]
    centers = centers + alpha * err
    return (centers, pd.DataFrame(rows))


def coarse_interval(prep: Sequence[np.ndarray], a: Anchor, b: Anchor) -> Tuple[np.ndarray, pd.DataFrame]:
    f, flog = coarse_forward(prep, a, b)
    br, blog = coarse_backward(prep, a, b)
    n = len(f)
    alpha = np.linspace(0, 1, n)[:, None]
    centers = (1 - alpha) * f + alpha * br
    centers[0] = [a.x, a.y]
    centers[-1] = [b.x, b.y]
    logs = pd.concat([flog, blog], ignore_index=True)
    return (centers, logs)



# ========================================================================
# Stage 2: template-matching residual refinement
# ========================================================================

def score_map_for_template(frame: np.ndarray, coarse_x: float, coarse_y: float, template: np.ndarray) -> np.ndarray:
    h = SNAP_TEMPLATE_HALF_WIDTH
    r = SNAP_SEARCH_RADIUS
    search = extract_patch(frame, coarse_x, coarse_y, h + r)
    sp = preprocess_match(search)
    tp = preprocess_match(template)
    result = cv2.matchTemplate(sp.astype(np.float32), tp.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    expected = 2 * r + 1
    if result.shape != (expected, expected):
        raise RuntimeError(f'matchTemplate returned {result.shape}; expected {(expected, expected)}')
    return result.astype(np.float32)


def robust_low_threshold(values: np.ndarray) -> float:
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return ABS_BAD_MATCH_SCORE
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad <= 1e-12:
        robust = med
    else:
        robust = med - BAD_SCORE_MAD_MULTIPLIER * 1.4826 * mad
    return max(ABS_BAD_MATCH_SCORE, robust)


def build_residual_emissions(vessel: np.ndarray, a: Anchor, b: Anchor, coarse_centers: np.ndarray) -> Dict[str, np.ndarray]:
    n = len(coarse_centers)
    h = SNAP_TEMPLATE_HALF_WIDTH
    r = SNAP_SEARCH_RADIUS
    tpl_a = extract_patch(vessel[a.frame], a.x, a.y, h)
    tpl_b = extract_patch(vessel[b.frame], b.x, b.y, h)
    side = 2 * r + 1
    emissions = np.empty((n, side, side), np.float32)
    max_scores = np.full(n, np.nan)
    sharp = np.full(n, np.nan)
    for j in range(n):
        t = a.frame + j
        alpha = j / max(1, n - 1)
        ma = score_map_for_template(vessel[t], coarse_centers[j, 0], coarse_centers[j, 1], tpl_a)
        mb = score_map_for_template(vessel[t], coarse_centers[j, 0], coarse_centers[j, 1], tpl_b)
        e = (1 - alpha) * ma + alpha * mb
        emissions[j] = e
        max_scores[j] = float(np.max(e))
        sharp[j] = local_sharpness(vessel[t], coarse_centers[j, 0], coarse_centers[j, 1])
    return {'emissions': emissions, 'max_scores': max_scores, 'sharpness': sharp}


def residual_states() -> Tuple[np.ndarray, int]:
    r = SNAP_SEARCH_RADIUS
    vals = np.arange(-r, r + 1, dtype=float)
    states = np.array([(dx, dy) for dy in vals for dx in vals], dtype=float)
    zero = int(np.where((states[:, 0] == 0) & (states[:, 1] == 0))[0][0])
    return (states, zero)


def transition_penalty(states: np.ndarray) -> np.ndarray:
    dx = states[:, 0][:, None] - states[:, 0][None, :]
    dy = states[:, 1][:, None] - states[:, 1][None, :]
    dist2 = dx * dx + dy * dy
    dist = np.sqrt(dist2)
    p = SNAP_SMOOTH_LAMBDA * dist2
    excess = np.clip(dist - LARGE_RESIDUAL_STEP_PX, 0, None)
    p += LARGE_STEP_EXTRA_PENALTY * excess * excess
    return p.astype(np.float32)


def solve_residual_path(emissions: np.ndarray, tracking_bad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = emissions.shape[0]
    flat = emissions.reshape(n, -1).astype(np.float32)
    states, zero = residual_states()
    trans = transition_penalty(states)
    for t in range(n):
        if tracking_bad[t]:
            flat[t, :] = 0.0
    flat[0, :] = -np.inf
    flat[0, zero] = 0.0
    flat[-1, :] = -np.inf
    flat[-1, zero] = 0.0
    s = len(states)
    back = np.zeros((n, s), np.int32)
    dp = flat[0].copy()
    for t in range(1, n):
        scores = dp[:, None] - trans
        best_prev = np.argmax(scores, axis=0)
        best = scores[best_prev, np.arange(s)]
        dp = flat[t] + best
        back[t] = best_prev
    path = np.zeros(n, np.int32)
    path[-1] = zero
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    offsets = states[path]
    original_flat = emissions.reshape(n, -1)
    chosen = original_flat[np.arange(n), path]
    return (offsets, chosen)


def refine_interval(vessel: np.ndarray, a: Anchor, b: Anchor, coarse_centers: np.ndarray) -> Tuple[np.ndarray, pd.DataFrame]:
    data = build_residual_emissions(vessel, a, b, coarse_centers)
    max_scores = data['max_scores']
    sharp = data['sharpness']
    thresh = robust_low_threshold(max_scores)
    tracking_bad = max_scores < thresh
    tracking_bad[0] = False
    tracking_bad[-1] = False
    sharp_ratio = local_median_ratio(sharp, SHARPNESS_LOCAL_RADIUS)
    quant_bad = (max_scores < ABS_QUANT_BAD_MATCH_SCORE) | tracking_bad & np.isfinite(sharp_ratio) & (sharp_ratio < CATASTROPHIC_SHARPNESS_RATIO)
    quant_bad[0] = False
    quant_bad[-1] = False
    offsets, chosen = solve_residual_path(data['emissions'], tracking_bad)
    refined = coarse_centers + offsets
    refined[0] = [a.x, a.y]
    refined[-1] = [b.x, b.y]
    mag = np.sqrt(np.sum(offsets * offsets, axis=1))
    boundary = (np.abs(offsets[:, 0]) >= SNAP_SEARCH_RADIUS) | (np.abs(offsets[:, 1]) >= SNAP_SEARCH_RADIUS)
    qc = pd.DataFrame({'frame_1based': np.arange(a.frame, b.frame + 1) + 1, 'residual_dx_px': offsets[:, 0], 'residual_dy_px': offsets[:, 1], 'residual_magnitude_px': mag, 'residual_at_search_boundary': boundary, 'best_template_score': max_scores, 'chosen_template_score': chosen, 'tracking_excluded_from_snap': tracking_bad, 'quantification_excluded': quant_bad, 'sharpness_ratio_local': sharp_ratio, 'tracking_score_threshold': thresh})
    return (refined, qc)



# ========================================================================
# Frame transforms and TIFF export
# ========================================================================

def centers_to_transform(centers: np.ndarray, target: Anchor) -> pd.DataFrame:
    return pd.DataFrame({'frame_1based': np.arange(len(centers)) + 1, 'landmark_x_px': centers[:, 0], 'landmark_y_px': centers[:, 1], 'shift_x_px': target.x - centers[:, 0], 'shift_y_px': target.y - centers[:, 1], 'method': METHOD_NAME})


def cast_like(img: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.rint(img), info.min, info.max).astype(dtype)
    return img.astype(dtype)


def write_stabilized_movie(path: Path, movie: np.ndarray, transform: pd.DataFrame):
    sx = transform['shift_x_px'].to_numpy(float)
    sy = transform['shift_y_px'].to_numpy(float)
    out = np.empty_like(movie)
    for t in range(movie.shape[0]):
        for c in range(movie.shape[1]):
            shifted = ndi_shift(movie[t, c].astype(np.float32), shift=(sy[t], sx[t]), order=INTERPOLATION_ORDER, mode='constant', cval=0, prefilter=False)
            out[t, c] = cast_like(shifted, movie.dtype)
    big = out.nbytes > int(3.5 * 1024 ** 3)
    tifffile.imwrite(path, out, imagej=not big, bigtiff=big, metadata={'axes': 'TCYX'}, compression=TIFF_COMPRESSION)



# =============================================================================
# Stabilization workflow
# =============================================================================

def track_landmark(vessel: np.ndarray, anchors: Sequence[Anchor]) -> np.ndarray:
    """Determine landmark coordinates for each frame from vessel images and ROIs.

    Positions outside the annotated frame interval retain the nearest manual
    anchor, matching the behavior of the original analysis script. For
    reliable full-video stabilization, provide anchors near both ends.
    """
    preprocessed = [preprocess_for_lk(frame) for frame in vessel]
    centers = np.zeros((len(vessel), 2), dtype=float)
    centers[:anchors[0].frame + 1] = (anchors[0].x, anchors[0].y)
    centers[anchors[-1].frame:] = (anchors[-1].x, anchors[-1].y)

    for left, right in zip(anchors[:-1], anchors[1:]):
        coarse, _flow_diagnostics = coarse_interval(preprocessed, left, right)
        refined, _snap_diagnostics = refine_interval(vessel, left, right, coarse)
        centers[left.frame:right.frame + 1] = refined
    return centers


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stabilize a two-channel intravital TIFF using a Fiji point-ROI ZIP."
    )
    parser.add_argument("--video", type=Path, required=True, help="Input multi-frame TIFF")
    parser.add_argument("--rois", type=Path, required=True, help="Fiji ROI Manager point-ROI ZIP")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output stabilized TIFF (default: <video_stem>_stabilized.tif alongside input)"
    )
    parser.add_argument(
        "--save-transform", action="store_true",
        help="Also save the frame-by-frame x/y translations as a CSV file."
    )
    args = parser.parse_args()
    video = args.video.expanduser().resolve()
    rois = args.rois.expanduser().resolve()
    output = (args.output.expanduser().resolve() if args.output else
              video.with_name(f"{video.stem}_stabilized.tif"))

    if not video.is_file():
        parser.error(f"Video not found: {video}")
    if not rois.is_file():
        parser.error(f"ROI archive not found: {rois}")
    if output.suffix.lower() not in (".tif", ".tiff"):
        parser.error("Output must be a .tif or .tiff file")
    if output in (video, rois):
        parser.error("Output path must not overwrite the input TIFF or ROI ZIP")

    print(f"Reading: {video.name}")
    movie, axes = load_tiff(video)
    print(f"Input axes: {axes}; frames/channels/height/width: {movie.shape}")
    anchors = load_anchors(rois, len(movie))
    if len(anchors) < 4:
        raise ValueError("At least four point anchors on distinct frames are required")
    print(f"Loaded {len(anchors)} manual anchors from {rois.name}")
    if anchors[0].frame > 0 or anchors[-1].frame < len(movie) - 1:
        warnings.warn("ROI anchors do not cover the entire movie; the "
                      "outer frames retain the nearest landmark position")

    vessel = movie[:, VESSEL_CHANNEL]
    centers = track_landmark(vessel, anchors)
    reference = choose_target_anchor(anchors)  # Same reference as original.
    transform = centers_to_transform(centers, reference)

    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing stabilized movie: {output}")
    write_stabilized_movie(output, movie, transform)

    if args.save_transform:
        transform_path = output.with_name(f"{output.stem}_transform.csv")
        transform.to_csv(transform_path, index=False)
        print(f"Frame shifts: {transform_path}")
    print("Done.")


if __name__ == "__main__":
    main()
