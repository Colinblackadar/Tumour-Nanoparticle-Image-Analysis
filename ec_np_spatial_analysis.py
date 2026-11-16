#!/usr/bin/env python3
"""Quantify four tumour nanoparticle readouts relative to CD31-positive vessels.

Required input: multichannel TIFF images (DAPI, CD31, darkfield nanoparticle
signal) and a CSV manifest with file,mouse,timepoint_h columns (slice optional).

The processing pipeline and default parameters are unchanged from the original
spatial-analysis script:
  1. Define valid tumour tissue from DAPI, excluding low-density regions and
     tissue edges.
  2. Subtract local CD31 background and segment CD31-positive vessel regions.
  3. Estimate and subtract low-frequency darkfield background.
  4. Compute whole-tumour corrected NP MFI, vessel corrected NP MFI, vessel
     fraction of total positive NP signal, and signal-weighted mean distance
     from the nearest CD31-positive vessel.
  5. Pool raw numerators and denominators across sections belonging to the
     same biological replicate and timepoint (without statistical testing).

Results: Excel workbook (Per_Image, Per_Mouse, Parameters, Processing_Log) and
quality-control overlays. The script does not calculate p-values, means across
mice, error bars, or any other statistical comparisons.

See README_spatial_analysis.md for definitions, requirements, and run examples.
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import sys
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree

import cv2
import numpy as np
import tifffile
import xlsxwriter
from scipy.ndimage import binary_fill_holes

SUPPORTED_EXTENSIONS = {".tif", ".tiff"}

# Locked analysis defaults from the original study; override only intentionally.
DEFAULTS = {
    "mask_downsample": 4,
    "tissue_offset": 350.0,
    "lowdensity_offset": 650.0,
    "density_sigma_um": 20.0,
    "close_um": 80.0,
    "edge_exclude_um": 25.0,
    "min_lowdensity_area_um2": 5000.0,
    "cd31_bg_sigma_um": 15.5,
    "cd31_threshold": 1200.0,
    "min_vessel_area_um2": 20.0,
    "darkfield_bg_sigma_um": 130.0,
    "background_downsample": 8,
    "background_clip_percentile": 95.0,
    "tile_size": 1024,
    "qc_max_dimension": 1800,
    "distance_downsample": 2,
}




# Original processing function: blur
def blur(x, sigma_px):
    return cv2.GaussianBlur(
        x,
        (0, 0),
        sigmaX=max(0.01, float(sigma_px)),
        sigmaY=max(0.01, float(sigma_px)),
        borderType=cv2.BORDER_REFLECT,
    )



# Original processing function: block_mean_downsample
def block_mean_downsample(image, factor):
    f = max(1, int(factor))
    if f == 1:
        return np.asarray(image).astype(np.float32)
    h, w = image.shape
    h2 = (h // f) * f
    w2 = (w // f) * f
    if h2 < f or w2 < f:
        raise ValueError("Image too small for downsample factor %d" % f)
    view = np.asarray(image[:h2, :w2])
    return view.reshape(h2 // f, f, w2 // f, f).mean(axis=(1, 3), dtype=np.float32)



# Original processing function: block_max_downsample_mask
def block_max_downsample_mask(mask, factor):
    f = max(1, int(factor))
    if f == 1:
        return np.asarray(mask, dtype=bool)
    h, w = mask.shape
    h2 = (h // f) * f
    w2 = (w // f) * f
    if h2 < f or w2 < f:
        raise ValueError("Mask too small for downsample factor %d" % f)
    view = np.asarray(mask[:h2, :w2], dtype=np.uint8)
    return view.reshape(h2 // f, f, w2 // f, f).max(axis=(1, 3)).astype(bool)



# Original processing function: largest_component
def largest_component(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask.astype(bool)
    k = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return labels == k



# Original processing function: keep_large
def keep_large(mask, min_px):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    keep = np.zeros(n, np.uint8)
    if n > 1:
        keep[1:] = (stats[1:, cv2.CC_STAT_AREA] >= int(min_px)).astype(np.uint8)
    out = keep[labels].astype(bool)
    del labels
    return out



# Original processing function: dapi_background_mode
def dapi_background_mode(dapi_small):
    nz = dapi_small[dapi_small > 0]
    if nz.size == 0:
        return 0.0
    upper = max(65537, int(nz.max()) + 17)
    h, e = np.histogram(nz, bins=np.arange(0, upper, 16))
    i = int(np.argmax(h))
    return float((e[i] + e[i + 1]) / 2)



# Original processing function: px_size_um
def px_size_um(path: Path):
    """Return calibrated isotropic µm/pixel; do not guess a pixel size.

    TIFF XResolution alone is not reliable: many uncalibrated TIFFs carry a
    default value of 1 pixel per unspecified unit. Accept explicit OME or
    ImageJ micrometre metadata; otherwise require --pixel-size-um.
    """
    with tifffile.TiffFile(path) as tf:
        if tf.ome_metadata:
            root = ElementTree.fromstring(tf.ome_metadata)
            for element in root.iter():
                if element.tag.split("}")[-1] != "Pixels":
                    continue
                x_raw = element.attrib.get("PhysicalSizeX")
                if x_raw is None:
                    continue
                unit = element.attrib.get("PhysicalSizeXUnit", "µm").lower()
                conversions = {"µm": 1.0, "μm": 1.0, "um": 1.0,
                               "nm": 0.001, "mm": 1000.0}
                factor = conversions.get(unit)
                if factor is None:
                    continue
                x_um = float(x_raw) * factor
                y_raw = element.attrib.get("PhysicalSizeY")
                if y_raw is not None:
                    y_unit = element.attrib.get("PhysicalSizeYUnit", "µm").lower()
                    y_factor = conversions.get(y_unit)
                    if y_factor is not None:
                        y_um = float(y_raw) * y_factor
                        if not np.isclose(x_um, y_um, rtol=0.001):
                            raise ValueError("Anisotropic pixel spacing; supply isotropic input or adapt distance calculations")
                if np.isfinite(x_um) and x_um > 0:
                    return x_um

        page = tf.pages[0]
        xr = page.tags.get("XResolution")
        if xr is None:
            return None
        resolution = xr.value
        ppu = (resolution[0] / resolution[1] if isinstance(resolution, tuple)
               else float(resolution))
        description = page.description or ""
        imagej = tf.imagej_metadata or {}
        unit = str(imagej.get("unit", "")).lower()
        markers = ("unit=µm", "unit=μm", "unit=um", "unit=micron",
                   "unit=micrometer", "unit=micrometre")
        has_micrometre_unit = (unit in ("µm", "μm", "um", "micron", "microns",
                                        "micrometer", "micrometre")
                               or any(marker in description.lower() for marker in markers))
        if has_micrometre_unit and np.isfinite(ppu) and ppu > 0:
            return 1.0 / ppu
    return None



# Original processing function: load_sample_manifest
def load_sample_manifest(csv_path: Path, input_dir: Path):
    """Read image-to-sample mappings independently of TIFF file names.

    Required columns: file, mouse, timepoint_h. Optional: slice (integer).
    The `mouse` field identifies the biological sample; it may be any string.
    Relative filenames must be directly inside input_dir.
    """
    manifest = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"file", "mouse", "timepoint_h"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError("Manifest requires columns: file, mouse, timepoint_h (optional: slice)")
        for row_number, row in enumerate(reader, start=2):
            filename = (row.get("file") or "").strip()
            mouse = (row.get("mouse") or "").strip()
            if not filename or not mouse or Path(filename).name != filename:
                raise ValueError("Manifest row %d: provide a filename (not a path) and a biological sample ID" % row_number)
            if filename in manifest:
                raise ValueError("Manifest contains duplicate file entry: %s" % filename)
            path = input_dir / filename
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS or not path.is_file():
                raise ValueError("Manifest row %d: TIFF not found: %s" % (row_number, filename))
            try:
                time_h = float((row.get("timepoint_h") or "").strip())
            except ValueError as exc:
                raise ValueError("Manifest row %d: invalid timepoint_h" % row_number) from exc
            if not math.isfinite(time_h):
                raise ValueError("Manifest row %d: timepoint_h must be finite" % row_number)
            section_value = (row.get("slice") or "").strip()
            try:
                section = int(section_value) if section_value else None
            except ValueError as exc:
                raise ValueError("Manifest row %d: slice must be an integer" % row_number) from exc
            if section is not None and section <= 0:
                raise ValueError("Manifest row %d: slice must be positive" % row_number)
            manifest[filename] = {"mouse": mouse, "timepoint_h": time_h, "slice": section}
    if not manifest:
        raise ValueError("Manifest contains no images")

    # Missing section numbers are assigned within each sample/timepoint, with
    # existing explicitly assigned numbers reserved. This never reads names.
    used = defaultdict(set)
    for meta in manifest.values():
        if meta["slice"] is not None:
            used[(meta["mouse"], meta["timepoint_h"])].add(meta["slice"])
    for meta in manifest.values():
        if meta["slice"] is None:
            group = (meta["mouse"], meta["timepoint_h"])
            section = 1
            while section in used[group]:
                section += 1
            meta["slice"] = section
            used[group].add(section)
    return manifest



# Original processing function: open_channel_memmap
def open_channel_memmap(path: Path, channels=(0, 1, 2)):
    arr = tifffile.memmap(path)
    arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise ValueError("Expected 3-channel TIFF; got shape=%s" % (arr.shape,))
    if len(set(channels)) != 3 or min(channels) < 0:
        raise ValueError("DAPI, CD31 and darkfield indices must be distinct non-negative integers")
    if 3 <= arr.shape[0] <= 10 and max(channels) < arr.shape[0]:
        return arr, arr[channels[0]], arr[channels[1]], arr[channels[2]]
    if 3 <= arr.shape[-1] <= 10 and max(channels) < arr.shape[-1]:
        return arr, arr[..., channels[0]], arr[..., channels[1]], arr[..., channels[2]]
    raise ValueError("Cannot identify channel dimension from TIFF shape=%s" % (arr.shape,))



# Original processing function: dapi_masks_small
def dapi_masks_small(
    dapi, px, downsample, tissue_offset, lowdensity_offset, density_sigma_um,
    close_um, edge_exclude_um, min_lowdensity_area_um2,
):
    s = max(1, int(downsample))
    dapi_small = block_mean_downsample(dapi, s)
    pxs = px * s
    bg = dapi_background_mode(dapi_small)

    density = blur(dapi_small.astype(np.float32), density_sigma_um / pxs)
    seed = density > (bg + tissue_offset)

    r = max(1, int(round(close_um / pxs)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    section = cv2.morphologyEx(seed.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    section = binary_fill_holes(largest_component(section))

    low = section & (density < (bg + lowdensity_offset))
    min_px = max(1, int(round(min_lowdensity_area_um2 / (pxs ** 2))))
    low = keep_large(low, min_px)

    r = max(1, int(round(edge_exclude_um / pxs)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    inner = cv2.erode(section.astype(np.uint8), kernel).astype(bool)
    valid = section & inner & ~low

    del density, seed, inner
    gc.collect()
    return dapi_small, section.astype(bool), valid.astype(bool), low.astype(bool), bg



# Original processing function: upscale_mask
def upscale_mask(mask_small, full_shape):
    h, w = full_shape
    return cv2.resize(mask_small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)



# Original processing function: segment_cd31_tiled
def segment_cd31_tiled(cd31, valid_full, px, out_memmap_path, tile_size, bg_sigma_um, threshold, min_area_um2):
    h, w = cd31.shape
    sigma_px = bg_sigma_um / px
    overlap = max(32, int(math.ceil(4 * sigma_px)) + 8)
    min_px = max(1, int(round(min_area_um2 / (px ** 2))))

    vessel = np.memmap(out_memmap_path, dtype=np.uint8, mode="w+", shape=(h, w))
    vessel[:] = 0

    for y0 in range(0, h, tile_size):
        y1 = min(h, y0 + tile_size)
        ey0, ey1 = max(0, y0 - overlap), min(h, y1 + overlap)
        for x0 in range(0, w, tile_size):
            x1 = min(w, x0 + tile_size)
            ex0, ex1 = max(0, x0 - overlap), min(w, x1 + overlap)

            src = np.asarray(cd31[ey0:ey1, ex0:ex1]).astype(np.float32)
            local_bg = blur(src, sigma_px)
            corrected = src - local_bg
            mask_ext = corrected > threshold
            mask_ext &= valid_full[ey0:ey1, ex0:ex1]
            if min_px > 1 and np.any(mask_ext):
                mask_ext = keep_large(mask_ext, min_px)

            cy0, cy1 = y0 - ey0, y1 - ey0
            cx0, cx1 = x0 - ex0, x1 - ex0
            vessel[y0:y1, x0:x1] = mask_ext[cy0:cy1, cx0:cx1].astype(np.uint8)
            del src, local_bg, corrected, mask_ext

    vessel.flush()
    return vessel



# Original processing function: estimate_darkfield_background_small
def estimate_darkfield_background_small(dark, section_small, px, sigma_um, background_downsample, clip_pct):
    ds = max(1, int(background_downsample))
    d = block_mean_downsample(dark, ds)
    sh, sw = d.shape
    m = cv2.resize(section_small.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST).astype(bool)
    vals = d[m]
    if vals.size == 0:
        raise ValueError("No tissue pixels available for darkfield background estimation.")
    upper = float(np.percentile(vals, clip_pct))
    np.minimum(d, upper, out=d)
    wt = m.astype(np.float32)
    sigma_px = max(0.5, sigma_um / (px * ds))
    num = blur(d * wt, sigma_px)
    den = blur(wt, sigma_px)
    bg = num / np.maximum(den, 1e-4)
    return bg.astype(np.float32)



# Original processing function: resized_field_tile
def resized_field_tile(field_small, full_shape, y0, y1, x0, x1, interpolation=cv2.INTER_LINEAR):
    h, w = full_shape
    fh, fw = field_small.shape
    xs = np.arange(x0, x1, dtype=np.float32)
    ys = np.arange(y0, y1, dtype=np.float32)
    if w > 1:
        xs *= (fw - 1) / (w - 1)
    else:
        xs[:] = 0
    if h > 1:
        ys *= (fh - 1) / (h - 1)
    else:
        ys[:] = 0
    map_x, map_y = np.meshgrid(xs, ys)
    return cv2.remap(field_small, map_x, map_y, interpolation=interpolation, borderMode=cv2.BORDER_REPLICATE)



# Original processing function: make_distance_map_small
def make_distance_map_small(exact_mask, px, factor):
    factor = max(1, int(factor))
    vessel_small = block_max_downsample_mask(exact_mask, factor)
    if not np.any(vessel_small):
        raise ValueError("CD31 mask contains no vessels; cannot calculate vessel distance.")
    # OpenCV distanceTransform measures distance to zero pixels. Vessel pixels = zero.
    inv = (~vessel_small).astype(np.uint8)
    dist_px = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    return (dist_px * (px * factor)).astype(np.float32), vessel_small



# Original processing function: robust_u8
def robust_u8(x, mask=None, low=1.0, high=99.7):
    x = np.asarray(x, dtype=np.float32)
    vals = x[np.isfinite(x)] if mask is None else x[mask & np.isfinite(x)]
    if vals.size == 0:
        return np.zeros(x.shape, dtype=np.uint8)
    a, b = np.percentile(vals, [low, high])
    if not np.isfinite(a) or not np.isfinite(b) or b <= a:
        a, b = float(np.nanmin(vals)), float(np.nanmax(vals))
    y = np.clip((x - a) / (b - a + 1e-9), 0, 1)
    return (255 * y).astype(np.uint8)



# Original processing function: gray_rgb
def gray_rgb(g):
    return np.dstack([g, g, g])



# Original processing function: save_rgb
def save_rgb(path, rgb):
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

# ---------------------------------------------------------------------------
# Four measurements only
# ---------------------------------------------------------------------------

def quantify_four(dark, valid_full, vessel_mask, bg_small, distance_small, tile_size):
    """Accumulate original-method numerators and denominators per image.

    Signed background-corrected values (including negatives) are used for
    both MFIs. Positive-only corrected values (max(corrected, 0)) are used
    for total NP signal fraction and signal-weighted distance, matching the
    previous full analysis exactly.
    """
    height, width = dark.shape
    tumour_pixels = vessel_pixels = 0
    tumour_corrected_sum = vessel_corrected_sum = 0.0
    tumour_positive_sum = vessel_positive_sum = weighted_distance_sum = 0.0

    for y0 in range(0, height, tile_size):
        y1 = min(height, y0 + tile_size)
        for x0 in range(0, width, tile_size):
            x1 = min(width, x0 + tile_size)
            raw = np.asarray(dark[y0:y1, x0:x1]).astype(np.float32)
            local_background = resized_field_tile(bg_small, (height, width), y0, y1, x0, x1)
            corrected = raw - local_background
            positive = np.maximum(corrected, 0.0)
            valid = valid_full[y0:y1, x0:x1]
            distance = resized_field_tile(distance_small, (height, width), y0, y1, x0, x1)

            if np.any(valid):
                valid_corrected = corrected[valid]
                valid_positive = positive[valid]
                valid_distance = distance[valid]
                tumour_pixels += int(valid.sum())
                tumour_corrected_sum += float(valid_corrected.sum(dtype=np.float64))
                tumour_positive_sum += float(valid_positive.sum(dtype=np.float64))
                weighted_distance_sum += float((valid_positive * valid_distance).sum(dtype=np.float64))

            vessel = np.asarray(vessel_mask[y0:y1, x0:x1], dtype=bool)
            if np.any(vessel):
                vessel_corrected = corrected[vessel]
                vessel_positive = positive[vessel]
                vessel_pixels += int(vessel.sum())
                vessel_corrected_sum += float(vessel_corrected.sum(dtype=np.float64))
                vessel_positive_sum += float(vessel_positive.sum(dtype=np.float64))

    if tumour_pixels == 0:
        raise ValueError("The valid tumour tissue mask contains no pixels")
    if vessel_pixels == 0:
        raise ValueError("No CD31-positive vessel pixels were identified")
    # A specimen with no positive signal still yields both MFIs; its signal
    # fraction and signal-weighted distance are undefined (blank in Excel).
    positive_fraction = (vessel_positive_sum / tumour_positive_sum if tumour_positive_sum > 0
                         else float("nan"))
    mean_distance = (weighted_distance_sum / tumour_positive_sum if tumour_positive_sum > 0
                     else float("nan"))

    return {
        "whole_corrected_np_mfi": tumour_corrected_sum / tumour_pixels,
        "exact_corrected_np_mfi": vessel_corrected_sum / vessel_pixels,
        "exact_total_positive_signal_fraction": positive_fraction,
        "signal_weighted_mean_distance_um": mean_distance,
        # Internal quantities for exact pooling across sections of one mouse.
        "_whole_count": tumour_pixels,
        "_whole_corr_sum": tumour_corrected_sum,
        "_whole_pos_sum": tumour_positive_sum,
        "_exact_count": vessel_pixels,
        "_exact_corr_sum": vessel_corrected_sum,
        "_exact_pos_sum": vessel_positive_sum,
        "_weighted_distance_num": weighted_distance_sum,
    }


# ---------------------------------------------------------------------------
# Reviewable quality-control images (not additional quantitative readouts)
# ---------------------------------------------------------------------------

def save_qc(qc_dir, dapi, cd31, dark, section_small, valid_small,
            exact, bg_small, distance_small, max_dim):
    qc_dir.mkdir(parents=True, exist_ok=True)
    h, w = dapi.shape
    factor = max(1, int(math.ceil(max(h, w) / float(max_dim))))
    dapi_p = block_mean_downsample(dapi, factor)
    cd31_p = block_mean_downsample(cd31, factor)
    dark_p = block_mean_downsample(dark, factor)
    ph, pw = dapi_p.shape

    section_p = cv2.resize(section_small.astype(np.uint8), (pw, ph),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
    valid_p = cv2.resize(valid_small.astype(np.uint8), (pw, ph),
                         interpolation=cv2.INTER_NEAREST).astype(bool)
    vessel_p = cv2.resize(np.asarray(exact), (pw, ph),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
    background_p = cv2.resize(bg_small, (pw, ph), interpolation=cv2.INTER_LINEAR)
    corrected_p = dark_p - background_p
    distance_p = cv2.resize(distance_small, (pw, ph), interpolation=cv2.INTER_LINEAR)

    dapi_panel = gray_rgb(robust_u8(dapi_p, section_p))
    dapi_panel[section_p & ~valid_p] = (255, 0, 255)
    save_rgb(qc_dir / "01_DAPI_exclusion.png", dapi_panel)

    cd31_panel = gray_rgb(robust_u8(cd31_p, valid_p))
    cd31_panel[vessel_p] = (255, 0, 0)
    save_rgb(qc_dir / "02_CD31_segmentation.png", cd31_panel)

    bg_panel = gray_rgb(robust_u8(background_p, section_p, 1, 99))
    bg_panel[~section_p] = 0
    save_rgb(qc_dir / "03_darkfield_estimated_background.png", bg_panel)

    signal_panel = gray_rgb(robust_u8(np.maximum(corrected_p, 0), valid_p, 1, 99.7))
    signal_panel[~valid_p] = 0
    save_rgb(qc_dir / "04_darkfield_background_corrected.png", signal_panel)

    distance_panel = signal_panel.copy()
    for boundary in (3, 6, 10, 20, 40, 80):
        band = (np.abs(distance_p - boundary) <= max(1.0, boundary * 0.025)) & valid_p
        distance_panel[band] = (255, 255, 0)
    distance_panel[vessel_p] = (255, 0, 0)
    save_rgb(qc_dir / "05_distance_from_CD31_overlay.png", distance_panel)


def analyze_one(path, meta, out_root, args):
    image, dapi, cd31, dark = open_channel_memmap(
        path, (args.dapi_channel, args.cd31_channel, args.darkfield_channel)
    )
    h, w = dapi.shape
    px = args.pixel_size_um if args.pixel_size_um is not None else px_size_um(path)
    if px is None or not np.isfinite(px) or px <= 0:
        raise ValueError("Pixel calibration not found; specify --pixel-size-um in µm/pixel")

    dapi_small, section_small, valid_small, low_small, dapi_bg = dapi_masks_small(
        dapi, px, args.mask_downsample, args.tissue_offset, args.lowdensity_offset,
        args.density_sigma_um, args.close_um, args.edge_exclude_um,
        args.min_lowdensity_area_um2,
    )
    valid_full = upscale_mask(valid_small, (h, w))
    qc_dir = out_root / "QC" / path.name
    qc_dir.mkdir(parents=True, exist_ok=True)
    temporary_mask = qc_dir / ".cd31_mask_tmp.dat"

    try:
        vessel_mask = segment_cd31_tiled(
            cd31, valid_full, px, temporary_mask, args.tile_size,
            args.cd31_bg_sigma_um, args.cd31_threshold, args.min_vessel_area_um2,
        )
        background_small = estimate_darkfield_background_small(
            dark, section_small, px, args.darkfield_bg_sigma_um,
            args.background_downsample, args.background_clip_percentile,
        )
        distance_small, _ = make_distance_map_small(vessel_mask, px, args.distance_downsample)
        q = quantify_four(dark, valid_full, vessel_mask, background_small,
                          distance_small, args.tile_size)

        save_qc(qc_dir, dapi, cd31, dark, section_small, valid_small,
                vessel_mask, background_small, distance_small, args.qc_max_dimension)

        if args.save_fullres_masks:
            tifffile.imwrite(qc_dir / "fullres_valid_tumour_mask.tif",
                             valid_full.astype(np.uint8) * 255, compression="zlib")
            tifffile.imwrite(qc_dir / "fullres_CD31_mask.tif",
                             np.asarray(vessel_mask) * 255, compression="zlib")
        vessel_mask.flush()
        del vessel_mask
    finally:
        gc.collect()
        try:
            os.remove(temporary_mask)
        except OSError:
            pass

    return {
        "file": path.name,
        "mouse": meta["mouse"],
        "timepoint_h": meta["timepoint_h"],
        "slice": meta["slice"],
        **q,
    }


def build_per_mouse(image_rows):
    """Pool raw intensity sums and pixel counts; never treat slices as independent n."""
    grouped = defaultdict(list)
    for row in image_rows:
        grouped[(row["mouse"], row["timepoint_h"])].append(row)
    output = []
    for (mouse, time_h), rows in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        whole_count = sum(r["_whole_count"] for r in rows)
        vessel_count = sum(r["_exact_count"] for r in rows)
        whole_corr = sum(r["_whole_corr_sum"] for r in rows)
        vessel_corr = sum(r["_exact_corr_sum"] for r in rows)
        whole_positive = sum(r["_whole_pos_sum"] for r in rows)
        vessel_positive = sum(r["_exact_pos_sum"] for r in rows)
        weighted_distance = sum(r["_weighted_distance_num"] for r in rows)
        output.append({
            "mouse": mouse,
            "timepoint_h": time_h,
            "n_images": len(rows),
            "whole_corrected_np_mfi": whole_corr / whole_count,
            "exact_corrected_np_mfi": vessel_corr / vessel_count,
            "exact_total_positive_signal_fraction": (vessel_positive / whole_positive if whole_positive > 0 else float("nan")),
            "signal_weighted_mean_distance_um": (weighted_distance / whole_positive if whole_positive > 0 else float("nan")),
        })
    return output


# ---------------------------------------------------------------------------
# Excel: only two results tables, the parameters, and a processing log
# ---------------------------------------------------------------------------

def excel_safe(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_workbook(path, per_image, per_mouse, log_rows, args):
    workbook = xlsxwriter.Workbook(str(path))
    workbook.set_properties({
        "title": "Four-readout tumour nanoparticle spatial analysis",
        "subject": "Background-corrected NP MFI and vessel-associated signal",
    })
    header_fmt = workbook.add_format({"bold": True, "font_color": "white",
                                       "bg_color": "#244766", "border": 1,
                                       "text_wrap": True, "valign": "vcenter"})
    text_fmt = workbook.add_format({"border": 1})
    num_fmt = workbook.add_format({"border": 1, "num_format": "0.000000"})
    int_fmt = workbook.add_format({"border": 1, "num_format": "0"})
    frac_fmt = workbook.add_format({"border": 1, "num_format": "0.00%"})

    measurement_cols = [
        ("whole_corrected_np_mfi", "Total tumour NP MFI", num_fmt),
        ("exact_corrected_np_mfi", "NP MFI in CD31+ vessels", num_fmt),
        ("exact_total_positive_signal_fraction", "Fraction of total NP signal in vessels", frac_fmt),
        ("signal_weighted_mean_distance_um", "Signal-weighted distance to vessel (µm)", num_fmt),
    ]
    image_cols = [
        ("file", "File", text_fmt), ("mouse", "Biological sample", text_fmt),
        ("timepoint_h", "Timepoint (h)", num_fmt), ("slice", "Slice", int_fmt),
        *measurement_cols,
    ]
    mouse_cols = [
        ("mouse", "Biological sample", text_fmt),
        ("timepoint_h", "Timepoint (h)", num_fmt),
        ("n_images", "Number of sections", int_fmt),
        *measurement_cols,
    ]

    def write_sheet(name, rows, columns):
        sheet = workbook.add_worksheet(name)
        for c, (_, title, _) in enumerate(columns):
            sheet.write(0, c, title, header_fmt)
            sheet.set_column(c, c, 43 if c >= len(columns) - 4 else 19)
        sheet.set_row(0, 35)
        for r, row in enumerate(rows, 1):
            for c, (field, _, fmt) in enumerate(columns):
                v = excel_safe(row.get(field))
                if v is None:
                    sheet.write_blank(r, c, None, fmt)
                else:
                    sheet.write(r, c, v, fmt)
        sheet.freeze_panes(1, 0)
        if rows:
            sheet.autofilter(0, 0, len(rows), len(columns) - 1)

    write_sheet("Per_Image", per_image, image_cols)
    write_sheet("Per_Mouse", per_mouse, mouse_cols)

    sheet = workbook.add_worksheet("Parameters")
    sheet.write_row(0, 0, ["Parameter", "Value", "Description"], header_fmt)
    sheet.set_column(0, 0, 32)
    sheet.set_column(1, 1, 22)
    sheet.set_column(2, 2, 82)
    parameters = [
        ("analysis_date", datetime.now().isoformat(timespec="seconds"), "Run date (local computer time)"),
        ("dapi_channel", args.dapi_channel, "Zero-based input channel index"),
        ("cd31_channel", args.cd31_channel, "Zero-based input channel index"),
        ("darkfield_channel", args.darkfield_channel, "Zero-based input channel index"),
        ("pixel_size_um_override", args.pixel_size_um if args.pixel_size_um is not None else "TIFF metadata", "µm/pixel; individual TIFF calibration is used if unset"),
        ("valid_tissue", "DAPI derived", "Tissue, DAPI-low-density and peripheral exclusion masks"),
        ("mfi_definition", "signed", "Mean of signed, background-corrected NP intensity (negative residuals included)"),
        ("signal_fraction_definition", "positive-signal sum", "Sum max(corrected NP,0) inside exact CD31 / sum max(corrected NP,0) in valid tumour"),
        ("distance_definition", "positive-signal weighted", "Sum [max(corrected NP,0) × vessel distance] / sum max(corrected NP,0)"),
        ("pooling", "by sample and timepoint", "Raw pixel/intensity totals pooled across sections; no between-mouse statistics"),
        ("selected_timepoints_h", ",".join("%g" % t for t in args.timepoints) if args.timepoints else "all", "Selected timepoints (hours); all if omitted"),
    ]
    parameters.extend((name, getattr(args, name), "Original pipeline default; configurable at command line")
                      for name in DEFAULTS)
    for idx, values in enumerate(parameters, 1):
        for col, v in enumerate(values):
            sheet.write(idx, col, excel_safe(v))
    sheet.freeze_panes(1, 0)

    log_cols = [("file", "File", text_fmt), ("status", "Status", text_fmt),
                ("message", "Message", text_fmt)]
    write_sheet("Processing_Log", log_rows, log_cols)
    workbook.close()


def run_batch(args):
    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise SystemExit("Input folder does not exist: %s" % input_dir)
    try:
        manifest = load_sample_manifest(Path(args.metadata_csv).expanduser().resolve(), input_dir)
    except (OSError, ValueError) as exc:
        raise SystemExit("Invalid sample manifest: %s" % exc) from exc

    out_root = (Path(args.output_dir).expanduser().resolve() if args.output_dir
                else input_dir / "EC_NP_spatial_analysis")
    out_root.mkdir(parents=True, exist_ok=True)
    target = set(args.timepoints) if args.timepoints else None
    rows, log = [], []
    for file_number, (filename, meta) in enumerate(sorted(manifest.items()), 1):
        path = input_dir / filename
        if target is not None and meta["timepoint_h"] not in target:
            log.append({"file": filename, "status": "SKIPPED", "message": "Timepoint not selected"})
            continue
        print("[%d/%d] %s" % (file_number, len(manifest), filename), flush=True)
        try:
            row = analyze_one(path, meta, out_root, args)
            rows.append(row)
            log.append({"file": filename, "status": "OK", "message": "Processed"})
        except Exception as exc:
            log.append({"file": filename, "status": "ERROR", "message": str(exc)})
            print("ERROR for %s: %s" % (filename, exc), file=sys.stderr)
            if args.verbose_errors:
                traceback.print_exc()

    if not rows:
        raise SystemExit("No images were successfully processed")
    rows.sort(key=lambda r: (r["timepoint_h"], r["mouse"], r["slice"], r["file"]))
    mice = build_per_mouse(rows)
    output = out_root / "EC_NP_spatial_results.xlsx"
    write_workbook(output, rows, mice, log, args)
    print("Processed %d images from %d biological sample/timepoint groups." % (len(rows), len(mice)))
    print("Workbook: %s" % output)
    print("QC directory: %s" % (out_root / "QC"))
    if any(item["status"] == "ERROR" for item in log):
        print("WARNING: some images failed; see Processing_Log in the workbook")
    return output


def build_parser():
    parser = argparse.ArgumentParser(
        description="Measure four tumour nanoparticle spatial readouts, without statistical analysis"
    )
    parser.add_argument("input_dir", help="Folder of multichannel TIFF images")
    parser.add_argument("--metadata-csv", required=True,
                        help="CSV with file,mouse,timepoint_h (optional slice). Filenames can be arbitrary")
    parser.add_argument("--timepoints", nargs="+", type=float, default=None,
                        help="Optional subset of timepoints (hours); default all")
    parser.add_argument("--output-dir", default=None, help="Results directory")
    parser.add_argument("--pixel-size-um", type=float, default=None,
                        help="µm/pixel; required if no reliable TIFF calibration exists")
    parser.add_argument("--dapi-channel", type=int, default=0)
    parser.add_argument("--cd31-channel", type=int, default=1)
    parser.add_argument("--darkfield-channel", type=int, default=2)
    parser.add_argument("--save-fullres-masks", action="store_true",
                        help="Save valid tissue and exact vessel segmentation masks as TIFF")
    parser.add_argument("--verbose-errors", action="store_true")
    for option, default in DEFAULTS.items():
        kind = int if isinstance(default, int) else float
        parser.add_argument("--" + option.replace("_", "-"), type=kind, default=default)
    return parser


if __name__ == "__main__":
    run_batch(build_parser().parse_args())
