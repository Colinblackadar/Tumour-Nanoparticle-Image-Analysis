# ROI-anchored intravital video stabilization

This Python script stabilizes a **two-channel intravital microscopy time-lapse** using manually annotated landmark positions saved in a **Fiji/ImageJ ROI Manager ZIP**. It accepts arbitrary input filenames and directories: no sample IDs, mouse labels, or hotspot numbers need to be encoded in the names.

## Inputs

- **Time-lapse TIFF:** a two-channel, 2D time series with a vessel reference image in **channel 1** and a nanoparticle signal in **channel 2**. TIFF metadata should identify the time and channel dimensions. If you have a z-stack, export a 2D time-lapse or projection first.
- **ROI Manager ZIP:** at least **four point ROIs on distinct frames**, all marking the **same recognizable vessel landmark**. Place ROIs across the movie, ideally including the first and last frames. The location may move with tissue motion.

To associate each point ROI with the correct frame, name it `F0001`, `F0020`, `F0040`, etc. (the number is the **one-based frame number**). Alternatively, the script can read a Fiji ROI's saved time/stack position when available. These are generic annotations, not input-file naming conventions. From ROI Manager, save the point ROIs as a `.zip` archive. Each ZIP describes **one** landmark's trajectory. For multiple regions, run the script separately with the corresponding ROI ZIP.

## Installation

With Python 3.10 or newer:

```bash
python -m pip install -r requirements_intravital_stabilization.txt
```

## Run

```bash
python intravital_roi_stabilization.py \
  --video "/path/to/any_video.tif" \
  --rois "/path/to/landmark_points.zip"
```

By default, the output is saved beside the input as `any_video_stabilized.tif`. Specify a different output filename and location with `--output`:

```bash
python intravital_roi_stabilization.py \
  --video "/path/to/any_video.tif" \
  --rois "/another/folder/landmark_points.zip" \
  --output "/path/to/results/stabilized_movie.tif"
```

To **optionally** save a CSV of the framewise x/y translations as well:

```bash
python intravital_roi_stabilization.py \
  --video "/path/to/any_video.tif" \
  --rois "/path/to/landmark_points.zip" \
  --save-transform
```

The script does not convert the TIFF to MP4, burn in timestamps, or make scale bars; those are separate presentation steps.

## Stabilization method

This is the same core registration method as in the original analysis script. The parameters are fixed at the beginning of the Python file.

1. The vessel channel is preprocessed with robust intensity normalization and CLAHE contrast enhancement.
2. Bidirectional, grid-based Lucas–Kanade optical flow estimates movement between manually annotated anchors: 3 × 3 tracking points, a 31-pixel window, and three pyramid levels, with forward/backward checking and interpolation between hard manual-anchor positions.
3. A local template-matching refinement uses the **raw vessel channel**, 25 × 25-pixel reference patches, a ±8-pixel search radius, and a temporal smoothness penalty of **0.08** to refine the tracked position.
4. Each frame is translated so the tracked landmark coincides with the **middle manual anchor** (in chronological anchor order). The identical translation is applied to both image channels.
5. Pixel values are resampled with first-order interpolation, constant-zero borders, and conversion back to the source data type. The output TIFF is saved with `TCYX` axes.

**Coverage note:** frames before the first point ROI and after the last point ROI retain the position of the nearest annotated anchor, following the original method. For reliable stabilization over the complete movie, include anchors near both ends.

## Output

- **`<name>_stabilized.tif`** — the stabilized two-channel time-lapse, preserving frame count, frame dimensions, and the input dtype.
- **`<name>_stabilized_transform.csv`** — optional; includes the 1-based frame index, estimated landmark position, and per-frame x/y shifts (pixels).

The script performs **no nanoparticle quantification, statistical tests, batch discovery, or automated exclusion of frames**. Review the stabilized movie for tracking quality before using it for analysis or supplementary videos. For quantitative intensity measurements, repeated interpolation can affect pixel values; consider applying the exported framewise transforms to original images or ROI coordinates as appropriate.
