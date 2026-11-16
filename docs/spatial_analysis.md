# Tumour nanoparticle spatial analysis: four measurement outputs

This script quantifies four microscopy measurements of nanoparticle (NP) distribution in tumour sections, using DAPI to identify analysable tissue, CD31 to segment vessels, and darkfield intensity to quantify nanoparticle signal. It is adapted from the original analysis pipeline, retaining the original image-processing steps and default numerical parameters while removing study-specific file naming conventions and all statistical testing.

## Outputs

The Excel workbook `EC_NP_spatial_results.xlsx` contains two measurement tables:

- **`Per_Image`**: one row for each TIFF section, including file, biological sample, timepoint, section number, and the four values below.
- **`Per_Mouse`**: one row for each biological sample and timepoint. Sections from the same sample are pooled from the original signal sums and pixel counts **before** calculating the four values. Sections do not become separate biological replicates.

| Column name | Definition |
|---|---|
| `Total tumour NP MFI` | Arithmetic mean of **signed, background-corrected** darkfield intensity over every valid tumour pixel, including negative residual values |
| `NP MFI in CD31+ vessels` | Arithmetic mean of the same corrected intensity within the **exact CD31+ mask** (not a dilated mask) |
| `Fraction of total NP signal in vessels` | Sum of positive corrected signal (`max(corrected intensity, 0)`) in the exact CD31+ mask, divided by the sum of positive corrected signal in valid tumour tissue |
| `Signal-weighted distance to vessel (µm)` | Sum of `max(corrected intensity, 0) × distance to nearest CD31+ pixel`, divided by total positive corrected tumour signal |

**Important:** The third value is an *integrated signal fraction*, **not** vessel MFI divided by whole-tumour MFI. These are different measurements. The script retains the original `exact_total_positive_signal_fraction` definition so its output agrees with the original analysis.

**No statistical analysis is performed.** In particular, there are no p-values, timepoint averages across animals, SEM/SD calculations, statistical plots, or inferential tests. Use the `Per_Mouse` data for analysis in GraphPad Prism.

The workbook also includes `Parameters` (processing settings and measurement definitions) and `Processing_Log` (success/error tracking). Per-image QC PNGs are saved in `QC/<filename>/` to check segmentation and background subtraction. Optionally, you can export the full-resolution valid-tumour and CD31 masks for inspection.

## Requirements

Python 3.10+ and the packages in `requirements_spatial_analysis.txt`:

```bash
python -m pip install -r requirements_spatial_analysis.txt
```

## Inputs

Each image must be a **single-plane, three-channel TIFF** containing DAPI, CD31, and darkfield nanoparticle signal. Channel indices default to DAPI=0, CD31=1, darkfield=2, for channel-first (`C,Y,X`) or channel-last (`Y,X,C`) data. Use `--dapi-channel`, `--cd31-channel`, and `--darkfield-channel` if the order differs. The script requires a TIFF readable through `tifffile.memmap`; it does not automatically register or merge separate darkfield and fluorescence files.

Provide a separate CSV identifying images and biological samples; names of image files do not encode any data to the program:

```csv
file,mouse,timepoint_h,slice
section_001.tif,animal_A,1,1
section_002.tif,animal_A,1,2
section_003.tif,animal_B,8,1
```

`file`, `mouse`, and `timepoint_h` are required; `slice` is optional. Filenames must identify TIFF files directly in the input folder. All TIFFs listed in the CSV are analysed, unless filtered with `--timepoints`; unlisted TIFFs are ignored. Different images with the same sample ID and timepoint are treated as sections belonging to the same biological replicate.

### Run

```bash
python ec_np_spatial_analysis.py /path/to/tiff_folder \
  --metadata-csv /path/to/samples.csv \
  --pixel-size-um 1.0 \
  --output-dir /path/to/results
```

**The `1.0` value is an example placeholder**, not the experiment's calibration. Enter the acquisition-specific value in micrometres per pixel, or omit `--pixel-size-um` when the TIFF contains a suitable explicit micrometre calibration (e.g. OME metadata). The script does not assume any dataset-specific fallback pixel size. Use the *actual calibration used in the original analysis* to reproduce the reported results. Different pixel calibrations alter segmentation, distance estimates, and the analysis units.

Other optional flags include `--timepoints 1 8 24`, `--save-fullres-masks`, and `--verbose-errors`. To list all parameters:

```bash
python ec_np_spatial_analysis.py --help
```

## Image-processing steps (unchanged from the original)

1. **DAPI-derived tissue exclusion.** Downsample DAPI 4-fold. Estimate the DAPI background intensity from the mode of non-zero intensities. Smooth DAPI using a Gaussian sigma of 20 µm and threshold above the background by 350 intensity units. Apply closing with an elliptical kernel of radius 80 µm; retain the largest connected component and fill holes. Exclude low-DAPI-density regions (intensity below background +650; area ≥5000 µm²) and an outer 25 µm tissue border.
2. **CD31 vessel segmentation.** Subtract a local Gaussian background from CD31 (sigma 15.5 µm). Threshold corrected CD31 signal above 1200 intensity units within valid tissue. Remove small detected components below 20 µm². Process with overlapping tiles to reduce image memory requirements. The **exact** CD31 mask defines vessel-associated pixels. No 3/10 µm vessel dilations are needed for these four outputs.
3. **Darkfield background subtraction.** Downsample darkfield 8-fold; clip intensities at the 95th percentile of tissue pixels for *background estimation only*. Compute a tissue-weighted Gaussian background with sigma 130 µm and interpolate it to native resolution. Subtract this background field from the **original** darkfield image. Keep negative residuals when calculating corrected MFI. For signal fractions and signal-weighted distances only, use `max(corrected intensity, 0)`.
4. **Vessel distance map.** Block-maximum downsample the CD31 mask 2-fold, then apply the OpenCV L2 distance transform (5-pixel mask). Convert approximate pixel distances to micrometres and interpolate distances to full resolution. The distance output is the **positive-signal-weighted mean**, not the median.
5. **Measurement and pooling.** Accumulate corrected intensity, positive signal, pixel counts, and positive-signal-weighted distances across valid tumour and exact CD31 vessel pixels. Pool these sums over all sections with the same `mouse` and `timepoint_h` and recompute each readout. **No downstream statistics are calculated.**

The `Parameters` sheet records the numerical settings used for each run, all of which can be changed through command-line options; unchanged defaults preserve the original method.

## Quality control and interpretation

Review the QC images for the DAPI valid-tissue mask, CD31 vessel segmentation, estimated darkfield background, corrected NP signal, and distance contours. The analysis assumes the channels are spatially registered and that CD31-positive regions adequately identify vessels. The approximate distance transform is not a subpixel-accurate measure. Vessel association and distance to CD31 alone do not resolve the physical mechanism by which nanoparticles cross the vessel wall.
