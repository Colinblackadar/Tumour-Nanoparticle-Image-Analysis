# Tumour-Nanoparticle-Image-Analysis

Python scripts for analysing nanoparticle distribution in tumour tissue sections and stabilizing intravital microscopy time-lapses. These tools implement two image-processing workflows used to investigate endothelial-associated nanoparticle signal and its spatial distribution.

The repository contains two independent scripts. Each can be run separately; raw microscopy data are not included.

## Scripts

### 1. Tumour nanoparticle spatial analysis

**`ec_np_spatial_analysis.py`** quantifies nanoparticle signal in tumour tissue sections using DAPI, CD31, and darkfield microscopy channels. It identifies valid tissue from DAPI, segments CD31-positive vessels, corrects spatially varying darkfield background, and calculates distances to the nearest vessel.

It reports four measurements:

1. Mean background-corrected nanoparticle intensity across valid tumour tissue.
2. Mean background-corrected nanoparticle intensity within the CD31-positive vessel mask.
3. Fraction of the total positive, background-corrected tumour nanoparticle signal located within the vessel mask.
4. Positive-signal-weighted mean distance to the nearest CD31-positive vessel (µm).

Inputs: three-channel, single-plane TIFF images and a CSV connecting each filename to a biological sample and timepoint. The output is an Excel workbook with **per-image** and **per-sample/timepoint** measurements, plus image-processing QC images. Sections from the same sample/timepoint are pooled from their pixel and signal totals. **No statistical tests are performed**; downstream statistics can be completed separately.

Install:

```bash
python -m pip install -r requirements_spatial_analysis.txt
```

Example sample CSV (`samples.csv`):

```csv
file,mouse,timepoint_h,slice
section_001.tif,sample_A,1,1
section_002.tif,sample_A,1,2
section_003.tif,sample_B,8,1
```

Run:

```bash
python ec_np_spatial_analysis.py /path/to/tiff_folder \
  --metadata-csv /path/to/samples.csv \
  --pixel-size-um YOUR_PIXEL_SIZE
```

Replace `YOUR_PIXEL_SIZE` with the calibrated micrometres-per-pixel value, or omit this option when valid micrometre calibration is embedded in the TIFF. Channel order defaults to DAPI, CD31, and darkfield (zero-based indices 0, 1, 2); alternate orders can be specified using command-line flags.

[Full instructions, processing settings, and output definitions](docs/spatial_analysis.md).

### 2. Intravital video stabilization

**`intravital_roi_stabilization.py`** stabilizes a two-channel intravital TIFF time-lapse using manually placed Fiji/ImageJ point ROIs on a vascular landmark. It estimates landmark movement with bidirectional Lucas–Kanade optical flow and local template matching, then applies the same translation to both channels.

Inputs: a two-channel TIFF time-lapse (vessel channel first, nanoparticle channel second) and a Fiji ROI Manager ZIP containing **at least four point ROIs on distinct frames**, marking the same landmark. Name ROIs with their one-based frame numbers (e.g., `F0001`, `F0020`) or use compatible stored Fiji frame positions. Ideally annotate near the first and last frames.

Install:

```bash
python -m pip install -r requirements_intravital_stabilization.txt
```

Run:

```bash
python intravital_roi_stabilization.py \
  --video /path/to/movie.tif \
  --rois /path/to/point_rois.zip
```

The output is a stabilized two-channel TIFF. Add `--save-transform` to export frame-by-frame x/y shifts as a CSV. No nanoparticle quantification or statistical analysis is performed by this script.

[Full instructions, stabilization algorithm, and output details](docs/intravital_stabilization.md).

## Reproducibility notes

- The image segmentation and stabilization parameters are documented in the linked methods pages and defined in each script.
- Inspect QC images and stabilized movies before interpreting measurements or using exported video in figures.
- Pixel-size calibration and spatial registration of channels must be correct for meaningful spatial measurements.
- The fraction of nanoparticle signal in vessels is an **integrated positive-signal fraction**, **not** a ratio of vessel MFI to tumour MFI.
- Source TIFFs and Fiji ROI files are not included in this code repository.

For all command-line options, run `python ec_np_spatial_analysis.py --help` or `python intravital_roi_stabilization.py --help`.
