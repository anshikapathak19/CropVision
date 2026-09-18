# CropVision

**CropVision** is a headless, classical Computer Vision toolkit built with Python 3.10+ and OpenCV. It analyzes crop leaf images, performs automated leaf segmentation, isolates visually abnormal regions (lesions, necrosis, chlorosis), computes spatial area metrics, and classifies overall visual severity into project-defined heuristic tiers.

> **Disclaimer**: This toolkit performs visual anomaly analysis based strictly on classical image color, intensity, and geometric properties. It does **not** provide agricultural, biological, or medical disease diagnosis.

---

## Problem Statement

Manual visual inspection of crop leaves for damage or discoloration in precision agriculture is subjective, slow, and non-quantitative. Automated image analysis requires lightweight, interpretable toolkits that can measure affected leaf areas without relying on heavy deep-learning dependencies or cloud infrastructure.

## Project Objectives

1. **Headless & Classical Architecture**: Implement pure classical computer vision routines using OpenCV and NumPy without deep learning frameworks.
2. **Robust Leaf Segmentation**: Segment crop leaves from background using configurable HSV thresholding and morphological hole-filling.
3. **Targeted Abnormality Detection**: Isolate visually abnormal regions (reddish/brown necrosis, chlorotic yellowing, dark lesions) strictly within the leaf boundary.
4. **Quantitative Metrics**: Compute leaf area in pixels, affected area in pixels, affected percentage, region centroids, bounding boxes, and region counts.
5. **Structured Multi-Format Exporters**: Generate JSON data reports, tabular CSV data, human-readable summary text files, and annotated output images.

---

## Key Features

- **Aspect-Ratio Preserved Resizing**: Downscales large images while preserving physical proportions.
- **Border Noise Elimination**: Erodes the leaf boundary prior to abnormality extraction, eliminating edge anti-aliasing false positives.
- **Shape & Geometric Filtering**: Discards thin vein lines and sparse noise using aspect ratio ($< 8.0$) and solidity ($\ge 0.25$) heuristics.
- **Auto-Fitting Visual HUD**: Renders dynamic annotated images with green leaf contours, semi-transparent red lesion overlays, bounding boxes, magenta centroids, and non-clipping text headers.
- **Healthy Leaf Support**: Handles clean healthy leaves gracefully, returning 0 regions, 0.0% affected percentage, and `LOW` severity without crashing.
- **Pipeline Step Exporter**: Saves 5 intermediate pipeline step images when `--save-steps` is enabled.

---

## Classical Computer Vision Pipeline

```text
+-------------------+      +-------------------+      +-------------------+
| 1. Input Image    | ---> | 2. Resize & HSV   | ---> | 3. Leaf Mask &    |
|    Validation     |      |    Conversion     |      |    Main Contour   |
+-------------------+      +-------------------+      +-------------------+
                                                                |
                                                                v
+-------------------+      +-------------------+      +-------------------+
| 6. Render HUD &   | <--- | 5. Centroids,     | <--- | 4. Eroded ROI &   |
|    Reports Exporter|      |    Area & Severity|      |    Abnormal Detection|
+-------------------+      +-------------------+      +-------------------+
```

---

## Technology Stack

- **Python**: 3.10+
- **OpenCV (`opencv-python`)**: Core image processing, HSV colorspace conversions, morphological operations, contour finding, spatial moments, and visual annotations.
- **NumPy**: Array manipulation, pixel masking, and boolean logic operations.
- **Pandas**: Structured CSV report formatting.
- **Pillow (PIL)**: Image IO fallback handler.
- **Pytest**: Automated unit and integration testing framework.

---

## Project Structure

```text
src/
  ├── __init__.py
  ├── main.py              # CLI entry point and pipeline orchestration
  ├── config.py            # Centralized threshold configurations and parameters
  ├── preprocessing.py     # Image loading, aspect-ratio resizing, BGR to HSV
  ├── segmentation.py      # Leaf HSV thresholding, cleaning, contour selection
  ├── morphology.py         # Morphological opening, closing, hole filling, area filtering
  ├── disease_detection.py # Abnormal region extraction inside leaf mask
  ├── analysis.py          # Component statistics, metrics calculation, severity grading
  ├── visualization.py     # Visual rendering, dynamic non-clipping HUD, step images
  └── utils.py             # File validation, path creation, JSON/CSV/TXT exporters

input/                 # Place input crop leaf images here
output/                # Destination for annotated output images and step images
reports/               # Destination for CSV, JSON, and TXT analysis reports
tests/                 # Pytest test suite (unit and integration tests)
docs/                  # Technical documentation (architecture, methodology, limitations)
screenshots/           # Visual previews and output screenshots
requirements.txt       # Project dependencies
```

---

## Installation

Ensure Python 3.10+ is installed on your system.

```bash
pip install -r requirements.txt
```

---

## Usage & CLI Options

Run CropVision using standard module execution:

```bash
python -m src.main [OPTIONS]
```

### Available Command-Line Arguments

| Flag | Long Argument | Type | Default | Description |
|---|---|---|---|---|
| `-i` | `--input` | `str` | `input/leaf.jpg` | Path to input crop leaf image file. |
| `-o` | `--output` | `str` | `output` | Directory path to save output annotated images. |
| `-r` | `--reports` | `str` | `reports` | Directory path to save CSV, JSON, and TXT reports. |
| `-s` | `--save-steps` | `flag` | `False` | Save intermediate pipeline step images for inspection. |
| | `--max-dim` | `int` | `1200` | Maximum width/height in pixels for aspect-ratio preserved resizing. |
| | `--min-area` | `int` | `100` | Minimum area threshold (in pixels) for an abnormal region. |
| `-h` | `--help` | `flag` | | Show help message and exit. |

---

## Example Commands

### 1. Basic Analysis Run

```bash
python -m src.main --input input/leaf.jpg --output output --reports reports --save-steps
```

### 2. Analysis of Diseased Leaf with Step Saving

```bash
python -m src.main --input input/leaf_diseased.jpeg --output output/diseased --reports reports/diseased --save-steps
```

### 3. Display CLI Help

```bash
python -m src.main --help
```

---

## Output Structure

When executed for an input image (e.g. `leaf_diseased.jpeg`), CropVision produces:

### 1. Reports

```text
reports/
└── <case>/
    └── <image_name>/
        ├── analysis.json
        ├── region_data.csv
        └── summary.txt
```

- **`analysis.json`**: Complete structured JSON containing image dimensions, leaf area, affected area, affected percentage, severity level, region count, threshold parameters used, disclaimer, and per-region details.
- **`region_data.csv`**: Tabular CSV file containing one row per detected region with columns: `region_id,area_px,centroid_x,centroid_y,bbox_x,bbox_y,bbox_w,bbox_h`.
- **`summary.txt`**: Human-readable text summary formatted with headers, severity breakdown, region coordinates, and disclaimer text.

### 2. Output Annotated Image (`output/<case>_analyzed.png`)
- Green leaf contour boundary.
- Semi-transparent red fill overlay on affected regions.
- Gold/Cyan bounding boxes and Magenta centroid markers.
- Region ID text labels (`#1`, `#2`, ...).
- Top HUD header showing severity level, affected percentage, and pixel totals.
- Bottom disclaimer banner.

### 3. Pipeline Step Images (`output/steps/<case>/`) *(Enabled with `--save-steps`)*
- `01_preprocessed.png`: Aspect-ratio resized image.
- `02_leaf_mask.png`: Binary segmented leaf mask.
- `03_affected_mask.png`: Binary affected region mask.
- `04_detected_regions.png`: Detected region contours & bounding boxes overlay.
- `05_final_annotated.png`: Final annotated output.

---

## Severity Heuristic Thresholds

CropVision classifies visual severity based on the percentage of leaf area occupied by visually abnormal regions:

| Severity Level | Affected Percentage Range |
|---|---|
| **`LOW`** | `< 5.0%` |
| **`MODERATE`** | `5.0% – < 15.0%` |
| **`HIGH`** | `15.0% – < 30.0%` |
| **`SEVERE`** | `>= 30.0%` |

> **Note**: Severity tiers are **project-defined visual heuristics** intended for image analysis stratification. They are **NOT official agricultural disease severity standards**.

---

## Limitations

- **Lighting & Exposure Sensitivity**: Classical HSV thresholding depends on consistent lighting. Extreme specular reflections or harsh shadows can affect segmentation.
- **Visual Anomaly Scope**: Detects visual discolorations (necrosis, chlorosis, lesions). Physical dirt, mechanical tears, or sunburn will also be detected as visual anomalies.
- **Calibrated Crop Scope**: Default HSV thresholds are calibrated for green crop leaves (such as tomato foliage).
- *For complete technical limitations, see [docs/limitations.md](docs/limitations.md).*

---

## Non-Diagnostic Disclaimer

> **Notice**: CropVision is a computer vision image analysis tool designed to detect visually abnormal surface regions and compute spatial area metrics. It does **not** provide agricultural, biological, or medical disease diagnosis.

---

## Testing

Run the automated Pytest suite:

```bash
python -m pytest -q
```

---

## Documentation Links

- [Architecture Guide](docs/architecture.md)
- [Computer Vision Methodology](docs/methodology.md)
- [System Limitations & Scope](docs/limitations.md)
