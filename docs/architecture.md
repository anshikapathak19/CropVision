# CropVision System Architecture

This document describes the high-level architecture, module responsibilities, and data flow of the **CropVision** computer vision toolkit.

---

## High-Level Architecture Overview

CropVision is structured as a modular, headless Python pipeline utilizing classical OpenCV image processing operations. It processes input crop leaf images through discrete sequential stages: validation, preprocessing, leaf segmentation, abnormal region isolation, component statistics extraction, severity grading, report generation, and visualization rendering.

```text
+-----------------------+
|  Input Image File     |
| (JPEG / PNG / TIFF)   |
+-----------+-----------+
            |
            v
+-----------------------+
|  src/utils.py         | ---> Validate path & format
+-----------+-----------+
            |
            v
+-----------------------+
|  src/preprocessing.py | ---> Aspect-ratio preserved resize & BGR-to-HSV conversion
+-----------+-----------+
            |
            v
+-----------------------+
|  src/segmentation.py  | ---> HSV thresholding, morphology & main leaf contour selection
+-----------+-----------+
            |
            v
+-----------------------+
|src/disease_detection.py| ---> Border-eroded ROI & abnormal color/intensity rule extraction
+-----------+-----------+
            |
            v
+-----------------------+
|  src/analysis.py      | ---> Connected components, area px, centroid, percentage & severity
+-----------+-----------+
            |
            +---------------------------------+
            |                                 |
            v                                 v
+-----------------------+         +-----------------------+
| src/visualization.py  |         |  src/utils.py         |
+-----------+-----------+         +-----------+-----------+
            |                                 |
            v                                 v
+-----------------------+         +-----------------------+
| Output Annotated Image|         | Professional Reports  |
| output/<case>_analyzed|         | analysis.json         |
| output/steps/<case>/  |         | region_data.csv       |
+-----------------------+         | summary.txt           |
                                  +-----------------------+
```

---

## Module Responsibilities

| Module | Core Responsibility | Key Functions |
|---|---|---|
| **`src/config.py`** | Centralized dataclass configuration for HSV thresholds, morphology kernel sizes, minimum region areas, aspect ratio/solidity limits, severity bounds, and styling constants. | `PipelineConfig`, `DEFAULT_CONFIG` |
| **`src/utils.py`** | IO utilities, file path validation, directory creation, and structured exporter routines. | `validate_image_path`, `ensure_directory`, `save_image`, `save_report_json`, `save_region_csv`, `save_summary_txt` |
| **`src/preprocessing.py`** | Image loading, aspect-ratio preserved downscaling, and HSV colorspace conversion. | `load_image`, `resize_image`, `convert_bgr_to_hsv` |
| **`src/segmentation.py`** | Leaf boundary segmentation using color range thresholding, morphological cleaning, and primary contour isolation. | `segment_leaf_hsv`, `clean_leaf_mask`, `select_main_leaf_contour` |
| **`src/morphology.py`** | Low-level morphological primitives for noise elimination, gap closing, hole filling, and area filtering. | `apply_opening`, `apply_closing`, `fill_holes`, `remove_small_objects` |
| **`src/disease_detection.py`**| Extraction of visually abnormal non-healthy / necrotic / chlorotic / dark pixels strictly inside the eroded leaf ROI. | `detect_abnormal_regions`, `clean_abnormal_mask` |
| **`src/analysis.py`** | Quantitative region metric calculation, connected component moments/centroids, area percentages, and severity classification. | `extract_region_statistics`, `calculate_metrics`, `classify_severity`, `build_analysis_report` |
| **`src/visualization.py`**| Rendering leaf boundaries, semi-transparent lesion overlays, bounding boxes, centroid markers, region ID tags, auto-fitted HUD banners, and intermediate step images. | `fit_text_to_width`, `draw_visualizations`, `create_step_visualizations` |
| **`src/main.py`** | Orchestration CLI entry point with `argparse`, progress logging, and error handling. | `parse_args`, `run_pipeline`, `main` |

---

## Data Flow Summary

1. **Input Phase**: `src/main.py` parses command-line arguments and validates input path via `src/utils.py`.
2. **Preprocessing Phase**: Image is loaded and downscaled if `max(width, height) > max_image_dim` (default 1200px) while maintaining physical proportions. Image is converted from BGR to HSV.
3. **Segmentation Phase**: Leaf pixels are thresholded in HSV space (`25..95` Hue). Morphological opening, closing, and hole-filling produce a clean binary mask. `select_main_leaf_contour` picks the largest connected contour ≥ `min_leaf_area`.
4. **Abnormality Detection Phase**: The leaf mask is eroded slightly to form an inner ROI (eliminating border artifacts). Candidate abnormal pixels (brown necrosis, chlorotic yellow, dark spots, non-green hue anomalies) are isolated and morphologically cleaned.
5. **Analysis Phase**: `cv2.findContours` extracts candidate abnormal regions. Regions are filtered by `min_abnormal_area` (100 px), `max_aspect_ratio` (8.0), and `min_solidity` (0.25). Centroids $(c_x, c_y)$ are computed via spatial moments.
6. **Reporting & Visualization Phase**: Annotated BGR image is rendered with auto-scaled text banners. Structured reports (`analysis.json`, `region_data.csv`, `summary.txt`) and step images are exported to target output directories.
