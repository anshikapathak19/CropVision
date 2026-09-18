"""
Utility functions for file validation, directory management, and file persistence.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Union
import cv2
import numpy as np
import pandas as pd

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

def validate_image_path(image_path: Union[str, Path]) -> Path:
    """
    Validates that the given image file path exists, is a file, and has a supported image extension.
    
    Args:
        image_path: Path to input image file.
        
    Returns:
        Path object pointing to the validated image file.
        
    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If path is not a file or has unsupported extension.
    """
    path = Path(image_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input image path does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Input path is not a file: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format '{path.suffix}'. Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return path

def ensure_directory(dir_path: Union[str, Path]) -> Path:
    """
    Ensures that a directory exists, creating missing parent directories as needed.
    
    Args:
        dir_path: Directory path to create.
        
    Returns:
        Path object of the created/verified directory.
    """
    path = Path(dir_path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_image(image_bgr: np.ndarray, output_path: Union[str, Path]) -> Path:
    """
    Saves a BGR OpenCV image array to specified file path.
    
    Args:
        image_bgr: Image numpy array in BGR format.
        output_path: Destination file path.
        
    Returns:
        Path of the saved image file.
        
    Raises:
        IOError: If image writing fails using both OpenCV and Pillow.
    """
    path = Path(output_path).resolve()
    ensure_directory(path.parent)
    
    success = cv2.imwrite(str(path), image_bgr)
    if not success:
        try:
            from PIL import Image
            rgb_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)
            pil_img.save(str(path))
        except Exception as err:
            raise IOError(f"Failed to save image to target path '{path}': {err}")
            
    return path

def save_report_json(report_data: Dict[str, Any], json_path: Union[str, Path]) -> Path:
    """
    Saves report dictionary to a formatted JSON file.
    
    Args:
        report_data: Dictionary containing report metrics and details.
        json_path: Path to destination JSON file.
        
    Returns:
        Path to saved JSON file.
    """
    path = Path(json_path).resolve()
    ensure_directory(path.parent)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4)
    return path

def save_region_csv(regions_data: List[Dict[str, Any]], csv_path: Union[str, Path]) -> Path:
    """
    Saves region data to a CSV file with columns:
    region_id, area_px, centroid_x, centroid_y, bbox_x, bbox_y, bbox_w, bbox_h.
    
    Args:
        regions_data: List of dictionaries representing regions.
        csv_path: Path to destination CSV file.
        
    Returns:
        Path to saved CSV file.
    """
    path = Path(csv_path).resolve()
    ensure_directory(path.parent)
    
    columns = ["region_id", "area_px", "centroid_x", "centroid_y", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]
    
    if not regions_data:
        df = pd.DataFrame(columns=columns)
    else:
        rows = []
        for r in regions_data:
            bbox = r.get("bounding_box", {})
            rows.append({
                "region_id": r.get("region_id"),
                "area_px": r.get("area_px"),
                "centroid_x": r.get("centroid_x"),
                "centroid_y": r.get("centroid_y"),
                "bbox_x": bbox.get("x") if isinstance(bbox, dict) else bbox[0],
                "bbox_y": bbox.get("y") if isinstance(bbox, dict) else bbox[1],
                "bbox_w": bbox.get("width") if isinstance(bbox, dict) else bbox[2],
                "bbox_h": bbox.get("height") if isinstance(bbox, dict) else bbox[3],
            })
        df = pd.DataFrame(rows, columns=columns)
        
    df.to_csv(path, index=False)
    return path

def save_summary_txt(report_data: Dict[str, Any], txt_path: Union[str, Path]) -> Path:
    """
    Saves a human-readable summary report to a text file.
    
    Args:
        report_data: Dictionary containing report metrics and details.
        txt_path: Path to destination TXT file.
        
    Returns:
        Path to saved TXT file.
    """
    path = Path(txt_path).resolve()
    ensure_directory(path.parent)
    
    dims = report_data.get("image_dimensions", {})
    w = dims.get("width", 0)
    h = dims.get("height", 0)
    
    thresh = report_data.get("severity_thresholds_used", {})
    regions = report_data.get("regions", [])
    
    lines = [
        "=" * 80,
        "CROPVISION LEAF ANALYSIS SUMMARY REPORT",
        "=" * 80,
        f"Input Image            : {report_data.get('input_image', 'N/A')}",
        f"Image Dimensions       : {w} x {h} px",
        f"Leaf Area              : {report_data.get('leaf_area_pixels', 0):,} pixels",
        f"Affected Area          : {report_data.get('affected_area_pixels', 0):,} pixels",
        f"Affected Percentage    : {report_data.get('affected_percentage', 0.0):.2f}%",
        f"Severity Level         : {report_data.get('severity_level', 'UNKNOWN')}",
        f"Abnormal Regions Count : {report_data.get('num_abnormal_regions', 0)} region(s)",
        "",
        "SEVERITY THRESHOLDS USED:",
    ]
    
    for lvl, rule in thresh.items():
        lines.append(f"  - {lvl:<10} : {rule}")
        
    lines.extend([
        "",
        "DETECTED REGIONS BREAKDOWN:",
    ])
    
    if not regions:
        lines.append("  - No visually abnormal regions detected (Leaf tissue is clear/healthy).")
    else:
        for r in regions:
            rid = r.get("region_id")
            area = r.get("area_px")
            cx = r.get("centroid_x")
            cy = r.get("centroid_y")
            bbox = r.get("bounding_box", {})
            bx = bbox.get("x")
            by = bbox.get("y")
            bw = bbox.get("width")
            bh = bbox.get("height")
            lines.append(
                f"  - Region #{rid}: Area = {area} px, Centroid = ({cx}, {cy}), BBox = [x: {bx}, y: {by}, w: {bw}, h: {bh}]"
            )
            
    lines.extend([
        "",
        "DISCLAIMER:",
        report_data.get("disclaimer", ""),
        "=" * 80,
    ])
    
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
        
    return path
