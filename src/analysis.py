"""
Quantitative visual metrics calculation, connected component analysis, and severity classification.
"""

from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any
import cv2
import numpy as np

from src.config import (
    PipelineConfig,
    SEVERITY_LOW,
    SEVERITY_MODERATE,
    SEVERITY_HIGH,
    SEVERITY_SEVERE,
    DISCLAIMER_TEXT
)

@dataclass
class RegionDetail:
    """Quantitative stats for a single detected visually abnormal region."""
    region_id: int
    area_px: int
    centroid: Tuple[int, int]
    bounding_box: Tuple[int, int, int, int]  # (x, y, width, height)

def extract_region_statistics(
    abnormal_mask: np.ndarray,
    min_area: int = 100,
    max_aspect_ratio: float = 8.0,
    min_solidity: float = 0.25
) -> List[RegionDetail]:
    """
    Detects connected components/contours in abnormal mask and extracts geometric statistics.
    Filters out noise by area, extreme aspect ratio, and low solidity.
    
    Args:
        abnormal_mask: Cleaned binary abnormal mask.
        min_area: Minimum area in pixels to retain a region.
        max_aspect_ratio: Maximum allowed ratio of max(w,h)/min(w,h) to filter thin line noise.
        min_solidity: Minimum ratio of area / convex_hull_area to filter sparse contours.
        
    Returns:
        List of RegionDetail objects sorted by region area (descending).
    """
    contours, _ = cv2.findContours(abnormal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    regions: List[RegionDetail] = []
    idx = 1
    
    for cnt in contours:
        area = int(round(cv2.contourArea(cnt)))
        if area < min_area:
            continue
            
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Filter extreme aspect ratios (thin vein lines or border noise)
        aspect_ratio = max(w, h) / max(1.0, float(min(w, h)))
        if aspect_ratio > max_aspect_ratio:
            continue
            
        # Filter sparse fragmented noise
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = float(area) / float(hull_area)
            if solidity < min_solidity:
                continue
        
        # Calculate centroid using moments
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(round(M["m10"] / M["m00"]))
            cy = int(round(M["m01"] / M["m00"]))
        else:
            cx = x + w // 2
            cy = y + h // 2
            
        regions.append(RegionDetail(
            region_id=idx,
            area_px=area,
            centroid=(cx, cy),
            bounding_box=(x, y, w, h)
        ))
        idx += 1
        
    # Sort regions by area descending for consistent, readable numbering
    regions.sort(key=lambda r: r.area_px, reverse=True)
    # Re-assign IDs after sorting
    for i, reg in enumerate(regions, start=1):
        reg.region_id = i
        
    return regions

def calculate_metrics(
    leaf_mask: np.ndarray,
    abnormal_mask: np.ndarray,
    regions: List[RegionDetail]
) -> Dict[str, Any]:
    """
    Calculates overall leaf area, affected area, and percentage.
    
    Args:
        leaf_mask: Binary single-leaf mask.
        abnormal_mask: Cleaned binary abnormal mask.
        regions: List of detected RegionDetail objects.
        
    Returns:
        Dictionary of computed area metrics.
    """
    leaf_area_px = int(cv2.countNonZero(leaf_mask))
    affected_area_px = int(cv2.countNonZero(abnormal_mask))
    
    if leaf_area_px > 0:
        affected_percentage = float(round((affected_area_px / float(leaf_area_px)) * 100.0, 2))
    else:
        affected_percentage = 0.0
        
    return {
        "leaf_area_pixels": leaf_area_px,
        "affected_area_pixels": affected_area_px,
        "affected_percentage": affected_percentage,
        "num_affected_regions": len(regions)
    }

def classify_severity(affected_percentage: float, config: PipelineConfig) -> str:
    """
    Classifies visual severity level based on configurable project percentage thresholds.
    
    LOW: < 5%
    MODERATE: 5-15%
    HIGH: 15-30%
    SEVERE: >= 30%
    
    Args:
        affected_percentage: Percentage of leaf area that is abnormal.
        config: PipelineConfig instance with severity thresholds.
        
    Returns:
        Severity string (LOW, MODERATE, HIGH, SEVERE).
    """
    if affected_percentage < config.severity_low_thresh:
        return SEVERITY_LOW
    elif affected_percentage < config.severity_moderate_thresh:
        return SEVERITY_MODERATE
    elif affected_percentage < config.severity_high_thresh:
        return SEVERITY_HIGH
    else:
        return SEVERITY_SEVERE

def build_analysis_report(
    image_filename: str,
    image_dimensions: Tuple[int, int],
    metrics: Dict[str, Any],
    severity: str,
    regions: List[RegionDetail],
    config: PipelineConfig
) -> Dict[str, Any]:
    """
    Assembles a complete, structured analysis report dictionary.
    
    Args:
        image_filename: Source image file name.
        image_dimensions: (width, height) tuple of processed image.
        metrics: Dict returned from calculate_metrics.
        severity: Severity classification string.
        regions: List of RegionDetail objects.
        config: PipelineConfig used for analysis.
        
    Returns:
        Complete JSON-serializable report dictionary.
    """
    report = {
        "input_image": image_filename,
        "image_dimensions": {
            "width": image_dimensions[0],
            "height": image_dimensions[1]
        },
        "leaf_area_pixels": metrics["leaf_area_pixels"],
        "affected_area_pixels": metrics["affected_area_pixels"],
        "affected_percentage": metrics["affected_percentage"],
        "severity_level": severity,
        "num_abnormal_regions": metrics["num_affected_regions"],
        "severity_thresholds_used": {
            "LOW": f"< {config.severity_low_thresh}%",
            "MODERATE": f"{config.severity_low_thresh}% - {config.severity_moderate_thresh}%",
            "HIGH": f"{config.severity_moderate_thresh}% - {config.severity_high_thresh}%",
            "SEVERE": f">= {config.severity_high_thresh}%"
        },
        "disclaimer": DISCLAIMER_TEXT,
        "regions": [
            {
                "region_id": r.region_id,
                "area_px": r.area_px,
                "centroid_x": r.centroid[0],
                "centroid_y": r.centroid[1],
                "bounding_box": {
                    "x": r.bounding_box[0],
                    "y": r.bounding_box[1],
                    "width": r.bounding_box[2],
                    "height": r.bounding_box[3]
                }
            }
            for r in regions
        ]
    }
    return report
