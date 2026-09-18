"""
Visualization module for rendering annotated output images, HUD banners, and step-by-step pipeline images.
"""

from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np

from src.config import (
    PipelineConfig,
    DISCLAIMER_TEXT,
    SEVERITY_LOW,
    SEVERITY_MODERATE,
    SEVERITY_HIGH,
    SEVERITY_SEVERE
)
from src.analysis import RegionDetail

def get_severity_color(severity: str) -> Tuple[int, int, int]:
    """Returns HUD text color (BGR) corresponding to severity grade."""
    if severity == SEVERITY_LOW:
        return (0, 230, 0)       # Green
    elif severity == SEVERITY_MODERATE:
        return (0, 215, 255)     # Yellow
    elif severity == SEVERITY_HIGH:
        return (0, 140, 255)     # Orange
    else:
        return (0, 0, 255)       # Red

def fit_text_to_width(
    text: str,
    font: int,
    initial_scale: float,
    thickness: int,
    max_width: int
) -> float:
    """
    Dynamically scales down text font size until it fits within max_width pixels.
    
    Args:
        text: Text string to measure.
        font: OpenCV font type.
        initial_scale: Starting font scale.
        thickness: Text line thickness.
        max_width: Maximum allowed width in pixels.
        
    Returns:
        Adjusted font scale factor.
    """
    scale = initial_scale
    while scale > 0.15:
        (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
        if tw <= max_width:
            break
        scale -= 0.02
    return max(0.15, scale)

def draw_visualizations(
    image_bgr: np.ndarray,
    leaf_contour: Optional[np.ndarray],
    abnormal_mask: np.ndarray,
    regions: List[RegionDetail],
    metrics: Dict[str, Any],
    severity: str,
    config: PipelineConfig
) -> np.ndarray:
    """
    Renders an annotated result image displaying leaf boundary, affected regions, bounding boxes,
    centroids, region IDs, and a non-clipping HUD header/footer.
    
    Args:
        image_bgr: Resized base image in BGR format.
        leaf_contour: Main leaf contour numpy array.
        abnormal_mask: Cleaned binary abnormal region mask.
        regions: List of RegionDetail objects.
        metrics: Computed metrics dictionary.
        severity: Classified severity level string.
        config: PipelineConfig instance with style options.
        
    Returns:
        Annotated output BGR image array.
    """
    canvas = image_bgr.copy()
    h, w = canvas.shape[:2]
    
    # 1. Highlight affected regions with semi-transparent overlay
    if cv2.countNonZero(abnormal_mask) > 0:
        overlay = canvas.copy()
        overlay[abnormal_mask > 0] = config.abnormal_fill_color
        cv2.addWeighted(overlay, config.abnormal_overlay_alpha, canvas, 1.0 - config.abnormal_overlay_alpha, 0, canvas)
        
    # 2. Draw green leaf boundary contour
    if leaf_contour is not None:
        cv2.drawContours(canvas, [leaf_contour], -1, config.leaf_contour_color, config.leaf_contour_thickness, cv2.LINE_AA)
        
    # 3. Draw abnormal region outlines, bounding boxes, centroids, and IDs
    abnormal_contours, _ = cv2.findContours(abnormal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if abnormal_contours:
        cv2.drawContours(canvas, abnormal_contours, -1, config.abnormal_contour_color, config.abnormal_contour_thickness, cv2.LINE_AA)
        
    for r in regions:
        rx, ry, rw, rh = r.bounding_box
        cx, cy = r.centroid
        
        # Bounding box
        cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), config.bbox_color, config.bbox_thickness, cv2.LINE_AA)
        
        # Centroid marker
        cv2.circle(canvas, (cx, cy), config.centroid_radius, config.centroid_color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), config.centroid_radius + 2, (0, 0, 0), 1, cv2.LINE_AA)
        
        # Region ID tag label
        tag_text = f"#{r.region_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_thick = 1
        (tw, th), baseline = cv2.getTextSize(tag_text, font, font_scale, font_thick)
        
        tag_x = max(0, min(w - tw - 4, rx))
        tag_y = max(th + 4, ry - 4)
        
        cv2.rectangle(canvas, (tag_x, tag_y - th - 2), (tag_x + tw + 4, tag_y + baseline), (0, 0, 0), -1)
        cv2.putText(canvas, tag_text, (tag_x + 2, tag_y), font, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)
        
    # 4. Top HUD Header Banner with Auto-Fitting Text
    max_text_width = max(100, w - 24)
    font = cv2.FONT_HERSHEY_SIMPLEX
    sev_color = get_severity_color(severity)
    
    line1 = f"CropVision Analysis | SEVERITY: {severity}"
    line2 = f"Affected: {metrics['affected_percentage']:.2f}% | Leaf: {metrics['leaf_area_pixels']:,} px | Regions: {metrics['num_affected_regions']}"
    
    scale1 = fit_text_to_width(line1, font, 0.55, 2, max_text_width)
    scale2 = fit_text_to_width(line2, font, 0.45, 1, max_text_width)
    
    hud_height = max(58, int(h * 0.11))
    hud_bg = np.zeros((hud_height, w, 3), dtype=np.uint8)
    hud_bg[:] = config.hud_bg_color
    
    canvas[0:hud_height, 0:w] = cv2.addWeighted(canvas[0:hud_height, 0:w], 0.2, hud_bg, 0.8, 0)
    
    cv2.putText(canvas, line1, (12, max(20, int(hud_height * 0.4))), font, scale1, sev_color, 2, cv2.LINE_AA)
    cv2.putText(canvas, line2, (12, max(40, int(hud_height * 0.78))), font, scale2, (220, 220, 220), 1, cv2.LINE_AA)
    
    # 5. Bottom Disclaimer Banner with Guaranteed Visibility
    footer_height = max(26, int(h * 0.05))
    footer_y = h - footer_height
    if footer_y > hud_height:
        footer_bg = np.zeros((footer_height, w, 3), dtype=np.uint8)
        canvas[footer_y:h, 0:w] = cv2.addWeighted(canvas[footer_y:h, 0:w], 0.25, footer_bg, 0.75, 0)
        
        disc_scale = fit_text_to_width(DISCLAIMER_TEXT, font, 0.38, 1, max_text_width)
        (dtw, dth), _ = cv2.getTextSize(DISCLAIMER_TEXT, font, disc_scale, 1)
        disc_y = footer_y + int((footer_height + dth) / 2) - 2
        
        cv2.putText(canvas, DISCLAIMER_TEXT, (10, disc_y), font, disc_scale, (200, 200, 200), 1, cv2.LINE_AA)
        
    return canvas

def create_step_visualizations(
    image_bgr: np.ndarray,
    leaf_mask: np.ndarray,
    abnormal_mask: np.ndarray,
    regions: List[RegionDetail],
    annotated_result: np.ndarray,
    config: PipelineConfig
) -> Dict[str, np.ndarray]:
    """
    Creates a dictionary of intermediate pipeline step images:
    - 01_preprocessed.png
    - 02_leaf_mask.png
    - 03_affected_mask.png
    - 04_detected_regions.png
    - 05_final_annotated.png
    
    Args:
        image_bgr: Resized base image in BGR format.
        leaf_mask: Cleaned binary leaf mask array.
        abnormal_mask: Cleaned binary abnormal mask array.
        regions: List of detected RegionDetail objects.
        annotated_result: Final annotated result image array.
        config: PipelineConfig instance.
        
    Returns:
        Dictionary mapping step image filenames to BGR image arrays.
    """
    steps = {}
    
    # 01 Preprocessed Image
    steps["01_preprocessed.png"] = image_bgr.copy()
    
    # 02 Leaf Mask
    steps["02_leaf_mask.png"] = cv2.cvtColor(leaf_mask, cv2.COLOR_GRAY2BGR)
    
    # 03 Affected-Region Mask
    steps["03_affected_mask.png"] = cv2.cvtColor(abnormal_mask, cv2.COLOR_GRAY2BGR)
    
    # 04 Detected Regions (Contours + BBoxes overlay on preprocessed image)
    regions_img = image_bgr.copy()
    abnormal_contours, _ = cv2.findContours(abnormal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if abnormal_contours:
        cv2.drawContours(regions_img, abnormal_contours, -1, config.abnormal_contour_color, config.abnormal_contour_thickness, cv2.LINE_AA)
    for r in regions:
        rx, ry, rw, rh = r.bounding_box
        cv2.rectangle(regions_img, (rx, ry), (rx + rw, ry + rh), config.bbox_color, config.bbox_thickness, cv2.LINE_AA)
    steps["04_detected_regions.png"] = regions_img
    
    # 05 Final Annotated Result
    steps["05_final_annotated.png"] = annotated_result.copy()
    
    return steps
