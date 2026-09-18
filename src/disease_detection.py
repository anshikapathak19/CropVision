"""
Detection and morphological cleaning of visually abnormal regions inside the segmented leaf boundary.
"""

from typing import List, Tuple
import cv2
import numpy as np

from src.config import PipelineConfig
from src.morphology import apply_opening, apply_closing, remove_small_objects

def detect_abnormal_regions(
    hsv_image: np.ndarray,
    leaf_mask: np.ndarray,
    config: PipelineConfig
) -> np.ndarray:
    """
    Detects visually abnormal regions INSIDE the leaf mask using targeted classical CV color/intensity rules.
    Erodes the outer leaf boundary slightly to eliminate border artifacts and noise.
    
    Args:
        hsv_image: HSV format image array.
        leaf_mask: Binary single-leaf mask array.
        config: PipelineConfig instance with threshold parameters.
        
    Returns:
        Raw binary mask of detected abnormal regions.
    """
    if cv2.countNonZero(leaf_mask) == 0:
        return np.zeros_like(leaf_mask)
        
    # Erode leaf boundary slightly to avoid leaf edge anti-aliasing false positives
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.border_erosion_kernel)
    inner_leaf_mask = cv2.erode(leaf_mask, erosion_kernel, iterations=1)
    if cv2.countNonZero(inner_leaf_mask) == 0:
        inner_leaf_mask = leaf_mask  # Fallback for very small leaves
        
    # Rule 1: Specific abnormal HSV ranges (Reddish/Brown necrosis, Chlorotic yellow, Dark lesions)
    specific_abnormal_mask = np.zeros_like(leaf_mask)
    for lower_hsv, upper_hsv in config.specific_abnormal_hsv_ranges:
        lower_b = np.array(lower_hsv, dtype=np.uint8)
        upper_b = np.array(upper_hsv, dtype=np.uint8)
        range_mask = cv2.inRange(hsv_image, lower_b, upper_b)
        specific_abnormal_mask = cv2.bitwise_or(specific_abnormal_mask, range_mask)
        
    # Rule 2: Explicit non-green Hue anomalies inside inner leaf (Hue outside 22..95 with significant Saturation & Value)
    h_channel = hsv_image[:, :, 0]
    s_channel = hsv_image[:, :, 1]
    v_channel = hsv_image[:, :, 2]
    
    non_green_hue = ((h_channel < 22) | (h_channel > 95)) & (s_channel >= 30) & (v_channel >= 30)
    non_green_mask = np.zeros_like(leaf_mask)
    non_green_mask[non_green_hue] = 255
    
    # Combine candidate abnormal regions
    combined_raw = cv2.bitwise_or(specific_abnormal_mask, non_green_mask)
    
    # Restrict strictly to inner leaf region
    raw_abnormal_mask = cv2.bitwise_and(combined_raw, inner_leaf_mask)
    
    return raw_abnormal_mask

def clean_abnormal_mask(
    raw_abnormal_mask: np.ndarray,
    leaf_mask: np.ndarray,
    config: PipelineConfig
) -> np.ndarray:
    """
    Cleans the raw abnormal region mask using morphological opening, closing, and minimum area filtering.
    
    Args:
        raw_abnormal_mask: Initial binary mask of abnormal pixels.
        leaf_mask: Binary leaf mask to maintain boundary constraints.
        config: PipelineConfig instance with kernel sizes and min area settings.
        
    Returns:
        Cleaned binary mask of valid abnormal regions.
    """
    if cv2.countNonZero(raw_abnormal_mask) == 0:
        return np.zeros_like(raw_abnormal_mask)
        
    # Morphological opening to eliminate isolated high-frequency noise specks
    opened = apply_opening(raw_abnormal_mask, kernel_size=config.abnormal_morph_kernel, iterations=1)
    
    # Morphological closing to bridge close lesion clusters
    closed = apply_closing(opened, kernel_size=config.abnormal_morph_kernel, iterations=1)
    
    # Remove small objects below minimum area threshold
    area_filtered = remove_small_objects(closed, min_area=config.min_abnormal_area)
    
    # Final boundary check to keep 100% inside leaf mask
    cleaned_mask = cv2.bitwise_and(area_filtered, leaf_mask)
    
    return cleaned_mask
