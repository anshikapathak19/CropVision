"""
Leaf segmentation routines using configurable HSV thresholding, morphological refinement, and main contour selection.
"""

from typing import Tuple, Optional
import cv2
import numpy as np

from src.morphology import apply_opening, apply_closing, fill_holes

def segment_leaf_hsv(
    hsv_image: np.ndarray,
    lower_hsv: Tuple[int, int, int],
    upper_hsv: Tuple[int, int, int]
) -> np.ndarray:
    """
    Segments candidate leaf pixels using configurable HSV color range thresholding.
    
    Args:
        hsv_image: HSV format image array.
        lower_hsv: (Hue, Saturation, Value) lower threshold tuple.
        upper_hsv: (Hue, Saturation, Value) upper threshold tuple.
        
    Returns:
        Binary threshold mask (255 for leaf pixels, 0 for background).
    """
    lower_bound = np.array(lower_hsv, dtype=np.uint8)
    upper_bound = np.array(upper_hsv, dtype=np.uint8)
    return cv2.inRange(hsv_image, lower_bound, upper_bound)

def clean_leaf_mask(raw_mask: np.ndarray, kernel_size: Tuple[int, int] = (7, 7)) -> np.ndarray:
    """
    Cleans the raw leaf binary mask using opening, closing, and hole filling.
    
    Args:
        raw_mask: Raw binary leaf mask.
        kernel_size: Morphological kernel size tuple.
        
    Returns:
        Cleaned binary leaf mask.
    """
    # Remove small external background noise
    opened = apply_opening(raw_mask, kernel_size=kernel_size, iterations=1)
    # Bridge gaps inside leaf body
    closed = apply_closing(opened, kernel_size=kernel_size, iterations=1)
    # Fill interior leaf holes
    filled = fill_holes(closed)
    return filled

def select_main_leaf_contour(
    mask: np.ndarray,
    min_leaf_area: int = 500
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Selects the primary leaf contour based on maximum area and creates a clean single-leaf mask.
    
    Args:
        mask: Cleaned binary leaf mask.
        min_leaf_area: Minimum required area in pixels for valid leaf selection.
        
    Returns:
        Tuple of (single_leaf_mask, leaf_contour).
        If no contour satisfies min_leaf_area, returns (empty_mask, None).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    leaf_mask = np.zeros_like(mask)
    if not contours:
        return leaf_mask, None
        
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    
    if area < min_leaf_area:
        return leaf_mask, None
        
    # Draw the largest contour filled to produce a smooth, solid leaf mask
    cv2.drawContours(leaf_mask, [largest_contour], -1, color=255, thickness=cv2.FILLED)
    
    return leaf_mask, largest_contour
