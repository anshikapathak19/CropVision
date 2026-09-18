"""
Morphological operations for binary mask cleaning, hole filling, and noise suppression.
"""

from typing import Tuple
import cv2
import numpy as np

def get_kernel(kernel_size: Tuple[int, int]) -> np.ndarray:
    """Creates an elliptical morphological structuring element."""
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)

def apply_opening(mask: np.ndarray, kernel_size: Tuple[int, int] = (3, 3), iterations: int = 1) -> np.ndarray:
    """
    Applies morphological opening (erosion followed by dilation) to remove small noise objects.
    
    Args:
        mask: Binary mask array (0 or 255).
        kernel_size: (width, height) structuring element dimensions.
        iterations: Number of opening iterations.
        
    Returns:
        Cleaned binary mask array.
    """
    kernel = get_kernel(kernel_size)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)

def apply_closing(mask: np.ndarray, kernel_size: Tuple[int, int] = (3, 3), iterations: int = 1) -> np.ndarray:
    """
    Applies morphological closing (dilation followed by erosion) to close small gaps and holes.
    
    Args:
        mask: Binary mask array (0 or 255).
        kernel_size: (width, height) structuring element dimensions.
        iterations: Number of closing iterations.
        
    Returns:
        Cleaned binary mask array.
    """
    kernel = get_kernel(kernel_size)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)

def fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fills internal holes inside binary foreground regions.
    
    Args:
        mask: Binary mask array (0 or 255).
        
    Returns:
        Hole-filled binary mask array.
    """
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    if contours:
        cv2.drawContours(filled, contours, -1, color=255, thickness=cv2.FILLED)
    return filled

def remove_small_objects(mask: np.ndarray, min_area: int) -> np.ndarray:
    """
    Removes connected foreground components whose pixel area is smaller than min_area.
    
    Args:
        mask: Binary mask array (0 or 255).
        min_area: Minimum area in pixels required to retain a region.
        
    Returns:
        Filtered binary mask array.
    """
    if min_area <= 0:
        return mask.copy()
        
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    output_mask = np.zeros_like(mask)
    
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            output_mask[labels == label] = 255
            
    return output_mask
