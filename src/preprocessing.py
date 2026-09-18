"""
Image loading, aspect-ratio preserving resizing, and color space conversion routines.
"""

from pathlib import Path
from typing import Tuple, Union
import cv2
import numpy as np

from src.utils import validate_image_path

def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    Validates and loads an image from file path into BGR format array.
    
    Args:
        image_path: Path to input image file.
        
    Returns:
        Loaded image array in BGR format.
        
    Raises:
        ValueError: If OpenCV fails to decode image file.
    """
    validated_path = validate_image_path(image_path)
    image_bgr = cv2.imread(str(validated_path))
    
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError(f"OpenCV failed to read/decode image file: {validated_path}")
        
    return image_bgr

def resize_image(image: np.ndarray, max_dim: int = 1200) -> Tuple[np.ndarray, float]:
    """
    Resizes image while strictly preserving its aspect ratio if its max dimension exceeds max_dim.
    
    Args:
        image: Original input image array.
        max_dim: Maximum allowed width or height in pixels.
        
    Returns:
        Tuple of (resized_image, scale_factor). Scale factor is 1.0 if no resizing occurred.
    """
    h, w = image.shape[:2]
    max_current = max(h, w)
    
    if max_current <= max_dim or max_dim <= 0:
        return image.copy(), 1.0
        
    scale = max_dim / float(max_current)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized_image, scale

def convert_bgr_to_hsv(image_bgr: np.ndarray) -> np.ndarray:
    """
    Converts BGR image array to HSV color space.
    
    Args:
        image_bgr: Input BGR image array.
        
    Returns:
        HSV image array (Hue: 0..180, Saturation: 0..255, Value: 0..255).
    """
    if image_bgr is None or len(image_bgr.shape) != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Input image must be a 3-channel BGR numpy array.")
        
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
