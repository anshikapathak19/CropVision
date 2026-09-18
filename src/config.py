"""
Configuration settings and default thresholds for CropVision toolkit.
Centralized configuration to avoid magic numbers in pipeline logic.
"""

from dataclasses import dataclass, field
from typing import Tuple, List

# Severity levels
SEVERITY_LOW = "LOW"
SEVERITY_MODERATE = "MODERATE"
SEVERITY_HIGH = "HIGH"
SEVERITY_SEVERE = "SEVERE"

DISCLAIMER_TEXT = "Notice: Visually abnormal region analysis. Not an agricultural or medical diagnosis."

@dataclass
class PipelineConfig:
    """Configuration class for image processing, segmentation, and detection."""
    # Preprocessing
    max_image_dim: int = 1200
    
    # Leaf Segmentation (HSV)
    # Default range covers typical plant green hues (Hue: 25..95 in OpenCV 0..180 scale)
    leaf_hsv_lower: Tuple[int, int, int] = (25, 25, 25)
    leaf_hsv_upper: Tuple[int, int, int] = (95, 255, 255)
    
    # Healthy leaf HSV range (for identifying non-healthy tissue inside leaf)
    healthy_hsv_lower: Tuple[int, int, int] = (30, 35, 35)
    healthy_hsv_upper: Tuple[int, int, int] = (90, 255, 255)
    
    # Minimum area thresholds (pixels)
    min_leaf_area: int = 500
    min_abnormal_area: int = 100
    
    # Region shape & geometry filtering
    max_aspect_ratio: float = 8.0
    min_solidity: float = 0.25
    border_erosion_kernel: Tuple[int, int] = (5, 5)
    
    # Morphology kernels (width, height)
    leaf_morph_kernel: Tuple[int, int] = (7, 7)
    abnormal_morph_kernel: Tuple[int, int] = (5, 5)
    
    # Severity Thresholds (in percentages)
    severity_low_thresh: float = 5.0        # < 5.0%
    severity_moderate_thresh: float = 15.0   # 5.0% - 15.0%
    severity_high_thresh: float = 30.0       # 15.0% - 30.0%
    # SEVERE is >= 30.0%
    
    # Visualization Styling (BGR format for OpenCV)
    leaf_contour_color: Tuple[int, int, int] = (0, 220, 0)       # Vibrant Green
    leaf_contour_thickness: int = 2
    
    abnormal_contour_color: Tuple[int, int, int] = (0, 0, 255)   # Red
    abnormal_contour_thickness: int = 2
    abnormal_fill_color: Tuple[int, int, int] = (0, 0, 180)      # Dark Red overlay
    abnormal_overlay_alpha: float = 0.35
    
    bbox_color: Tuple[int, int, int] = (255, 215, 0)             # Gold / Cyan
    bbox_thickness: int = 1
    
    centroid_color: Tuple[int, int, int] = (255, 0, 255)         # Magenta
    centroid_radius: int = 3
    
    hud_bg_color: Tuple[int, int, int] = (20, 20, 20)           # Dark HUD
    hud_text_color: Tuple[int, int, int] = (255, 255, 255)       # White
    
    # Specific abnormal HSV ranges targeting distinct reddish/brown/yellow/dark lesions
    # [ (lower_hsv, upper_hsv), ... ]
    specific_abnormal_hsv_ranges: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = field(
        default_factory=lambda: [
            # Brown / Reddish / Necrotic Lesions (Range 1: Lower Red-Brown)
            ((0, 25, 20), (22, 255, 220)),
            # Brown / Reddish / Necrotic Lesions (Range 2: Upper Red-Purple)
            ((160, 25, 20), (180, 255, 220)),
            # Chlorotic / Yellow Spots (Requires strong saturation & brightness)
            ((15, 50, 80), (34, 255, 255)),
            # Dark / Black spots inside leaf
            ((0, 0, 0), (180, 255, 45)),
        ]
    )

# Default global instance
DEFAULT_CONFIG = PipelineConfig()
