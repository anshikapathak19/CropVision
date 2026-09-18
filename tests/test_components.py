"""
Unit tests for CropVision modules: config, utils, preprocessing, segmentation, morphology, disease_detection, and analysis.
"""

import pytest
import numpy as np
from pathlib import Path
import cv2

from src.config import PipelineConfig, SEVERITY_LOW, SEVERITY_MODERATE, SEVERITY_HIGH, SEVERITY_SEVERE
from src.utils import validate_image_path, ensure_directory, save_image
from src.preprocessing import resize_image, convert_bgr_to_hsv
from src.segmentation import segment_leaf_hsv, clean_leaf_mask, select_main_leaf_contour
from src.morphology import apply_opening, apply_closing, fill_holes, remove_small_objects
from src.disease_detection import detect_abnormal_regions, clean_abnormal_mask
from src.analysis import (
    extract_region_statistics,
    calculate_metrics,
    classify_severity,
    build_analysis_report
)

def test_config_defaults():
    config = PipelineConfig()
    assert config.max_image_dim == 1200
    assert config.min_leaf_area == 500
    assert config.min_abnormal_area == 100
    assert config.severity_low_thresh == 5.0

def test_utils_directory(tmp_path):
    target_dir = tmp_path / "sub" / "dir"
    created = ensure_directory(target_dir)
    assert created.exists()
    assert created.is_dir()

def test_preprocessing_resize():
    # Create large image 2000x1000
    img = np.zeros((1000, 2000, 3), dtype=np.uint8)
    resized, scale = resize_image(img, max_dim=1000)
    assert scale == 0.5
    assert resized.shape == (500, 1000, 3)

    # Test small image
    small_img = np.zeros((100, 200, 3), dtype=np.uint8)
    resized_small, scale_small = resize_image(small_img, max_dim=1000)
    assert scale_small == 1.0
    assert resized_small.shape == (100, 200, 3)

def test_segmentation_and_contour():
    config = PipelineConfig()
    # Create synthetic leaf image (green rectangle on black background)
    img_bgr = np.zeros((300, 400, 3), dtype=np.uint8)
    img_bgr[50:250, 50:350] = (30, 180, 40) # Vibrant green BGR
    
    hsv = convert_bgr_to_hsv(img_bgr)
    raw_mask = segment_leaf_hsv(hsv, config.leaf_hsv_lower, config.leaf_hsv_upper)
    cleaned_mask = clean_leaf_mask(raw_mask)
    leaf_mask, contour = select_main_leaf_contour(cleaned_mask, min_leaf_area=100)
    
    assert cv2.countNonZero(leaf_mask) > 0
    assert contour is not None
    assert cv2.contourArea(contour) >= 100

def test_abnormal_detection():
    config = PipelineConfig()
    # Create synthetic leaf with brown spot inside
    img_bgr = np.zeros((300, 400, 3), dtype=np.uint8)
    img_bgr[50:250, 50:350] = (30, 180, 40) # Green leaf
    img_bgr[100:150, 100:150] = (20, 40, 140) # Brown spot BGR
    
    hsv = convert_bgr_to_hsv(img_bgr)
    raw_leaf_mask = segment_leaf_hsv(hsv, config.leaf_hsv_lower, config.leaf_hsv_upper)
    cleaned_leaf_mask = clean_leaf_mask(raw_leaf_mask)
    leaf_mask, contour = select_main_leaf_contour(cleaned_leaf_mask, min_leaf_area=100)
    
    raw_abnormal = detect_abnormal_regions(hsv, leaf_mask, config)
    cleaned_abnormal = clean_abnormal_mask(raw_abnormal, leaf_mask, config)
    
    regions = extract_region_statistics(cleaned_abnormal, min_area=config.min_abnormal_area)
    metrics = calculate_metrics(leaf_mask, cleaned_abnormal, regions)
    
    assert metrics["num_affected_regions"] >= 1
    assert metrics["affected_area_pixels"] > 0
    assert metrics["affected_percentage"] > 0.0

def test_healthy_leaf_zero_regions():
    config = PipelineConfig()
    # Create a 100% healthy green leaf image
    img_bgr = np.zeros((300, 400, 3), dtype=np.uint8)
    img_bgr[50:250, 50:350] = (35, 175, 45) # Pure healthy green BGR
    
    hsv = convert_bgr_to_hsv(img_bgr)
    raw_leaf_mask = segment_leaf_hsv(hsv, config.leaf_hsv_lower, config.leaf_hsv_upper)
    cleaned_leaf_mask = clean_leaf_mask(raw_leaf_mask)
    leaf_mask, contour = select_main_leaf_contour(cleaned_leaf_mask, min_leaf_area=100)
    
    raw_abnormal = detect_abnormal_regions(hsv, leaf_mask, config)
    cleaned_abnormal = clean_abnormal_mask(raw_abnormal, leaf_mask, config)
    
    regions = extract_region_statistics(cleaned_abnormal, min_area=config.min_abnormal_area)
    metrics = calculate_metrics(leaf_mask, cleaned_abnormal, regions)
    severity = classify_severity(metrics["affected_percentage"], config)
    
    assert metrics["num_affected_regions"] == 0
    assert metrics["affected_area_pixels"] == 0
    assert metrics["affected_percentage"] == 0.0
    assert severity == SEVERITY_LOW

def test_noise_filtering():
    config = PipelineConfig()
    # Create a leaf image with tiny noise specks (1-2 pixels) and 1 legitimate large brown lesion (25x25 pixels)
    img_bgr = np.zeros((300, 400, 3), dtype=np.uint8)
    img_bgr[50:250, 50:350] = (35, 175, 45) # Healthy green leaf
    
    # Tiny noise specks (should be filtered out by morphology opening and min_area)
    img_bgr[60, 60] = (10, 20, 100)
    img_bgr[70, 80] = (200, 10, 10)
    img_bgr[90:92, 90:92] = (15, 30, 120)
    
    # Large real lesion (25x25 = 625 pixels)
    img_bgr[120:145, 120:145] = (20, 45, 150)
    
    hsv = convert_bgr_to_hsv(img_bgr)
    raw_leaf_mask = segment_leaf_hsv(hsv, config.leaf_hsv_lower, config.leaf_hsv_upper)
    cleaned_leaf_mask = clean_leaf_mask(raw_leaf_mask)
    leaf_mask, contour = select_main_leaf_contour(cleaned_leaf_mask, min_leaf_area=100)
    
    raw_abnormal = detect_abnormal_regions(hsv, leaf_mask, config)
    cleaned_abnormal = clean_abnormal_mask(raw_abnormal, leaf_mask, config)
    
    regions = extract_region_statistics(
        cleaned_abnormal,
        min_area=config.min_abnormal_area,
        max_aspect_ratio=config.max_aspect_ratio,
        min_solidity=config.min_solidity
    )
    
    # Only the 1 real large lesion should survive
    assert len(regions) == 1
    assert regions[0].area_px >= 500

def test_centroid_calculation():
    # Create a 300x300 binary mask with a single 40x40 square centered at (100..140, 100..140)
    mask = np.zeros((300, 300), dtype=np.uint8)
    mask[100:140, 100:140] = 255
    
    regions = extract_region_statistics(mask, min_area=50)
    assert len(regions) == 1
    reg = regions[0]
    # Expected centroid is center of (100..139) -> cx=119 or 120, cy=119 or 120
    assert abs(reg.centroid[0] - 119) <= 1
    assert abs(reg.centroid[1] - 119) <= 1
    assert reg.bounding_box == (100, 100, 40, 40)

def test_affected_percentage_calculation():
    leaf_mask = np.zeros((200, 200), dtype=np.uint8)
    leaf_mask[50:150, 50:150] = 255 # 100x100 = 10,000 px
    
    abnormal_mask = np.zeros((200, 200), dtype=np.uint8)
    abnormal_mask[60:100, 60:100] = 255 # 40x40 = 1,600 px
    
    regions = extract_region_statistics(abnormal_mask, min_area=50)
    metrics = calculate_metrics(leaf_mask, abnormal_mask, regions)
    
    assert metrics["leaf_area_pixels"] == 10000
    assert metrics["affected_area_pixels"] == 1600
    assert metrics["affected_percentage"] == 16.0
    assert metrics["num_affected_regions"] == 1

def test_min_region_area_filtering():
    mask = np.zeros((300, 300), dtype=np.uint8)
    # Small 5x5 region = 25 px
    mask[10:15, 10:15] = 255
    # Medium 12x12 region = 144 px (contourArea = 121)
    mask[50:62, 50:62] = 255
    # Large 20x20 region = 400 px (contourArea = 361)
    mask[100:120, 100:120] = 255
    
    regions = extract_region_statistics(mask, min_area=100)
    # 25 px region should be discarded
    assert len(regions) == 2
    areas = [r.area_px for r in regions]
    assert max(areas) >= 350
    assert 100 <= min(areas) <= 150

def test_severity_tier_boundaries():
    config = PipelineConfig()
    assert classify_severity(0.0, config) == SEVERITY_LOW
    assert classify_severity(4.99, config) == SEVERITY_LOW
    assert classify_severity(5.00, config) == SEVERITY_MODERATE
    assert classify_severity(14.99, config) == SEVERITY_MODERATE
    assert classify_severity(15.00, config) == SEVERITY_HIGH
    assert classify_severity(29.99, config) == SEVERITY_HIGH
    assert classify_severity(30.00, config) == SEVERITY_SEVERE
    assert classify_severity(65.0, config) == SEVERITY_SEVERE

def test_morphology_operations():
    # Test hole filling
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(mask, (20, 20), (80, 80), 255, 2) # Empty hollow box
    filled = fill_holes(mask)
    assert cv2.countNonZero(filled) == 3965
    
    # Test small object removal
    mask_small = np.zeros((100, 100), dtype=np.uint8)
    mask_small[10:13, 10:13] = 255 # 9 px
    filtered = remove_small_objects(mask_small, min_area=20)
    assert cv2.countNonZero(filtered) == 0
