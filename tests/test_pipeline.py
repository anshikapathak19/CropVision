"""
Integration tests for the CropVision pipeline, report generation, error handling, and CLI.
"""

from pathlib import Path
import pytest
import numpy as np
import cv2
import json

from src.main import run_pipeline, parse_args

def create_synthetic_leaf_image(path: Path) -> Path:
    """Helper to generate a realistic synthetic green leaf with brown lesions."""
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    # Draw an elliptical green leaf
    cv2.ellipse(img, (200, 200), (120, 70), 30, 0, 360, (35, 175, 45), -1)
    # Draw two brown lesion spots inside the leaf
    cv2.circle(img, (180, 180), 18, (20, 50, 150), -1)
    cv2.circle(img, (230, 210), 12, (15, 60, 130), -1)
    
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return path

def test_pipeline_execution(tmp_path):
    input_img_path = tmp_path / "synthetic_leaf.jpg"
    create_synthetic_leaf_image(input_img_path)
    
    out_dir = tmp_path / "output"
    reports_dir = tmp_path / "reports"
    
    exit_code = run_pipeline(
        input_path=str(input_img_path),
        output_dir=str(out_dir),
        reports_dir=str(reports_dir),
        save_steps=True
    )
    
    assert exit_code == 0
    
    # 1. Verify annotated image exists
    annotated_file = out_dir / "synthetic_leaf_analyzed.png"
    assert annotated_file.exists()
    
    # 2. Verify professional reports directory structure
    case_report_dir = reports_dir / "synthetic_leaf"
    assert case_report_dir.exists()
    
    json_file = case_report_dir / "analysis.json"
    csv_file = case_report_dir / "region_data.csv"
    txt_file = case_report_dir / "summary.txt"
    
    assert json_file.exists()
    assert csv_file.exists()
    assert txt_file.exists()
    
    # Validate JSON report keys
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["input_image"] == "synthetic_leaf.jpg"
    assert "image_dimensions" in data
    assert "leaf_area_pixels" in data
    assert "affected_area_pixels" in data
    assert "affected_percentage" in data
    assert "severity_level" in data
    assert "severity_thresholds_used" in data
    assert "disclaimer" in data
    assert "regions" in data
    
    # 3. Verify step images exist with proper names
    steps_dir = out_dir / "steps" / "synthetic_leaf"
    assert steps_dir.exists()
    
    expected_steps = [
        "01_preprocessed.png",
        "02_leaf_mask.png",
        "03_affected_mask.png",
        "04_detected_regions.png",
        "05_final_annotated.png"
    ]
    for step_file in expected_steps:
        assert (steps_dir / step_file).exists()

def test_missing_input_file_error(tmp_path):
    missing_path = tmp_path / "does_not_exist.jpg"
    exit_code = run_pipeline(
        input_path=str(missing_path),
        output_dir=str(tmp_path / "out"),
        reports_dir=str(tmp_path / "rep")
    )
    assert exit_code == 1

def test_invalid_image_file_error(tmp_path):
    invalid_path = tmp_path / "fake.jpg"
    with open(invalid_path, "w") as f:
        f.write("not an image file")
        
    exit_code = run_pipeline(
        input_path=str(invalid_path),
        output_dir=str(tmp_path / "out"),
        reports_dir=str(tmp_path / "rep")
    )
    assert exit_code == 1

def test_cli_parse_args():
    parsed = parse_args(["--input", "input/sample.jpg", "--output", "custom_out", "--reports", "custom_rep", "--save-steps"])
    assert parsed.input == "input/sample.jpg"
    assert parsed.output == "custom_out"
    assert parsed.reports == "custom_rep"
    assert parsed.save_steps is True
