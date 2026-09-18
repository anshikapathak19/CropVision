"""
Main CLI entry point for CropVision headless Computer Vision pipeline.
Orchestrates image loading, segmentation, abnormal region analysis, visual rendering, and report generation.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from src.config import PipelineConfig, DEFAULT_CONFIG
from src.utils import (
    validate_image_path,
    ensure_directory,
    save_image,
    save_report_json,
    save_region_csv,
    save_summary_txt
)
from src.preprocessing import load_image, resize_image, convert_bgr_to_hsv
from src.segmentation import segment_leaf_hsv, clean_leaf_mask, select_main_leaf_contour
from src.disease_detection import detect_abnormal_regions, clean_abnormal_mask
from src.analysis import (
    extract_region_statistics,
    calculate_metrics,
    classify_severity,
    build_analysis_report
)
from src.visualization import draw_visualizations, create_step_visualizations

def parse_args(args: Optional[list] = None) -> argparse.Namespace:
    """
    Parses command-line arguments for the CropVision pipeline.
    
    Args:
        args: Optional list of command line strings (used for unit testing).
        
    Returns:
        Parsed argparse Namespace object.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.main",
        description="CropVision: Headless Computer Vision toolkit for leaf segmentation and visual abnormality detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "-i", "--input",
        type=str,
        default="input/leaf.jpg",
        help="Path to input crop leaf image file."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output",
        help="Directory path to save output annotated images."
    )
    parser.add_argument(
        "-r", "--reports",
        type=str,
        default="reports",
        help="Directory path to save CSV, JSON, and TXT analysis reports."
    )
    parser.add_argument(
        "-s", "--save-steps",
        action="store_true",
        help="Save intermediate pipeline step images for inspection."
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=DEFAULT_CONFIG.max_image_dim,
        help="Maximum width/height in pixels for aspect-ratio preserved resizing."
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=DEFAULT_CONFIG.min_abnormal_area,
        help="Minimum area threshold (in pixels) for an abnormal region."
    )
    
    return parser.parse_args(args)

def run_pipeline(
    input_path: str,
    output_dir: str,
    reports_dir: str,
    save_steps: bool = False,
    max_dim: int = DEFAULT_CONFIG.max_image_dim,
    min_abnormal_area: int = DEFAULT_CONFIG.min_abnormal_area
) -> int:
    """
    Executes the full computer vision analysis pipeline.
    
    Args:
        input_path: Path to input image.
        output_dir: Destination directory for annotated image.
        reports_dir: Destination directory for reports.
        save_steps: Whether to save intermediate step images.
        max_dim: Max dimension for image resizing.
        min_abnormal_area: Minimum pixel area threshold for abnormal regions.
        
    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    try:
        # 1. Validate input file
        try:
            valid_input_path = validate_image_path(input_path)
        except FileNotFoundError:
            print(f"[CropVision ERROR] Missing input file: '{input_path}'", file=sys.stderr)
            return 1
        except ValueError as ve:
            print(f"[CropVision ERROR] Invalid input image: {ve}", file=sys.stderr)
            return 1
            
        config = PipelineConfig(
            max_image_dim=max_dim,
            min_abnormal_area=min_abnormal_area
        )
        
        out_dir_path = ensure_directory(output_dir)
        rep_dir_path = ensure_directory(reports_dir)
        
        image_stem = valid_input_path.stem
        print(f"[CropVision] Processing image: '{valid_input_path.name}'...")
        
        # 2. Preprocess: Load, resize, BGR->HSV
        try:
            raw_bgr = load_image(valid_input_path)
        except Exception as load_err:
            print(f"[CropVision ERROR] Could not decode or read valid image: '{valid_input_path}' ({load_err})", file=sys.stderr)
            return 1
            
        resized_bgr, scale = resize_image(raw_bgr, max_dim=config.max_image_dim)
        h, w = resized_bgr.shape[:2]
        
        if scale < 1.0:
            print(f"[CropVision] Resized image from {raw_bgr.shape[1]}x{raw_bgr.shape[0]} to {w}x{h} (scale={scale:.3f})")
            
        hsv_image = convert_bgr_to_hsv(resized_bgr)
        
        # 3. Leaf Segmentation & Morphology
        raw_leaf_mask = segment_leaf_hsv(
            hsv_image,
            lower_hsv=config.leaf_hsv_lower,
            upper_hsv=config.leaf_hsv_upper
        )
        cleaned_leaf_mask = clean_leaf_mask(raw_leaf_mask, kernel_size=config.leaf_morph_kernel)
        leaf_mask, leaf_contour = select_main_leaf_contour(cleaned_leaf_mask, min_leaf_area=config.min_leaf_area)
        
        if leaf_contour is None:
            print(f"[CropVision WARNING] No leaf detected in image satisfying min_leaf_area ({config.min_leaf_area} px).")
            
        # 4. Abnormal Region Detection & Morphology Cleaning
        raw_abnormal_mask = detect_abnormal_regions(hsv_image, leaf_mask, config)
        cleaned_abnormal_mask = clean_abnormal_mask(raw_abnormal_mask, leaf_mask, config)
        
        regions = extract_region_statistics(
            cleaned_abnormal_mask,
            min_area=config.min_abnormal_area,
            max_aspect_ratio=config.max_aspect_ratio,
            min_solidity=config.min_solidity
        )
        metrics = calculate_metrics(leaf_mask, cleaned_abnormal_mask, regions)
        severity = classify_severity(metrics["affected_percentage"], config)
        
        if metrics["num_affected_regions"] == 0:
            print("[CropVision] No visually abnormal regions detected (Leaf is healthy).")
        else:
            print(f"[CropVision] Detected {metrics['num_affected_regions']} abnormal region(s) (Severity: {severity}).")
            
        # 5. Generate Analysis Report Dictionary
        report_data = build_analysis_report(
            image_filename=valid_input_path.name,
            image_dimensions=(w, h),
            metrics=metrics,
            severity=severity,
            regions=regions,
            config=config
        )
        
        # 6. Render Visualization
        annotated_image = draw_visualizations(
            image_bgr=resized_bgr,
            leaf_contour=leaf_contour,
            abnormal_mask=cleaned_abnormal_mask,
            regions=regions,
            metrics=metrics,
            severity=severity,
            config=config
        )
        
        # 7. Save Output Reports (analysis.json, region_data.csv, summary.txt)
        if rep_dir_path.name == image_stem:
            case_report_dir = rep_dir_path
        else:
            case_report_dir = ensure_directory(rep_dir_path / image_stem)
            
        json_path = case_report_dir / "analysis.json"
        save_report_json(report_data, json_path)
        print(f"[CropVision] Saved JSON report to: {json_path}")
        
        csv_path = case_report_dir / "region_data.csv"
        save_region_csv(report_data["regions"], csv_path)
        print(f"[CropVision] Saved CSV report to: {csv_path}")
        
        txt_path = case_report_dir / "summary.txt"
        save_summary_txt(report_data, txt_path)
        print(f"[CropVision] Saved Summary TXT to: {txt_path}")
        
        # Also maintain root-level convenience files for backward compatibility
        save_report_json(report_data, rep_dir_path / f"{image_stem}_report.json")
        save_region_csv(report_data["regions"], rep_dir_path / f"{image_stem}_report.csv")
        
        # Save Annotated Image
        annotated_path = out_dir_path / f"{image_stem}_analyzed.png"
        save_image(annotated_image, annotated_path)
        print(f"[CropVision] Saved annotated result image to: {annotated_path}")
        
        # Save step images if requested
        if save_steps:
            steps_dir = out_dir_path / "steps" / image_stem
            ensure_directory(steps_dir)
            step_images = create_step_visualizations(
                image_bgr=resized_bgr,
                leaf_mask=leaf_mask,
                abnormal_mask=cleaned_abnormal_mask,
                regions=regions,
                annotated_result=annotated_image,
                config=config
            )
            for step_filename, step_img in step_images.items():
                step_file_path = steps_dir / step_filename
                save_image(step_img, step_file_path)
            print(f"[CropVision] Saved {len(step_images)} intermediate step images to: {steps_dir}")
            
        # 8. Summary Log
        print("\n--- CropVision Analysis Summary ---")
        print(f" Image           : {valid_input_path.name}")
        print(f" Dimensions      : {w} x {h} px")
        print(f" Leaf Area       : {metrics['leaf_area_pixels']} px")
        print(f" Affected Area   : {metrics['affected_area_pixels']} px")
        print(f" Affected Pct    : {metrics['affected_percentage']:.2f}%")
        print(f" Severity Level  : {severity}")
        print(f" Abnormal Regions: {metrics['num_affected_regions']}")
        print(f" Disclaimer      : {report_data['disclaimer']}")
        print("------------------------------------\n")
        
        return 0

    except Exception as exc:
        print(f"[CropVision ERROR] Pipeline failed unexpected error: {exc}", file=sys.stderr)
        return 1

def main():
    args = parse_args()
    exit_code = run_pipeline(
        input_path=args.input,
        output_dir=args.output,
        reports_dir=args.reports,
        save_steps=args.save_steps,
        max_dim=args.max_dim,
        min_abnormal_area=args.min_area
    )
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
