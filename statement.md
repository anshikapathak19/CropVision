# CropVision — Project Statement

## Problem Statement

Manual inspection of crop leaves for visible damage and discoloration can be subjective, time-consuming, and difficult to quantify consistently. There is a need for a lightweight and interpretable computer vision system that can analyze leaf images, identify visually abnormal regions, and provide measurable spatial information without depending on deep learning models or cloud infrastructure.

CropVision addresses this problem using classical computer vision techniques to segment a crop leaf from its background, detect visually abnormal regions within the leaf, calculate affected area, and classify the observed visual severity using project-defined heuristic thresholds.

## Project Scope

CropVision is a command-line, headless computer vision toolkit focused on image-based analysis of crop leaves.

The project covers:

- Input image validation and preprocessing.
- Aspect-ratio-preserving image resizing.
- HSV-based leaf segmentation.
- Morphological noise removal and mask refinement.
- Detection of the primary leaf region.
- Classical color- and intensity-based detection of visually abnormal regions.
- Connected-component and contour-based region analysis.
- Calculation of leaf area and affected area in pixels.
- Calculation of affected leaf percentage.
- Calculation of abnormal-region centroids and bounding boxes.
- Project-defined visual severity classification.
- Generation of annotated images and structured JSON, CSV, and TXT reports.

The project does not provide agricultural, biological, or medical disease diagnosis. The detected regions represent visually abnormal areas based on image characteristics.

## Target Users

CropVision is intended for:

- Students and researchers studying classical computer vision.
- Computer vision learners working with image segmentation and region analysis.
- Agricultural technology developers exploring lightweight image-analysis approaches.
- Users who need a command-line tool for quantitative visual inspection of crop leaf images.

## High-Level Features

1. Leaf Segmentation
2. Visual Abnormality Detection
3. Spatial Analysis
4. Severity Classification
5. Visual Annotation
6. Structured Reports
7. Pipeline Inspection
8. Headless CLI Execution
