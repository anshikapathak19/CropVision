# CropVision System Limitations & Scope

This document outlines the technical limitations, environmental constraints, and non-diagnostic operational scope of the **CropVision** computer vision toolkit.

---

## Technical & Environmental Limitations

### 1. Classical Color-Based Sensitivity to Lighting & Backgrounds
Because CropVision relies strictly on classical computer vision algorithms (HSV thresholding and morphological operations), detection accuracy is sensitive to environmental lighting:
- **Direct Sunlight / Specular Highlights**: Harsh sunlight can produce bright white reflections (high Value, low Saturation) that disrupt green leaf segmentation or trigger false anomalies.
- **Deep Shadows**: Strong shadows cast across the leaf surface lower pixel Value ($V < 30$), which may lead to shadow regions being flagged as dark lesions.
- **Non-Standard Backgrounds**: Backgrounds containing green debris, weeds, or clothing matching the leaf HSV range ($H \in [25, 95]$) can cause background pixels to merge with the leaf mask.

### 2. Visually Abnormal Regions vs. Biological Disease
CropVision detects **visual discoloration and geometric anomalies** (brown necrosis, chlorotic yellowing, dark spots, non-green tissue). Visual anomalies are **not guaranteed to be caused by biological pathogens or plant diseases**:
- **Non-Disease Factors**: Physical abrasion, insect bite marks, mud/dirt specks, nutrient deficiencies, sunburn, or physiological senescence produce similar visual discolorations.
- **Sub-visual Pathogens**: Early-stage latent infections that do not manifest visible surface discoloration cannot be detected by classical visual thresholding.

### 3. Plant Species Scope (Tomato-Leaf Focused)
- The default HSV ranges ($H \in [25, 95]$, $S \ge 25$, $V \ge 25$) and severity heuristics have been calibrated primarily on tomato crop leaf datasets.
- Non-green foliage crops (e.g. purple cabbage, reddish ornamental leaves, or variegated leaves) will require threshold adjustments in `src/config.py`.

### 4. Overlapping Leaves & Complex Canopy Segmentation
- In dense crop canopies where multiple leaves overlap or touch, classical contour selection selects the largest contiguous connected component.
- Overlapping leaves with similar green hues will be segmented as a single combined leaf contour rather than individual isolated leaf blades.

### 5. Static Threshold Constraints
- Static HSV thresholds may require parameter tuning (`--min-area`, `--max-dim`, or `PipelineConfig` edits) when processing images acquired under significantly different camera sensors or white balance settings.

---

## Non-Diagnostic Disclaimer

> **IMPORTANT**: CropVision is a technical computer vision toolkit intended strictly for image analysis and quantitative surface measurement. It does **NOT** provide agricultural, biological, or medical disease diagnoses. All output metrics (affected area, affected percentage, severity classification) represent visual image properties only.
