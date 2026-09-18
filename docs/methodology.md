# CropVision Computer Vision Methodology

This document details the mathematical, algorithmic, and image processing methodology implemented in **CropVision**.

---

## 1. Image Preprocessing

### Aspect-Ratio Preserving Downscaling
High-resolution images are resized to ensure computational efficiency while preserving physical aspect ratio:

$$\text{scale} = \min\left(1.0, \frac{\text{max\_image\_dim}}{\max(H, W)}\right)$$

$$W_{\text{new}} = \lfloor W \cdot \text{scale} \rceil, \quad H_{\text{new}} = \lfloor H \cdot \text{scale} \rceil$$

Resizing is performed using OpenCV's `cv2.INTER_AREA` interpolation for smooth antialiased downsampling.

### Colorspace Conversion (BGR $\rightarrow$ HSV)
Images are converted from BGR to Hue-Saturation-Value (HSV) space because HSV decouples chromaticity (Hue) from illumination (Value) and color purity (Saturation):

$$\text{HSV} = \text{cv2.cvtColor}(\text{BGR}, \text{cv2.COLOR\_BGR2HSV})$$

- **Hue ($H$)**: $[0, 180]$ (representing color tone on the color wheel).
- **Saturation ($S$)**: $[0, 255]$ (representing color intensity/purity).
- **Value ($V$)**: $[0, 255]$ (representing brightness/lightness).

---

## 2. Leaf Segmentation

Leaf extraction segments the primary plant organ from the background:

1. **HSV Color Thresholding**: Green foliage typically occupies Hue values in $[25, 95]$ in OpenCV 8-bit scale:
   $$M_{\text{raw}}(x,y) = \begin{cases} 255 & \text{if } H(x,y) \in [25, 95] \wedge S(x,y) \ge 25 \wedge V(x,y) \ge 25 \\ 0 & \text{otherwise} \end{cases}$$

2. **Morphological Mask Refinement**:
   - **Opening** ($\circ$): Removes small background speckles using an elliptical $7 \times 7$ kernel:
     $$M_{\text{opened}} = M_{\text{raw}} \circ K_{(7 \times 7)}$$
   - **Closing** ($\bullet$): Bridges gaps and small fissures inside the leaf blade:
     $$M_{\text{closed}} = M_{\text{opened}} \bullet K_{(7 \times 7)}$$
   - **Hole Filling**: Fills interior holes (e.g. leaf veins or bright reflections) by finding external contours and drawing solid filled polygons (`cv2.FILLED`).

3. **Main Leaf Contour Selection**:
   - Finds external contours: $\mathcal{C} = \text{cv2.findContours}(M_{\text{closed}}, \text{RETR\_EXTERNAL})$.
   - Selects contour with maximum area: $C_{\text{main}} = \arg\max_{c \in \mathcal{C}} \text{Area}(c)$.
   - Retains $C_{\text{main}}$ if $\text{Area}(C_{\text{main}}) \ge \text{min\_leaf\_area}$ (500 px), creating solid single-leaf mask $M_{\text{leaf}}$.

---

## 3. Visually Abnormal Region Detection

To detect lesions, chlorosis, and necrosis inside the leaf boundary while avoiding leaf edge anti-aliasing artifacts:

1. **Border-Eroded ROI**: The leaf mask is eroded with a $5 \times 5$ elliptical kernel to yield an inner ROI:
   $$M_{\text{inner}} = M_{\text{leaf}} \ominus K_{(5 \times 5)}$$

2. **Targeted Abnormality Rules**:
   - **Necrotic / Brown / Reddish Lesions**:
     - Range 1: $H \in [0, 22], S \in [25, 255], V \in [20, 220]$
     - Range 2: $H \in [160, 180], S \in [25, 255], V \in [20, 220]$
   - **Chlorotic / Yellow Spots**:
     - Range: $H \in [15, 34], S \in [50, 255], V \in [80, 255]$
   - **Dark / Black Lesions**:
     - Range: $V \in [0, 45]$ inside $M_{\text{inner}}$
   - **Non-Green Hue Anomaly**:
     - $(H < 22 \vee H > 95) \wedge S \ge 30 \wedge V \ge 30$

3. **Morphological Cleaning**:
   - Morphological opening ($5 \times 5$ kernel) eliminates isolated 1–2 pixel noise.
   - Morphological closing ($5 \times 5$ kernel) merges adjacent lesion clusters.
   - Connected component area filtering discards objects with area $< \text{min\_abnormal\_area}$ (100 px).

---

## 4. Connected Component Analysis & Quantitative Metrics

### Shape and Geometric Filtering
Candidate abnormal contours are extracted via `cv2.findContours`. Each candidate region is evaluated against shape heuristics:

- **Aspect Ratio**:
  $$\text{Aspect Ratio} = \frac{\max(W_r, H_r)}{\max(1, \min(W_r, H_r))} \le 8.0$$
  *(Discards thin linear edge/vein artifacts)*

- **Solidity**:
  $$\text{Solidity} = \frac{\text{Area}(C)}{\text{Area}(\text{ConvexHull}(C))} \ge 0.25$$
  *(Discards sparse, fragmented noise)*

### Centroid Calculation
Spatial centroids $(c_x, c_y)$ for each accepted region are computed using 1st-order spatial moments:

$$M_{00} = \sum_{x,y} I(x,y), \quad M_{10} = \sum_{x,y} x \cdot I(x,y), \quad M_{01} = \sum_{x,y} y \cdot I(x,y)$$

$$c_x = \text{round}\left(\frac{M_{10}}{M_{00}}\right), \quad c_y = \text{round}\left(\frac{M_{01}}{M_{00}}\right)$$

### Area & Affected Percentage
- **Leaf Area ($A_{\text{leaf}}$)**: $\sum_{x,y} \mathbb{I}(M_{\text{leaf}}(x,y) > 0)$
- **Affected Area ($A_{\text{affected}}$)**: $\sum_{x,y} \mathbb{I}(M_{\text{abnormal}}(x,y) > 0)$
- **Affected Percentage ($P_{\text{affected}}$)**:

$$P_{\text{affected}} = \begin{cases} \frac{A_{\text{affected}}}{A_{\text{leaf}}} \times 100.0 & \text{if } A_{\text{leaf}} > 0 \\ 0.0 & \text{otherwise} \end{cases}$$

---

## 5. Severity Classification Heuristics

Project-defined heuristic thresholds categorize the visual severity level:

| Severity Level | Affected Percentage Range ($P_{\text{affected}}$) |
|---|---|
| **`LOW`** | $P_{\text{affected}} < 5.0\%$ |
| **`MODERATE`** | $5.0\% \le P_{\text{affected}} < 15.0\%$ |
| **`HIGH`** | $15.0\% \le P_{\text{affected}} < 30.0\%$ |
| **`SEVERE`** | $P_{\text{affected}} \ge 30.0\%$ |
