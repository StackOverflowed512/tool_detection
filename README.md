# Orthopedic Tool Detector

This project implements a computer vision system using OpenCV and Python to detect, classify, and measure common orthopedic tools found in an ACL/PCL surgical jig set from an input image.

## Description

The script analyzes an image containing orthopedic tools, identifies individual tools, classifies them into predefined categories (e.g., Drill Guide, Depth Gauge), and measures their key dimensions (length, width, diameter, hole properties, etc.) in millimeters. It provides both visual output (an annotated image) and structured data output (a JSON file).

## Features

*   **Tool Detection:** Locates potential tools in the image using contour detection based on edge maps and adaptive thresholding.
*   **Preprocessing:** Enhances image quality using grayscale conversion, CLAHE contrast enhancement, denoising, and morphological operations.
*   **Feature Extraction:** Calculates various shape and moment features for each detected contour (Area, Perimeter, Circularity, Aspect Ratio, Solidity, Hu Moments, Elongation, Holes).
*   **Classification:**
    *   Rule-based classification using extracted features and geometric properties.
    *   Optional classification using a pre-trained Scikit-learn (e.g., RandomForestClassifier) model loaded via `joblib`.
*   **Measurement:** Calculates relevant dimensions for each classified tool type (length, width, diameter, hole diameter/spacing, curvature radius, taper angle, etc.) using the minimum area bounding rectangle and contour analysis.
*   **Calibration:**
    *   Supports automatic calibration (`--auto-calibrate` enabled by default) by attempting to identify a known tool type and using its standard dimensions to estimate the pixel-to-millimeter ratio.
    *   Supports manual calibration simulation (`--calibrate <length_mm>`) where a reference length is provided, and the script uses the largest detected object to set the scale.
    *   Uses a default scale if calibration fails or is disabled.
*   **Visualization:** Generates an output image with detected tools highlighted, bounding boxes drawn, and labels including tool type, ID, and measured dimensions.
*   **Data Output:** Saves detailed results for each detected tool (ID, type, dimensions, area, contour points) to a JSON file.
*   **Debugging:** Optional debug mode (`--debug`) saves intermediate processing images (grayscale, edges, thresholded, cleaned).

## Installation

1.  **Prerequisites:**
    *   Python 3.7+
    *   `pip` (Python package installer)

2.  **Clone the Repository (Optional):**
    ```bash
    git clone [<your-repository-url>](https://github.com/StackOverflowed512/tool_detection/tree/main)
    ```

3.  **Set up a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The script is run from the command line.

**Basic Usage:**

```bash
python tool_detector.py path/to/your/image.jpg
