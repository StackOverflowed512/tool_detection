# --- START OF FILE tool_detector.py ---

import cv2
import numpy as np
import os
import math
from sklearn.ensemble import RandomForestClassifier
import joblib
import json
from typing import List, Dict, Tuple, Optional

class OrthopedicToolDetector:
    def __init__(self, model_path: str = None, pixel_to_mm: float = 0.1, debug_mode: bool = False):
        """
        Initialize the detector for ACL/PCL jig set tools.

        Args:
            model_path: Path to trained ML model (optional)
            pixel_to_mm: Conversion factor from pixels to millimeters
            debug_mode: Enable debugging output
        """
        # Tool-specific colors for visualization
        self.colors = {
            'drill_guide': (0, 255, 0),      # Green
            'depth_gauge': (0, 165, 255),    # Orange
            'sizing_block': (255, 0, 0),     # Blue
            'alignment_rod': (0, 0, 255),    # Red
            'tunnel_dilator': (255, 0, 255), # Purple
            'unknown': (128, 128, 128)       # Gray
        }

        # Detection thresholds and parameters
        self.thresholds = {
            'min_area': 300,                  # Reduced to catch smaller tools
            'circularity': 0.6,               # More lenient for circular tools
            'aspect_ratio': {                 # Fine-tuned for better classification
                'drill_guide': (1.5, 6.0),
                'depth_gauge': (2.5, 12.0),
                'sizing_block': (0.8, 1.7)
            },
            'solidity': 0.8,                 # For solid objects
            'hole_ratio': 0.15,              # For tools with holes
            'elongation': 3.0                # For rod-like tools
        }

        # Visualization settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 1 # Reduced thickness slightly for potentially more text
        self.text_color = (255, 255, 255)
        self.line_thickness = 2

        # Debugging
        self.debug_mode = debug_mode

        # Measurement conversion
        self.pixel_to_mm = pixel_to_mm

        # Classification model
        self.model = None
        self.classes = ['drill_guide', 'depth_gauge', 'sizing_block',
                       'alignment_rod', 'tunnel_dilator', 'unknown']

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """Load a trained classification model"""
        try:
            self.model = joblib.load(model_path)
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def preprocess_image(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Preprocess the image for tool detection.

        Args:
            image: Input BGR image

        Returns:
            tuple: (binary_mask, edge_map, grayscale_image) or (None, None, None) if error
        """
        if image is None or image.size == 0:
            print("Error: Input image is empty in preprocess_image.")
            return None, None, None

        # Make a copy
        original = image.copy()

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Denoise while preserving edges
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)

        # Edge detection
        edges = cv2.Canny(denoised, 30, 150)

        # Dilate edges to connect gaps
        kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)

        # Multi-level thresholding for robust tool extraction
        _, thresh1 = cv2.threshold(denoised, 60, 255, cv2.THRESH_BINARY_INV)
        thresh2 = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Combine thresholds
        combined_thresh = cv2.bitwise_or(thresh1, thresh2)

        # Clean up using morphological operations
        cleaned = cv2.morphologyEx(combined_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

        # Save intermediate results for debugging
        if self.debug_mode:
            cv2.imwrite('debug_gray.jpg', gray)
            cv2.imwrite('debug_enhanced.jpg', enhanced)
            cv2.imwrite('debug_edges.jpg', edges)
            cv2.imwrite('debug_thresholded.jpg', combined_thresh)
            cv2.imwrite('debug_cleaned.jpg', cleaned)

        return cleaned, dilated_edges, gray

    def detect_tools(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], List[Dict]]:
        """
        Detect and measure tools in the ACL/PCL jig set.

        Args:
            image: Input BGR image

        Returns:
            tuple: (annotated_image, detected_tools)
        """
        if image is None:
            print("Error: Input image is None in detect_tools.")
            return None, []

        # Create a larger canvas for visualization - INCREASED HEIGHT
        h, w = image.shape[:2]
        # --- CHANGE: Increased canvas height ---
        canvas_h = h + 150 # More space for text
        result_image = np.zeros((canvas_h, w, 3), dtype=np.uint8)
        result_image[:h, :w] = image.copy()

        # Preprocess the image
        mask, edges, gray = self.preprocess_image(image)
        if mask is None or edges is None or gray is None:
            print("Error: Preprocessing failed.")
            # Return the original image on canvas if preprocessing failed
            return result_image, []

        # Find contours from both mask and edges
        contours_mask, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_edges, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Combine and filter contours
        all_contours = list(contours_mask) + list(contours_edges)
        filtered_contours = []

        for contour in all_contours:
             # Check if contour is valid before calculating area
            if contour is not None and len(contour) > 0:
                area = cv2.contourArea(contour)
                if area >= self.thresholds['min_area']:
                    filtered_contours.append(contour)
            else:
                if self.debug_mode:
                    print("Warning: Invalid contour encountered during filtering.")


        # Remove duplicate contours (those with high overlap)
        unique_contours = self.remove_duplicate_contours(filtered_contours, image.shape[:2]) # Pass image shape

        # Sort contours by area (largest first)
        unique_contours = sorted(unique_contours, key=cv2.contourArea, reverse=True)

        # Process each tool
        results = []

        for i, contour in enumerate(unique_contours):
            # Skip extremely small contours again after potential duplicates removed
            if contour is None or len(contour) < 5 or cv2.contourArea(contour) < self.thresholds['min_area']:
                continue

            # Extract features and classify
            features = self.extract_features(contour, gray)
            tool_type = self.classify_tool(features, contour)

            # Skip if classified as unknown and too small (heuristic)
            if tool_type == 'unknown' and cv2.contourArea(contour) < self.thresholds['min_area'] * 1.5:
                 continue

            # Measure dimensions
            dimensions = self.measure_tool(contour, tool_type, image)

            if dimensions:
                # Ensure contour is serializable
                try:
                    contour_list = contour.tolist()
                except:
                    print(f"Warning: Could not convert contour {i+1} to list. Skipping.")
                    continue

                results.append({
                    'id': i+1,
                    'type': tool_type,
                    'dimensions': dimensions,
                    'area_pixels': cv2.contourArea(contour),
                    'contour': contour_list # Use the serializable list
                })

                # Draw results on image
                self.draw_detection(result_image, contour, tool_type, dimensions, i+1)

        # Add scale information at the bottom
        # --- CHANGE: Adjusted text position for taller canvas ---
        cv2.putText(
            result_image, f"Scale: 1 pixel = {self.pixel_to_mm:.4f} mm",
            (10, canvas_h - 60), self.font, self.font_scale, (255, 255, 255), self.font_thickness
        )
        cv2.putText(
            result_image, f"Total tools detected: {len(results)}",
            (10, canvas_h - 30), self.font, self.font_scale, (255, 255, 255), self.font_thickness
        )

        return result_image, results

    def remove_duplicate_contours(self, contours: List[np.ndarray], image_shape: Tuple[int, int]) -> List[np.ndarray]:
        """Remove duplicate contours based on overlap"""
        unique_contours = []
        h, w = image_shape # Get image dimensions

        for contour in contours:
            if contour is None or len(contour) == 0: continue # Skip invalid contours

            is_duplicate = False
            # Create masks with actual image dimensions
            mask1 = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask1, [contour], 0, 255, -1)
            area1 = cv2.countNonZero(mask1)
            if area1 == 0: continue # Skip empty contours

            for idx, existing in enumerate(unique_contours):
                if existing is None or len(existing) == 0: continue # Skip invalid existing contours

                mask2 = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(mask2, [existing], 0, 255, -1)
                area2 = cv2.countNonZero(mask2)
                if area2 == 0: continue # Skip empty existing contours

                # Find intersection
                intersection = cv2.bitwise_and(mask1, mask2)
                overlap_area = cv2.countNonZero(intersection)

                # If overlap is significant, consider it a duplicate
                if overlap_area > 0.7 * min(area1, area2):
                    is_duplicate = True
                    # Keep the one with larger area
                    if area1 > area2:
                        unique_contours[idx] = contour # Replace the smaller one
                    break # Stop checking against others once a duplicate is found

            if not is_duplicate:
                unique_contours.append(contour)

        return unique_contours

    def extract_features(self, contour: np.ndarray, gray_image: np.ndarray) -> List[float]:
        """Extract features from a contour for classification"""
        if contour is None or len(contour) < 5:
            # Return a list of zeros with the expected length
            return [0.0] * 12

        try:
            # Basic shape features
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = 0.0
            if perimeter > 0:
                 circularity = 4 * np.pi * area / (perimeter**2)

            # Bounding rectangle features
            rect = cv2.minAreaRect(contour)
            center, (width, height), angle = rect
            aspect_ratio = 0.0
            if min(width, height) > 1e-5: # Avoid division by zero
                 aspect_ratio = max(width, height) / min(width, height)

            # Convexity features
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = 0.0
            if hull_area > 1e-5: # Avoid division by zero
                 solidity = area / hull_area

            # Hole detection
            has_hole, hole_ratio, num_holes = self.detect_holes(contour, gray_image)

            # Moment features
            moments = cv2.moments(contour)
            hu_moments = cv2.HuMoments(moments).flatten() # Flatten for consistency

            # Elongation
            elongation = 0.0
            mu20 = moments.get('mu20', 0)
            mu02 = moments.get('mu02', 0)
            mu11 = moments.get('mu11', 0)
            denominator = (mu20 + mu02)**2
            if denominator > 1e-5: # Avoid division by zero
                numerator = (mu20 - mu02)**2 + 4*(mu11**2)
                elongation = numerator / denominator
                # Sometimes elongation can slightly exceed 1 due to noise, cap it
                elongation = min(elongation, 1.0)

            # Ensure hu_moments has enough elements, pad with 0 if necessary
            hu1 = hu_moments[0] if len(hu_moments) > 0 else 0.0
            hu2 = hu_moments[1] if len(hu_moments) > 1 else 0.0

            return [
                float(area), float(perimeter), float(circularity), float(aspect_ratio),
                float(solidity), float(has_hole), float(hole_ratio), float(num_holes),
                float(elongation), float(hu1), float(hu2), float(angle)
            ]
        except Exception as e:
             print(f"Error extracting features: {e}. Returning zeros.")
             # Return default values if any calculation fails
             return [0.0] * 12


    def detect_holes(self, contour: np.ndarray, gray_image: np.ndarray) -> Tuple[bool, float, int]:
        """
        Detect if the tool has holes and calculate hole ratio

        Returns:
            tuple: (has_hole, hole_ratio, num_holes)
        """
        has_hole = False
        hole_ratio = 0.0
        num_holes = 0

        if contour is None or len(contour) < 3 or gray_image is None:
            return has_hole, hole_ratio, num_holes

        try:
            # Create mask for the tool
            mask = np.zeros(gray_image.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], 0, 255, -1)

            # Find contours with hierarchy to detect holes
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

            area = cv2.contourArea(contour)
            if area <= 0: # Avoid division by zero later
                return has_hole, hole_ratio, num_holes

            if hierarchy is not None and len(hierarchy) > 0:
                # Hierarchy is typically shape (1, N, 4)
                # Iterate through top-level contours (hierarchy[0])
                for i, h in enumerate(hierarchy[0]):
                    # h[3] is the parent index. If it's >= 0, it's an inner contour (hole)
                    # We check if the parent is 0 (the main outer contour)
                    if h[3] == 0:
                        hole_contour = contours[i]
                        hole_area = cv2.contourArea(hole_contour)
                        # Filter small noise and holes almost as big as the object itself
                        if hole_area > area * 0.01 and hole_area < area * 0.9:
                            has_hole = True
                            hole_ratio += hole_area / area
                            num_holes += 1

        except Exception as e:
            print(f"Error detecting holes: {e}")
            # Return default values on error

        return has_hole, hole_ratio, num_holes


    def classify_tool(self, features: List[float], contour: np.ndarray) -> str:
        """Classify the tool using features or ML model"""
        # --- ML Model Classification (if available) ---
        if self.model is not None:
            try:
                # Ensure features are valid numbers
                if any(math.isnan(f) or math.isinf(f) for f in features):
                    print("Warning: NaN or Inf in features for ML classification. Falling back to rules.")
                else:
                    features_array = np.array(features).reshape(1, -1)
                    # Ensure the number of features matches the model's expectation
                    if features_array.shape[1] == self.model.n_features_in_:
                        prediction = self.model.predict(features_array)[0]
                        # Basic sanity check on prediction
                        if prediction in self.classes:
                            return prediction
                        else:
                             print(f"Warning: Model predicted unknown class '{prediction}'. Falling back to rules.")
                    else:
                        print(f"Warning: Feature length mismatch ({features_array.shape[1]} vs {self.model.n_features_in_}). Falling back to rules.")

            except Exception as e:
                print(f"ML classification failed: {e}. Falling back to rule-based.")

        # --- Enhanced Rule-Based Classification ---
        try:
            # Unpack features safely
            if len(features) != 12:
                 print(f"Warning: Incorrect feature vector length ({len(features)}) in rule-based classification.")
                 return 'unknown' # Cannot classify reliably

            area, perimeter, circularity, aspect_ratio, solidity, has_hole, hole_ratio, num_holes, elongation, *_ = features

            # Get shape approximation for better classification
            if perimeter <= 0: return 'unknown' # Cannot approximate without perimeter
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            num_vertices = len(approx) if approx is not None else 0

            # Refined Rules:
            # Drill guide - rectangularish, holes, specific aspect ratio
            if (self.thresholds['aspect_ratio']['drill_guide'][0] <= aspect_ratio <=
                self.thresholds['aspect_ratio']['drill_guide'][1] and
                (has_hole > 0.5 or num_vertices < 10) and solidity > 0.75): # Use > 0.5 for boolean feature
                return 'drill_guide'

            # Depth gauge - long, narrow, often has holes or high elongation
            # Increased aspect ratio lower bound slightly
            if (aspect_ratio >= self.thresholds['aspect_ratio']['depth_gauge'][0] and
                (num_holes > 0 or (elongation > self.thresholds['elongation'] and aspect_ratio > 4.0))): # Combine elongation with high AR
                return 'depth_gauge'

            # Sizing block - squarish, solid
            if (self.thresholds['aspect_ratio']['sizing_block'][0] <= aspect_ratio <=
                self.thresholds['aspect_ratio']['sizing_block'][1] and solidity > 0.85 and not has_hole): # Usually no large holes
                return 'sizing_block'

            # Alignment rod - very elongated or somewhat circular (ends)
            # Added condition for high elongation OR high circularity
            if (elongation > 0.9 and aspect_ratio > 5.0) or \
               (circularity > self.thresholds['circularity'] and aspect_ratio < 2.0): # Check for circular ends
                return 'alignment_rod'

            # Tunnel dilator - often has holes, might be tapered
            if has_hole > 0.5 and hole_ratio > self.thresholds['hole_ratio'] and num_holes >= 1: # Relaxed to 1 hole minimum
                 return 'tunnel_dilator'

            # Fallback for less distinct shapes based on elongation/circularity
            if elongation > 0.8 and aspect_ratio > 4.0:
                 return 'depth_gauge' # Could be depth gauge or similar long tool
            if circularity > 0.75 and aspect_ratio < 1.5:
                 return 'sizing_block' # Or potentially a washer/circular part


        except Exception as e:
             print(f"Error during rule-based classification: {e}")
             # Fallback to unknown if rules fail

        # Default case for unknown tools
        return 'unknown'


    def measure_tool(self, contour: np.ndarray, tool_type: str, image: np.ndarray) -> Optional[Dict[str, float]]:
        """Measure dimensions of the tool in millimeters"""
        if contour is None or len(contour) < 5:
            return None

        try:
            # Get minimum area rectangle for more accurate measurements
            rect = cv2.minAreaRect(contour)
            center, (width, height), angle = rect

            # Ensure width and height are non-negative
            width = abs(width)
            height = abs(height)

            # Convert to mm
            width_mm = width * self.pixel_to_mm
            height_mm = height * self.pixel_to_mm
            area_mm2 = cv2.contourArea(contour) * self.pixel_to_mm * self.pixel_to_mm

            # Tool-specific measurements
            if tool_type == 'drill_guide':
                hole_info = self.detect_drill_guide_holes(contour, image)
                return {
                    'length': round(max(width_mm, height_mm), 1),
                    'width': round(min(width_mm, height_mm), 1),
                    'hole_diameter': round(hole_info.get('diameter', 0) * self.pixel_to_mm, 1),
                    'hole_spacing': round(hole_info.get('spacing', 0) * self.pixel_to_mm, 1),
                    'num_holes': float(hole_info.get('count', 0)),
                    'area_mm2': round(area_mm2, 1)
                }
            elif tool_type == 'depth_gauge':
                markings = self.detect_markings(contour, image)
                # Max depth is typically close to the length for these tools
                max_depth_est = round(max(width_mm, height_mm) * 0.95, 1)
                return {
                    'length': round(max(width_mm, height_mm), 1),
                    'width': round(min(width_mm, height_mm), 1),
                    'markings_est': float(markings), # Estimated markings
                    'max_depth_est': max_depth_est, # Estimated max depth
                    'area_mm2': round(area_mm2, 1)
                }
            elif tool_type == 'sizing_block':
                # Thickness estimation is very rough, based on min dimension
                thickness_est = round(min(width_mm, height_mm) * 0.3, 1)
                return {
                    'width': round(width_mm, 1),
                    'height': round(height_mm, 1),
                    'thickness_est': thickness_est, # Estimated thickness
                    'diagonal': round(math.sqrt(width_mm**2 + height_mm**2), 1),
                    'area_mm2': round(area_mm2, 1)
                }
            elif tool_type == 'alignment_rod':
                # Diameter is the smaller dimension of the bounding box
                # Radius of curvature estimation
                curvature_radius_px = self.estimate_curvature(contour)
                curvature_radius_mm = curvature_radius_px * self.pixel_to_mm if curvature_radius_px < 999 else float('inf') # Use infinity for straight
                return {
                    'diameter': round(min(width_mm, height_mm), 1),
                    'length': round(max(width_mm, height_mm), 1),
                    'radius_of_curvature': round(curvature_radius_mm, 1) if curvature_radius_mm != float('inf') else curvature_radius_mm,
                    'area_mm2': round(area_mm2, 1)
                }
            elif tool_type == 'tunnel_dilator':
                hole_info = self.detect_holes_detailed(contour, image)
                # Inner diameter from detected holes, outer from bounding box
                # Length from bounding box
                taper_angle = self.calculate_taper(contour)
                return {
                    'outer_diameter': round(max(width_mm, height_mm), 1), # Approximation using max dim
                    'inner_diameter_est': round(hole_info.get('avg_diameter', 0) * self.pixel_to_mm, 1),
                    'length': round(max(width_mm, height_mm), 1), # Often length is max dim
                    'taper_angle_est': round(taper_angle, 1),
                    'num_holes': float(hole_info.get('count', 0)),
                    'area_mm2': round(area_mm2, 1)
                }
            else: # Unknown tool
                return {
                    'max_dimension': round(max(width_mm, height_mm), 1),
                    'min_dimension': round(min(width_mm, height_mm), 1),
                    'area_mm2': round(area_mm2, 1)
                }
        except Exception as e:
             print(f"Error measuring tool ({tool_type}): {e}")
             return None # Return None if measurement fails


    def detect_drill_guide_holes(self, contour: np.ndarray, image: np.ndarray) -> Dict:
        """
        Enhanced detection of holes in drill guides

        Returns:
            dict: With hole diameter, spacing, and count in pixels
        """
        hole_results = {'diameter': 0.0, 'spacing': 0.0, 'count': 0}
        if contour is None or len(contour) < 3 or image is None:
             return hole_results

        try:
             # Create a mask of the contour
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour], 0, 255, -1)

            # Find contours inside using hierarchy (better for holes)
            contours_inside, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

            valid_holes = []
            tool_area = cv2.contourArea(contour)
            if tool_area <=0: return hole_results # Avoid division by zero

            if hierarchy is not None and len(hierarchy) > 0:
                for i, h in enumerate(hierarchy[0]):
                    # Check if it's an inner contour (hole) of the main tool (parent index 0)
                    if h[3] == 0:
                        hole_contour = contours_inside[i]
                        area = cv2.contourArea(hole_contour)
                        # Filter holes: reasonable area relative to tool, enclosed circle radius > 1 pixel
                        if area > tool_area * 0.005 and area < tool_area * 0.5:
                            (x, y), radius = cv2.minEnclosingCircle(hole_contour)
                            if radius > 1.0: # Ensure it's not just noise
                                valid_holes.append({'center': (x, y), 'radius': radius, 'contour': hole_contour})

            hole_results['count'] = len(valid_holes)

            if len(valid_holes) > 0:
                # Calculate average diameter
                diameters = [2 * hole['radius'] for hole in valid_holes]
                avg_diameter = sum(diameters) / len(diameters)
                hole_results['diameter'] = avg_diameter

                # Calculate average spacing between hole centers if multiple holes exist
                if len(valid_holes) > 1:
                    centers = [hole['center'] for hole in valid_holes]
                    distances = []
                    for i in range(len(centers)):
                        for j in range(i + 1, len(centers)):
                            dist = math.dist(centers[i], centers[j]) # Use math.dist
                            distances.append(dist)

                    if distances:
                        avg_spacing = sum(distances) / len(distances)
                        hole_results['spacing'] = avg_spacing
                    else: # Only one hole, spacing is not applicable, maybe use tool dimension?
                         rect = cv2.minAreaRect(contour)
                         _, (w, h), _ = rect
                         hole_results['spacing'] = max(w, h) * 0.4 # Fallback estimate


            # Fallback estimations if no holes are detected but expected (e.g., drill guide)
            if hole_results['count'] == 0 and hole_results['diameter'] == 0:
                 rect = cv2.minAreaRect(contour)
                 _, (width, height), _ = rect
                 min_dim_px = min(width, height)
                 if min_dim_px > 0: # Avoid division by zero
                     # Estimate based on typical drill guide hole relative to tool width
                     hole_results['diameter'] = min_dim_px * 0.2
                     # Need at least one estimated hole if we provide a diameter
                     hole_results['count'] = 1
                     # Estimate spacing based on length if diameter was estimated
                     max_dim_px = max(width, height)
                     hole_results['spacing'] = max_dim_px * 0.4

        except Exception as e:
             print(f"Error detecting drill guide holes: {e}")
             # Return defaults on error

        # Ensure values are floats before returning
        hole_results['diameter'] = float(hole_results['diameter'])
        hole_results['spacing'] = float(hole_results['spacing'])
        hole_results['count'] = int(hole_results['count'])

        return hole_results

    def detect_holes_detailed(self, contour: np.ndarray, image: np.ndarray) -> Dict:
        """
        Detailed hole analysis for tools (like tunnel dilators)

        Returns:
            dict: With hole details in pixels {count, avg_diameter, max_diameter, min_diameter}
        """
        hole_details = {'count': 0, 'avg_diameter': 0.0, 'max_diameter': 0.0, 'min_diameter': 0.0}
        if contour is None or len(contour) < 3 or image is None:
            return hole_details

        try:
            # Create mask and find internal contours using hierarchy
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour], 0, 255, -1)
            contours_inside, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

            valid_holes = []
            tool_area = cv2.contourArea(contour)
            if tool_area <= 0: return hole_details

            if hierarchy is not None and len(hierarchy) > 0:
                for i, h in enumerate(hierarchy[0]):
                    if h[3] == 0: # Check if it's an inner contour of the main tool
                        hole_contour = contours_inside[i]
                        area = cv2.contourArea(hole_contour)
                        # Filter: area relative to tool, min radius
                        if area > tool_area * 0.01 and area < tool_area * 0.8:
                             (x, y), radius = cv2.minEnclosingCircle(hole_contour)
                             # Also check if center is well inside the main contour
                             if radius > 1.5 and cv2.pointPolygonTest(contour, (int(x), int(y)), False) > 0:
                                  valid_holes.append({'radius': radius})

            hole_details['count'] = len(valid_holes)

            if len(valid_holes) > 0:
                diameters = [2 * hole['radius'] for hole in valid_holes]
                hole_details['avg_diameter'] = sum(diameters) / len(diameters)
                hole_details['max_diameter'] = max(diameters)
                hole_details['min_diameter'] = min(diameters)
            else:
                 # Fallback estimation if no holes detected, based on min dimension
                 rect = cv2.minAreaRect(contour)
                 _, (width, height), _ = rect
                 min_dim_px = min(width, height)
                 if min_dim_px > 0:
                     estimated_diameter = min_dim_px * 0.3 # Estimate hole is 30% of min dim
                     hole_details['avg_diameter'] = estimated_diameter
                     hole_details['max_diameter'] = estimated_diameter
                     hole_details['min_diameter'] = estimated_diameter
                     # Assume at least one hole if we estimate diameter
                     hole_details['count'] = 1


        except Exception as e:
             print(f"Error detecting detailed holes: {e}")
             # Return defaults on error

        # Ensure float values
        hole_details['avg_diameter'] = float(hole_details['avg_diameter'])
        hole_details['max_diameter'] = float(hole_details['max_diameter'])
        hole_details['min_diameter'] = float(hole_details['min_diameter'])
        hole_details['count'] = int(hole_details['count'])

        return hole_details


    def detect_markings(self, contour: np.ndarray, image: np.ndarray) -> int:
        """
        Detect measurement markings on tools like depth gauges (ESTIMATION)

        Returns:
            int: Estimated number of markings
        """
        if contour is None or len(contour) < 5:
            return 0

        try:
             # For depth gauges, estimate the number of measurement markings
            rect = cv2.minAreaRect(contour)
            _, (width, height), _ = rect
            max_dim_px = max(abs(width), abs(height))

            # Simplified estimation based on typical marking spacing (~5-10mm apart usually)
            # Let's assume markings are roughly every 7mm = 7/pixel_to_mm pixels
            pixels_per_marking_est = 7.0 / self.pixel_to_mm if self.pixel_to_mm > 1e-5 else 70 # Avoid division by zero
            if pixels_per_marking_est < 5: pixels_per_marking_est = 5 # Minimum pixel spacing

            # Estimate number of markings along the longest dimension
            num_markings = int(max_dim_px / pixels_per_marking_est)

            # Return a reasonable minimum/maximum estimate
            return max(3, min(num_markings, 30)) # Clamp between 3 and 30

        except Exception as e:
             print(f"Error estimating markings: {e}")
             return 5 # Default estimate


    def estimate_curvature(self, contour: np.ndarray) -> float:
        """
        Estimate the radius of curvature for curved tools like alignment rods

        Returns:
            float: Estimated radius of curvature in pixels (or large value for straight)
        """
        if contour is None or len(contour) < 10: # Need more points for curvature
            return 10000.0 # Assume straight if not enough points

        try:
            # Fit an ellipse - major/minor axis ratio gives an idea of curvature
            ellipse = cv2.fitEllipse(contour)
            center, (minor_axis, major_axis), angle = ellipse

            # Ensure axes are positive
            minor_axis = abs(minor_axis)
            major_axis = abs(major_axis)

            if major_axis < 1e-5 or minor_axis < 1e-5:
                return 10000.0 # Cannot estimate if axes are zero

            aspect_ratio = major_axis / minor_axis

            # If aspect ratio is very high, it's likely straight or nearly straight
            if aspect_ratio > 15: # More aggressive threshold for straightness
                 return 10000.0

            # Simplified estimation: radius proportional to major axis squared / minor axis
            # This is not geometrically perfect but gives a relative measure
            # A more circular ellipse (aspect ratio ~ 1) should have small radius? No, that's wrong.
            # Let's use the formula for radius of curvature of an ellipse at the end of major axis: R = b^2 / a
            # where a = major_axis/2, b = minor_axis/2
            a = major_axis / 2.0
            b = minor_axis / 2.0
            if a > 1e-5:
                radius_of_curvature = (b**2) / a
                # If the radius is extremely small, it might be noise, treat as straight-ish
                # Or if it's very large, it's also straight-ish
                if radius_of_curvature < 5.0 or radius_of_curvature > 5000.0:
                     return 10000.0
                return radius_of_curvature
            else:
                return 10000.0 # Straight if major axis is tiny


        except cv2.error as e:
            # Handle cases where fitEllipse fails (e.g., contour is a straight line)
            if "points must be >= 5" in str(e) or "fitting failed" in str(e).lower():
                 # Check linearity by fitting a line
                 [vx,vy,x,y] = cv2.fitLine(contour, cv2.DIST_L2,0,0.01,0.01)
                 # Calculate distance of points from the line (simplified check)
                 max_dist = 0
                 p1 = np.array([x, y])
                 v = np.array([vx, vy])
                 for pt in contour:
                     p2 = pt[0]
                     d = np.linalg.norm(np.cross(v.flatten(), (p2-p1).flatten()))
                     if d > max_dist: max_dist = d
                 if max_dist < 3.0: # If points are very close to a line -> straight
                     return 10000.0
                 else: # If it failed ellipse fit but isn't linear, maybe highly irregular
                     return 1000.0 # Return moderately large radius

            else:
                 print(f"Error estimating curvature (fitEllipse): {e}")
                 return 10000.0 # Assume straight on other errors
        except Exception as e:
            print(f"Error estimating curvature: {e}")
            return 10000.0 # Assume straight on error


    def calculate_taper(self, contour: np.ndarray) -> float:
        """
        Calculate taper angle for tunnel dilators or other tapered tools (ESTIMATION)

        Returns:
            float: Taper angle in degrees (estimated)
        """
        if contour is None or len(contour) < 5:
            return 0.0 # Default no taper

        try:
            rect = cv2.minAreaRect(contour)
            center, (width, height), angle = rect
            width = abs(width)
            height = abs(height)

            if max(width, height) < 1e-5: return 0.0 # Avoid division by zero

            # Very rough estimate: Assume the difference between max/min width along the length
            # represents the taper. This requires finding the skeleton or principal axis.

            # Simpler approach using ellipse fitting (as before):
            ellipse = cv2.fitEllipse(contour)
            _, (minor_axis, major_axis), _ = ellipse
            minor_axis = abs(minor_axis)
            major_axis = abs(major_axis)

            if major_axis < 1e-5 or minor_axis < 1e-5 or major_axis <= minor_axis:
                return 0.0 # No taper if axes invalid or minor >= major

            # Estimate taper angle based on the difference between axes over the length
            # angle = atan( (MajorDiameter - MinorDiameter) / (2 * Length) )
            # Use ellipse major axis as approximate length, minor axis as average width
            # This is not accurate for taper, but provides *some* value related to non-uniform width.

            # Let's try relating aspect ratio of minAreaRect to taper
            aspect_ratio_rect = max(width, height) / min(width, height)
            # A high aspect ratio usually means less taper for a long tool.

            # Let's use the ellipse axes again, but maybe relate the difference to the major axis.
            # Taper angle is roughly related to how much the width changes per unit length.
            # Assume the change in diameter is (major_axis - minor_axis) over length major_axis
            # This isn't geometrically sound, but gives *an* indication.
            # angle ~ atan( (maj-min)/maj ) ?
            # Let's try the original calculation again but constrain it.
            if major_axis > minor_axis:
                 # Angle = atan ( (half_width_diff) / length )
                 # half_width_diff = (major_axis - minor_axis) / 2
                 # length = major_axis
                 # tan(angle) = (major_axis - minor_axis) / (2 * major_axis)
                 taper_angle_rad = math.atan((major_axis - minor_axis) / (2 * major_axis))
                 taper_angle_deg = math.degrees(taper_angle_rad)
                 # Constrain to reasonable values (0 to 20 degrees)
                 return round(max(0.0, min(20.0, taper_angle_deg)), 1)
            else:
                return 0.0

        except cv2.error:
             # fitEllipse might fail for lines etc.
             return 0.0 # No taper if it cannot fit ellipse
        except Exception as e:
            print(f"Error calculating taper: {e}")
            return 1.0 # Small default taper on error


    def draw_detection(self, image: np.ndarray, contour: np.ndarray,
                      tool_type: str, dimensions: Dict[str, float], item_id: int):
        """Draw detection results on the image"""
        if contour is None or len(contour) == 0 or dimensions is None:
            return

        try:
            color = self.colors.get(tool_type, self.colors['unknown'])

            # Draw contour
            cv2.drawContours(image, [contour], 0, color, self.line_thickness)

            # Get minimum area rectangle for better visualization
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.intp(box) # Use np.intp for points
            cv2.drawContours(image, [box], 0, color, 1) # Draw bounding box thinner

            # Get bounding rectangle for text placement (Axis-aligned)
            x, y, w, h = cv2.boundingRect(contour)

            # Format dimensions string based on tool type
            dim_str = self.format_dimensions(tool_type, dimensions)

            # --- Text Placement Logic ---
            # Place label above the contour
            label_y = y - 10
            # Ensure label is not drawn above image top
            if label_y < 10: label_y = y + 15

            # Draw tool type and ID
            cv2.putText(
                image, f"#{item_id}: {tool_type.replace('_', ' ').title()}",
                (x, label_y), self.font, self.font_scale, self.text_color, self.font_thickness, lineType=cv2.LINE_AA
            )

            # Draw dimensions below the contour
            y_offset = y + h + 20 # Initial offset below AA bounding box
            line_height = 18 # Estimated height per line of text
            img_h, img_w = image.shape[:2]

            lines = dim_str.split('\n')
            for i, line in enumerate(lines):
                 text_y = y_offset + i * line_height
                 # Basic check to prevent drawing off bottom
                 if text_y > img_h - 10:
                     text_y = img_h - 10 # Clamp near bottom

                 # Basic check to prevent drawing off right edge (adjust x if needed)
                 text_x = x
                 (text_width, _), _ = cv2.getTextSize(line, self.font, self.font_scale * 0.8, self.font_thickness)
                 if text_x + text_width > img_w - 10:
                     text_x = img_w - 10 - text_width # Shift left
                     if text_x < 10 : text_x = 10 # Don't go off left edge either


                 cv2.putText(
                    image, line,
                    (text_x, text_y), self.font, self.font_scale * 0.8, self.text_color, self.font_thickness, lineType=cv2.LINE_AA
                 )
        except Exception as e:
             print(f"Error drawing detection for tool #{item_id} ({tool_type}): {e}")


    def format_dimensions(self, tool_type: str, dimensions: Dict[str, float]) -> str:
        """Format dimensions string based on tool type, ensuring all keys are covered"""
        if not dimensions: return "Dims: N/A"

        try:
            # --- CHANGE: Explicitly format all expected dimensions for each type ---
            if tool_type == 'drill_guide':
                return (f"L: {dimensions.get('length', '?')}mm W: {dimensions.get('width', '?')}mm\n"
                        f"Hole Dia: {dimensions.get('hole_diameter', '?')}mm Spacing: {dimensions.get('hole_spacing', '?')}mm\n"
                        f"Num Holes: {int(dimensions.get('num_holes', '?'))} Area: {dimensions.get('area_mm2', '?')}mm2")
            elif tool_type == 'depth_gauge':
                return (f"L: {dimensions.get('length', '?')}mm W: {dimensions.get('width', '?')}mm\n"
                        f"Markings (est): {int(dimensions.get('markings_est', '?'))}\n"
                        f"Max Depth (est): {dimensions.get('max_depth_est', '?')}mm Area: {dimensions.get('area_mm2', '?')}mm2")
            elif tool_type == 'sizing_block':
                return (f"W: {dimensions.get('width', '?')}mm H: {dimensions.get('height', '?')}mm\n"
                        f"Thick (est): {dimensions.get('thickness_est', '?')}mm Diag: {dimensions.get('diagonal', '?')}mm\n"
                        f"Area: {dimensions.get('area_mm2', '?')}mm2")
            elif tool_type == 'alignment_rod':
                curvature = dimensions.get('radius_of_curvature', '?')
                curv_str = f"Curv Rad: {curvature}mm" if curvature != float('inf') and curvature != '?' else "Curv: Straight"
                return (f"Dia: {dimensions.get('diameter', '?')}mm L: {dimensions.get('length', '?')}mm\n"
                        f"{curv_str}\n"
                        f"Area: {dimensions.get('area_mm2', '?')}mm2")
            elif tool_type == 'tunnel_dilator':
                return (f"Outer Dia: {dimensions.get('outer_diameter', '?')}mm Inner Dia (est): {dimensions.get('inner_diameter_est', '?')}mm\n"
                        f"L: {dimensions.get('length', '?')}mm Taper (est): {dimensions.get('taper_angle_est', '?')}deg\n"
                        f"Num Holes: {int(dimensions.get('num_holes', '?'))} Area: {dimensions.get('area_mm2', '?')}mm2")
            else: # Unknown
                return (f"Max Dim: {dimensions.get('max_dimension', '?')}mm Min Dim: {dimensions.get('min_dimension', '?')}mm\n"
                        f"Area: {dimensions.get('area_mm2', '?')}mm2")
        except Exception as e:
            print(f"Error formatting dimensions for {tool_type}: {e}")
            return "Dims: Error"


    def auto_calibrate(self, image: np.ndarray, reference_tool_type: str = None) -> bool:
        """
        Attempt to automatically calibrate scale using a known tool from the image.

        Args:
            image: Input image
            reference_tool_type: Type of tool to use for calibration (optional)

        Returns:
            bool: True if calibration succeeded
        """
        print("Attempting auto-calibration...")
        # Use a temporary detector with default scale to find tools first
        temp_detector = OrthopedicToolDetector(model_path=self.model_path if hasattr(self, 'model_path') else None,
                                               pixel_to_mm=0.1) # Start with default guess
        _, tools = temp_detector.detect_tools(image)

        if not tools:
            print("Auto-calibration failed: No tools detected.")
            return False

        # Standard tool dimensions for calibration (in mm) - check these are realistic
        standard_dimensions = {
            'drill_guide': {'length': 120.0, 'width': 10.0, 'hole_diameter': 7.0},
            'depth_gauge': {'length': 150.0, 'width': 6.0},
            'alignment_rod': {'diameter': 3.0, 'length': 200.0}, # Adjusted diameter
            'tunnel_dilator': {'outer_diameter': 10.0, 'length': 130.0},
            'sizing_block': {'width': 50.0, 'height': 50.0} # Example size
        }

        reference_tool = None
        possible_refs = []

        # Find potential reference tools
        for tool in tools:
            tool_type = tool.get('type')
            if tool_type in standard_dimensions:
                 possible_refs.append(tool)

        if not possible_refs:
             print("Auto-calibration failed: No known tool types detected for reference.")
             return False

        # Select reference tool: specified type or largest reliable tool
        if reference_tool_type:
            for tool in possible_refs:
                if tool.get('type') == reference_tool_type:
                    reference_tool = tool
                    break
            if not reference_tool:
                print(f"Auto-calibration warning: Specified type '{reference_tool_type}' not found among detected known types. Trying largest.")
                reference_tool = max(possible_refs, key=lambda t: t.get('area_pixels', 0))
        else:
            # Use the largest detected tool of a known type
            reference_tool = max(possible_refs, key=lambda t: t.get('area_pixels', 0))

        if not reference_tool:
            print("Auto-calibration failed: Could not select a reference tool.")
            return False

        tool_type = reference_tool.get('type')
        dimensions_px = reference_tool.get('dimensions') # These are calculated with the *initial* scale (0.1)
        contour = np.array(reference_tool.get('contour')).astype(np.int32) # Need contour for pixel measurements

        if not dimensions_px or contour is None or len(contour) < 5:
             print(f"Auto-calibration failed: Missing data for reference tool {tool_type}.")
             return False

        print(f"Using detected '{tool_type}' (ID: {reference_tool.get('id')}) as reference.")

        # Get actual pixel measurements from the contour's bounding rectangle
        rect = cv2.minAreaRect(contour)
        _, (width_px, height_px), _ = rect
        width_px = abs(width_px)
        height_px = abs(height_px)
        max_dim_px = max(width_px, height_px)
        min_dim_px = min(width_px, height_px)

        std_dims = standard_dimensions[tool_type]
        new_pixel_to_mm = None

        # Try calibrating based on the most likely dimension
        if 'length' in std_dims and max_dim_px > 0:
            new_pixel_to_mm = std_dims['length'] / max_dim_px
            print(f"  Calibrating based on length ({std_dims['length']}mm / {max_dim_px:.1f}px)")
        elif 'outer_diameter' in std_dims and max_dim_px > 0: # Assume outer diameter corresponds to max dim for dilator
             new_pixel_to_mm = std_dims['outer_diameter'] / max_dim_px
             print(f"  Calibrating based on outer diameter ({std_dims['outer_diameter']}mm / {max_dim_px:.1f}px)")
        elif 'diameter' in std_dims and min_dim_px > 0: # Assume diameter corresponds to min dim for rod
            new_pixel_to_mm = std_dims['diameter'] / min_dim_px
            print(f"  Calibrating based on diameter ({std_dims['diameter']}mm / {min_dim_px:.1f}px)")
        elif 'width' in std_dims and 'height' in std_dims and width_px > 0 and height_px > 0: # For sizing block
             # Average calibration from width and height
             cal_w = std_dims['width'] / width_px
             cal_h = std_dims['height'] / height_px
             new_pixel_to_mm = (cal_w + cal_h) / 2.0
             print(f"  Calibrating based on width/height (avg of {std_dims['width']}mm / {width_px:.1f}px and {std_dims['height']}mm / {height_px:.1f}px)")
        elif 'width' in std_dims and min_dim_px > 0: # Fallback using width (e.g., drill guide width)
             new_pixel_to_mm = std_dims['width'] / min_dim_px
             print(f"  Calibrating based on width ({std_dims['width']}mm / {min_dim_px:.1f}px)")


        if new_pixel_to_mm is not None and 0.001 < new_pixel_to_mm < 1.0: # Sanity check scale
            self.pixel_to_mm = new_pixel_to_mm
            print(f"Auto-calibration successful: 1 pixel = {self.pixel_to_mm:.4f} mm")
            return True
        else:
            print(f"Auto-calibration failed: Could not derive a valid scale from {tool_type}. Using default.")
            self.pixel_to_mm = 0.1 # Reset to default if failed
            return False

    def manual_calibrate(self, image: np.ndarray, reference_length_mm: float) -> bool:
        """
        Calibrate the pixel-to-mm conversion using a known reference length.
        This version simulates selection by using the longest dimension of the largest detected tool.

        Args:
            image: Input image containing a reference object
            reference_length_mm: Known length of the reference object in mm

        Returns:
            bool: True if calibration succeeded
        """
        print(f"Attempting manual calibration with reference length: {reference_length_mm} mm.")
        print("(Simulation: Using longest dimension of largest detected tool as reference)")

        # Use a temporary detector with default scale to find tools first
        temp_detector = OrthopedicToolDetector(model_path=self.model_path if hasattr(self, 'model_path') else None,
                                               pixel_to_mm=0.1) # Use default guess
        _, tools = temp_detector.detect_tools(image)

        if not tools:
            print("Manual calibration failed: No tools detected to use as reference.")
            return False

        # Find the largest tool based on area
        largest_tool = max(tools, key=lambda t: t.get('area_pixels', 0))
        contour = np.array(largest_tool.get('contour')).astype(np.int32)
        tool_type = largest_tool.get('type', 'unknown')

        if contour is None or len(contour) < 5:
             print("Manual calibration failed: Could not get contour of largest tool.")
             return False

        # Get its longest dimension in pixels using minAreaRect
        rect = cv2.minAreaRect(contour)
        _, (width_px, height_px), _ = rect
        max_dim_px = max(abs(width_px), abs(height_px))

        if max_dim_px > 0:
            new_pixel_to_mm = reference_length_mm / max_dim_px
            if 0.001 < new_pixel_to_mm < 1.0: # Sanity check
                self.pixel_to_mm = new_pixel_to_mm
                print(f"Manual calibration successful (using largest tool '{tool_type}' max dim {max_dim_px:.1f}px):")
                print(f"  1 pixel = {self.pixel_to_mm:.4f} mm")
                return True
            else:
                 print(f"Manual calibration failed: Derived scale ({new_pixel_to_mm:.4f}) out of bounds.")
                 return False
        else:
            print("Manual calibration failed: Largest tool has zero max dimension.")
            return False

    def save_results(self, results: List[Dict], output_path: str):
        """Save detection results to a JSON file"""
        # Ensure contours are lists for JSON
        serializable_results = []
        for result in results:
             # Make a copy to avoid modifying the original list elements
             res_copy = result.copy()
             if 'contour' in res_copy:
                 # Ensure it's a list of lists/tuples, not a numpy array
                 if isinstance(res_copy['contour'], np.ndarray):
                     res_copy['contour'] = res_copy['contour'].tolist()
             # Convert numpy floats/ints in dimensions to standard types
             if 'dimensions' in res_copy and isinstance(res_copy['dimensions'], dict):
                 res_copy['dimensions'] = {k: (float(v) if isinstance(v, (np.float32, np.float64)) else
                                                int(v) if isinstance(v, (np.int32, np.int64)) else
                                                v)
                                           for k, v in res_copy['dimensions'].items()}
             # Convert area
             if 'area_pixels' in res_copy and isinstance(res_copy['area_pixels'], np.floating):
                 res_copy['area_pixels'] = float(res_copy['area_pixels'])

             serializable_results.append(res_copy)

        try:
            with open(output_path, 'w') as f:
                json.dump(serializable_results, f, indent=2, allow_nan=False) # Disallow NaN for strict JSON
            print(f"Results saved to {output_path}")
        except TypeError as e:
            print(f"Error saving results: Data is not JSON serializable. {e}")
            # Attempt to save again, replacing problematic values
            def make_serializable(obj):
                if isinstance(obj, dict):
                    return {k: make_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_serializable(i) for i in obj]
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.float64, np.float32)):
                    # Handle NaN/Inf for JSON
                    if np.isnan(obj): return 'NaN'
                    if np.isinf(obj): return 'Infinity' if obj > 0 else '-Infinity'
                    return float(obj)
                elif isinstance(obj, (float, int, str, bool)) or obj is None:
                    return obj
                else:
                    # Replace unknown types with their string representation
                    print(f"Warning: Converting non-serializable type {type(obj)} to string.")
                    return str(obj)

            try:
                 cleaned_results = make_serializable(serializable_results)
                 with open(output_path, 'w') as f:
                     json.dump(cleaned_results, f, indent=2)
                 print(f"Results saved to {output_path} (with potential type conversions).")
            except Exception as e2:
                 print(f"Error saving results even after cleaning: {e2}")

        except Exception as e:
            print(f"Error saving results: {e}")


def analyze_orthopedic_tools(image_path: str, model_path: str = None,
                           output_path: str = "result.jpg",
                           json_path: str = "results.json",
                           reference_length_mm: float = None,
                           auto_calibrate: bool = True,
                           debug: bool = False):
    """
    Analyze orthopedic tools in an image with enhanced accuracy.

    Args:
        image_path: Path to input image
        model_path: Path to trained model (optional)
        output_path: Path to save annotated output image
        json_path: Path to save JSON results
        reference_length_mm: Reference length for manual calibration (mm)
        auto_calibrate: Whether to attempt automatic calibration
        debug: Enable debugging output

    Returns:
        tuple: (result_image, detected_tools)
    """
    # Load image
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None, []
    print(f"Image loaded successfully: {image.shape}")

    # Initialize detector with improved parameters
    detector = OrthopedicToolDetector(model_path=model_path, pixel_to_mm=0.1, debug_mode=debug) # Start with default scale

    # --- Calibration ---
    calibration_done = False
    # Priority to manual calibration if reference length is provided
    if reference_length_mm is not None:
        print("--- Manual Calibration ---")
        if detector.manual_calibrate(image, reference_length_mm):
            calibration_done = True
        else:
            print("Manual calibration failed. Proceeding with default or auto-calibration.")

    # Attempt auto-calibration if enabled and manual wasn't done/failed
    if not calibration_done and auto_calibrate:
        print("--- Auto Calibration ---")
        if detector.auto_calibrate(image):
             calibration_done = True
        else:
            print("Auto-calibration failed. Using default scale (0.1 px/mm).")
            # Ensure default scale is set if auto-cal failed
            detector.pixel_to_mm = 0.1

    if not calibration_done:
         print("--- Using Default Scale (0.1 px/mm) ---")
         detector.pixel_to_mm = 0.1


    # --- Detect tools with the (potentially) calibrated detector ---
    print("\n--- Detecting Tools ---")
    result_image, tools = detector.detect_tools(image) # This uses the final detector.pixel_to_mm

    # --- Save and Summarize ---
    if result_image is not None:
        try:
            cv2.imwrite(output_path, result_image)
            print(f"\nResult image saved to {output_path}")
        except Exception as e:
             print(f"Error saving result image: {e}")
    else:
        print("\nWarning: Result image is None, cannot save.")


    if json_path and tools:
        detector.save_results(tools, json_path)
    elif not tools:
         print("No tools detected, JSON file not saved.")

    # Print summary
    print(f"\n--- Detection Summary ---")
    print(f"Final Scale Used: 1 pixel = {detector.pixel_to_mm:.4f} mm")
    print(f"Detected {len(tools)} tools:")
    if tools:
        for tool in tools:
            print(f"\nTool #{tool.get('id', 'N/A')}: {tool.get('type', 'unknown').replace('_', ' ').title()}")
            dims = tool.get('dimensions', {})
            if dims:
                for dim, value in dims.items():
                    unit = "mm"
                    if "area" in dim: unit = "mm2"
                    elif "angle" in dim: unit = "deg"
                    elif "num" in dim or "count" in dim or "est" in dim : unit = "" # No unit for counts/estimates
                    elif "curvature" in dim and value == float('inf'): value = "Straight"; unit = ""

                    print(f"  {dim.replace('_', ' ').title()}: {value} {unit}")
            else:
                print("  Dimensions: N/A")
    print("-" * 25)


    return result_image, tools

def main():
    import argparse
    import time

    parser = argparse.ArgumentParser(description='Enhanced ACL/PCL Jig Set Tool Detector')
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('-m', '--model', help='Path to trained model (optional)')
    parser.add_argument('-o', '--output', default='result.jpg', help='Output image path')
    parser.add_argument('-j', '--json', default='results.json', help='Path to save JSON results')
    parser.add_argument('-c', '--calibrate', type=float, help='Reference length in mm for manual calibration')
    parser.add_argument('--no-auto-calibrate', action='store_true', help='Disable automatic calibration')
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug mode (saves intermediate images)')
    parser.add_argument('--no-display', action='store_true', help='Do not display the result image window')

    args = parser.parse_args()

    start_time = time.time()

    # Run the enhanced analyzer
    result_image, tools = analyze_orthopedic_tools(
        image_path=args.image_path,
        model_path=args.model,
        output_path=args.output,
        json_path=args.json,
        reference_length_mm=args.calibrate,
        auto_calibrate=not args.no_auto_calibrate,
        debug=args.debug
    )

    end_time = time.time()
    print(f"\nTotal analysis time: {end_time - start_time:.2f} seconds")

    # Display the result if available and not disabled
    if not args.no_display and result_image is not None:
        print("\nDisplaying result image. Press any key to close.")
        try:
             # Resize if image is very large for display
            h, w = result_image.shape[:2]
            max_h, max_w = 800, 1200 # Max display size
            scale = min(max_h/h, max_w/w, 1.0) # Calculate scale factor, don't scale up
            if scale < 1.0:
                 display_img = cv2.resize(result_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            else:
                 display_img = result_image

            cv2.imshow("Enhanced Tool Detection Result", display_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error displaying image: {e}")
    elif args.no_display:
         print("Image display skipped (--no-display).")
    elif result_image is None:
         print("Result image is None, cannot display.")


if __name__ == "__main__":
    main()

# --- END OF FILE tool_detector.py ---