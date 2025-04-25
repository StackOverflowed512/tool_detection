# --- START OF FILE comprehensive_tool_detector.py ---

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
        Initialize the detector for a comprehensive set of ACL/PCL tools.

        Args:
            model_path: Path to trained ML model (optional - recommended for this many tools)
            pixel_to_mm: Conversion factor from pixels to millimeters
            debug_mode: Enable debugging output
        """
        # --- COMBINED Tool Names and Colors ---
        self.colors = {
            # Original Set
            'drill_guide': (0, 255, 0),       # Green
            'depth_gauge': (0, 165, 255),     # Orange
            'sizing_block': (255, 0, 0),      # Blue
            'alignment_rod': (255, 255, 0),   # Yellow (Changed from Red to avoid conflict)
            'tunnel_dilator': (255, 0, 255),  # Purple
            # Second Set
            'femoral_aimer': (0, 255, 255),    # Cyan
            'guide_wire': (0, 0, 255),        # Red
            'cannulated_reamer': (128, 0, 128),# Dark Purple
            'endobutton': (0, 128, 0),        # Dark Green
            # General
            'unknown': (128, 128, 128)         # Gray
        }

        # --- COMBINED Tool Classes ---
        self.classes = [
            'drill_guide', 'depth_gauge', 'sizing_block', 'alignment_rod', 'tunnel_dilator',
            'femoral_aimer', 'guide_wire', 'cannulated_reamer', 'endobutton', 'unknown'
        ]

        # --- COMBINED & ADJUSTED Detection Thresholds (Needs Tuning!) ---
        self.thresholds = {
            'min_area': 50,                   # Low for wires/endobuttons
            'circularity': 0.5,               # General baseline
            'aspect_ratio': {                 # Ranges cover all tools - refining happens in classification
                'guide_wire': (15.0, 200.0),
                'depth_gauge': (2.5, 20.0),   # Extended range
                'alignment_rod': (8.0, 100.0),# High AR, less than wire
                'cannulated_reamer': (3.0, 20.0),
                'drill_guide': (1.5, 6.0),
                'tunnel_dilator': (2.0, 15.0),
                'femoral_aimer': (1.2, 10.0), # Wide range
                'sizing_block': (0.8, 1.7),
                'endobutton': (1.0, 4.0),
            },
            'solidity': 0.70,                 # Lower general threshold allowing for complex shapes/flutes
            'hole_ratio': 0.02,               # Min hole ratio if holes are expected
            'elongation': {                   # Elongation (1 = max elongation) - Higher value means more elongated
                'guide_wire': 0.95,
                'alignment_rod': 0.90,
                'depth_gauge': 0.85,
                'cannulated_reamer': 0.7,
                'tunnel_dilator': 0.6,
                'drill_guide': 0.5,
                'femoral_aimer': 0.4,         # Can be less elongated
                'sizing_block': 0.1,
                'endobutton': 0.1
             }
        }

        # Visualization settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.font_thickness = 1
        self.text_color = (255, 255, 255)
        self.line_thickness = 2

        # Debugging & Calibration
        self.debug_mode = debug_mode
        self.pixel_to_mm = pixel_to_mm

        # Classification model
        self.model = None
        self.model_path = model_path
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("Info: No valid ML model path provided or model not found. Using rule-based classification.")
            print("      (Rule-based classification is less reliable for this many tool types).")


    def load_model(self, model_path: str):
        """Load a trained classification model"""
        try:
            self.model = joblib.load(model_path)
            print(f"ML Model loaded from {model_path}")
            # Basic check for compatibility
            if hasattr(self.model, 'classes_'):
                model_classes = list(self.model.classes_)
                if set(model_classes) != set(self.classes) - {'unknown'}: # Model shouldn't predict 'unknown' explicitly
                     print(f"Warning: Model classes {model_classes} may differ from code classes {self.classes}.")
            if hasattr(self.model, 'n_features_in_') and self.model.n_features_in_ != 12:
                 print(f"Warning: Model expects {self.model.n_features_in_} features, code extracts 12.")
                 # self.model = None # Optionally disable if features mismatch
        except Exception as e:
            print(f"Error loading model: {e}. Disabling ML model.")
            self.model = None

    def preprocess_image(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Preprocess the image. (Robust version kept)"""
        if image is None or image.size == 0: return None, None, None
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            denoised = cv2.fastNlMeansDenoising(enhanced, None, h=10, templateWindowSize=7, searchWindowSize=21)

            thresh_adapt = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 15, 3
            )
            _, thresh_otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            combined_thresh = cv2.bitwise_or(thresh_adapt, thresh_otsu) # Combine methods

            kernel_close = np.ones((5, 5), np.uint8)
            kernel_open = np.ones((3, 3), np.uint8)
            cleaned = cv2.morphologyEx(combined_thresh, cv2.MORPH_CLOSE, kernel_close, iterations=2)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open, iterations=1)

            edges = cv2.Canny(denoised, 50, 150) # Keep edge detection

            if self.debug_mode:
                cv2.imwrite('debug_gray.jpg', gray)
                cv2.imwrite('debug_denoised.jpg', denoised)
                cv2.imwrite('debug_combined_thresh.jpg', combined_thresh)
                cv2.imwrite('debug_cleaned.jpg', cleaned)
                cv2.imwrite('debug_edges.jpg', edges)

            return cleaned, edges, gray
        except Exception as e:
            print(f"Error during preprocessing: {e}")
            return None, None, None

    def detect_tools(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], List[Dict]]:
        """Detect and measure tools. (Includes overlap check for complex tools)"""
        if image is None: return None, []
        h, w = image.shape[:2]
        canvas_h = h + 150
        result_image = np.zeros((canvas_h, w, 3), dtype=np.uint8)
        result_image[:h, :w] = image.copy()

        mask, edges, gray = self.preprocess_image(image)
        if mask is None or gray is None: return result_image, []

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter initial contours by minimum size only
        potential_contours = []
        for contour in contours:
            if contour is not None and len(contour) >= 5:
                 area = cv2.contourArea(contour)
                 if area >= self.thresholds['min_area']:
                     potential_contours.append(contour)

        # Remove severe duplicates/internal noise contours
        unique_contours = self.remove_duplicate_contours(potential_contours, image.shape[:2])
        unique_contours = sorted(unique_contours, key=cv2.contourArea, reverse=True)

        results = []
        detected_tool_masks = np.zeros(image.shape[:2], dtype=np.uint8) # Mask of already detected areas

        for i, contour in enumerate(unique_contours):
            if contour is None or len(contour) < 5 : continue # Skip invalid contours post-deduplication
            current_area = cv2.contourArea(contour)
            if current_area < self.thresholds['min_area']: continue # Final area check

            # --- Overlap Check ---
            # Check if this contour significantly overlaps with an *already detected* tool area
            contour_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], 0, 255, -1)
            intersection = cv2.bitwise_and(contour_mask, detected_tool_masks)
            overlap_area = cv2.countNonZero(intersection)

            # If > 60% of this contour is already part of a detected tool, skip it (likely internal detail or noise)
            if overlap_area / current_area > 0.60:
                if self.debug_mode: print(f"Skipping contour {i} due to high overlap ({overlap_area/current_area:.1%}) with existing detections.")
                continue
            # --- End Overlap Check ---

            features = self.extract_features(contour, gray)
            if not features: continue # Skip if feature extraction failed

            tool_type = self.classify_tool(features, contour)

            # Skip small unknowns unless ML model said otherwise
            if tool_type == 'unknown' and current_area < self.thresholds['min_area'] * 5 and self.model is None:
                continue

            dimensions = self.measure_tool(contour, tool_type, image)

            if dimensions:
                 tool_id = len(results) + 1 # Simple sequential ID for detected tools

                 try: contour_list = contour.tolist()
                 except Exception: continue # Skip if contour isn't serializable

                 results.append({
                    'id': tool_id, 'type': tool_type, 'dimensions': dimensions,
                    'area_pixels': current_area, 'contour': contour_list
                 })

                 # Add this tool's mask to the detected area mask to prevent overlap in future iterations
                 detected_tool_masks = cv2.bitwise_or(detected_tool_masks, contour_mask)

                 self.draw_detection(result_image, contour, tool_type, dimensions, tool_id)

        # Add scale info
        cv2.putText(result_image, f"Scale: 1px = {self.pixel_to_mm:.4f} mm", (10, canvas_h - 60), self.font, self.font_scale, (255, 255, 255), self.font_thickness)
        cv2.putText(result_image, f"Tools Found: {len(results)}", (10, canvas_h - 30), self.font, self.font_scale, (255, 255, 255), self.font_thickness)

        return result_image, results

    def remove_duplicate_contours(self, contours: List[np.ndarray], image_shape: Tuple[int, int]) -> List[np.ndarray]:
        """Remove highly overlapping contours."""
        # Function remains the same as previously provided.
        if not contours: return []
        unique_contours = []
        h, w = image_shape
        contours.sort(key=cv2.contourArea, reverse=True) # Process larger first
        processed_mask = np.zeros((h, w), dtype=np.uint8) # Keep track of covered area

        for i, contour1 in enumerate(contours):
             if contour1 is None or len(contour1) < 3: continue
             area1 = cv2.contourArea(contour1)
             if area1 <= 0: continue

             # Create mask for current contour
             mask1 = np.zeros((h, w), dtype=np.uint8)
             cv2.drawContours(mask1, [contour1], 0, 255, -1)

             # Check overlap with *already processed* contours
             intersection = cv2.bitwise_and(mask1, processed_mask)
             overlap_area = cv2.countNonZero(intersection)

             # If the current contour significantly overlaps area already covered by larger contours, skip it
             if overlap_area / area1 > 0.7: # Threshold for skipping
                 continue

             # If it's not a significant overlap, add it to unique list and update processed mask
             unique_contours.append(contour1)
             processed_mask = cv2.bitwise_or(processed_mask, mask1)

        return unique_contours


    def extract_features(self, contour: np.ndarray, gray_image: np.ndarray) -> Optional[List[float]]:
        """Extract 12 features. (Robust version kept)"""
        # Function remains the same as previously provided.
        if contour is None or len(contour) < 5: return None
        features = [0.0] * 12
        try:
            area = cv2.contourArea(contour); perimeter = cv2.arcLength(contour, True)
            if area <= 0 or perimeter <= 0: return None
            circularity = 4 * np.pi * area / (perimeter**2)
            rect = cv2.minAreaRect(contour); _, (w, h), angle = rect; w, h = abs(w), abs(h)
            aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
            hull = cv2.convexHull(contour); hull_area = cv2.contourArea(hull)
            solidity = area / (hull_area + 1e-6)
            has_hole, hole_ratio, num_holes = self.detect_holes(contour, gray_image)
            moments = cv2.moments(contour); hu_moments = cv2.HuMoments(moments).flatten()
            mu20 = moments.get('mu20', 0); mu02 = moments.get('mu02', 0); mu11 = moments.get('mu11', 0)
            denom = (mu20 + mu02)**2
            elongation = min(((mu20 - mu02)**2 + 4*(mu11**2)) / (denom + 1e-6), 1.0) if denom > 1e-6 else 0.0

            features = [float(f) for f in [area, perimeter, circularity, aspect_ratio, solidity,
                                            has_hole, hole_ratio, num_holes, elongation,
                                            hu_moments[0], hu_moments[1], angle]]
            if any(math.isnan(f) or math.isinf(f) for f in features): return None
            return features
        except Exception: return None

    def detect_holes(self, contour: np.ndarray, gray_image: np.ndarray) -> Tuple[bool, float, int]:
        """Detect holes. (Robust version kept)"""
        # Function remains the same as previously provided.
        has_hole, hole_ratio, num_holes = False, 0.0, 0
        if contour is None or len(contour) < 3 or gray_image is None: return has_hole, hole_ratio, num_holes
        try:
            mask = np.zeros(gray_image.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], 0, 255, -1)
            cnt_inside, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            tool_area = cv2.contourArea(contour)
            if tool_area <= 0 or hier is None or len(hier) == 0: return has_hole, hole_ratio, num_holes
            for i, h in enumerate(hier[0]):
                if h[3] == 0: # Inner contour of the main contour
                    hole_c = cnt_inside[i]; hole_area = cv2.contourArea(hole_c)
                    if hole_area > max(5, tool_area * 0.002) and hole_area < tool_area * 0.8:
                        has_hole = True; hole_ratio += hole_area / tool_area; num_holes += 1
        except Exception: pass
        return has_hole, hole_ratio, num_holes

    # --- COMPREHENSIVE Rule-Based Classification (ML Model Recommended!) ---
    def classify_tool(self, features: List[float], contour: np.ndarray) -> str:
        """Classify using ML model if available, otherwise complex rule hierarchy."""
        if self.model is not None:
            try:
                features_array = np.array(features).reshape(1, -1)
                if features_array.shape[1] == getattr(self.model, 'n_features_in_', 12):
                    prediction = self.model.predict(features_array)[0]
                    if prediction in self.classes and prediction != 'unknown': # Trust model if valid class
                         return prediction
                    # else: Fall through to rules if model predicts unknown or invalid class
                # else: Fall through if feature mismatch
            except Exception as e:
                print(f"ML prediction failed: {e}. Falling back to rules.")

        # --- Rule-Based Hierarchy (Complex & Less Reliable than ML) ---
        try:
            if features is None or len(features) != 12: return 'unknown'
            area, perim, circ, ar, sol, has_hole, hr, num_h, elon, hu1, hu2, angle = features
            th = self.thresholds # Shorthand

            # 1. Guide Wire (Most Distinctive)
            if ar >= th['aspect_ratio']['guide_wire'][0] and elon >= th['elongation']['guide_wire']:
                # Extra check: very small min dimension?
                rect = cv2.minAreaRect(contour); _, (w,h),_ = rect
                if min(abs(w), abs(h)) < 15: # Heuristic pixel width limit
                     return 'guide_wire'

            # 2. Endobutton (Small, holey, low AR)
            if area < 1500 and th['aspect_ratio']['endobutton'][0] <= ar <= th['aspect_ratio']['endobutton'][1] and sol > 0.8 and has_hole > 0.5 and num_h >= 2:
                 return 'endobutton'

            # 3. Alignment Rod (High Elongation/AR OR High Circularity)
            if (ar >= th['aspect_ratio']['alignment_rod'][0] and elon >= th['elongation']['alignment_rod']) \
               or (circ > 0.75 and ar < 3.0): # Check for circular ends/views
                return 'alignment_rod'

            # 4. Depth Gauge (High AR, less than wire/rod, often holes)
            if th['aspect_ratio']['depth_gauge'][0] <= ar < th['aspect_ratio']['guide_wire'][0] and elon >= th['elongation']['depth_gauge']:
                 # Depth gauges often have holes or markings (lower solidity might indicate markings)
                 if has_hole > 0.5 or sol < 0.85:
                      return 'depth_gauge'

            # 5. Sizing Block (Low AR, High Solidity, No Holes usually)
            if th['aspect_ratio']['sizing_block'][0] <= ar <= th['aspect_ratio']['sizing_block'][1] and sol > 0.9 and not (has_hole > 0.5):
                 return 'sizing_block'

            # 6. Cannulated Reamer (Moderate AR/Elongation, Hole OR Low Solidity)
            if th['aspect_ratio']['cannulated_reamer'][0] <= ar <= th['aspect_ratio']['cannulated_reamer'][1] and elon >= th['elongation']['cannulated_reamer']:
                 if has_hole > 0.5 or sol < th['solidity']: # Check for hole OR lower solidity (flutes)
                      return 'cannulated_reamer'

            # 7. Drill Guide (Specific AR, Holes Expected)
            if th['aspect_ratio']['drill_guide'][0] <= ar <= th['aspect_ratio']['drill_guide'][1] and elon >= th['elongation']['drill_guide'] and sol > 0.75:
                 # Holes are typical
                 if has_hole > 0.5:
                     return 'drill_guide'

            # 8. Tunnel Dilator (Holes, moderate AR/Elon)
            if th['aspect_ratio']['tunnel_dilator'][0] <= ar <= th['aspect_ratio']['tunnel_dilator'][1] and elon >= th['elongation']['tunnel_dilator']:
                 if has_hole > 0.5 and hr > th['hole_ratio']:
                      return 'tunnel_dilator'

            # 9. Femoral Aimer (Complex Shape - Broad Fallback Rules)
            # Very difficult with rules. Catch larger objects not classified above.
            if area > 2000 and th['aspect_ratio']['femoral_aimer'][0] <= ar <= th['aspect_ratio']['femoral_aimer'][1] and elon >= th['elongation']['femoral_aimer']:
                  # This is weak. Might catch other large unknown objects.
                  return 'femoral_aimer'

        except Exception as e:
             print(f"Error during rule-based classification: {e}")

        return 'unknown' # Default if no rules match

    # --- COMPREHENSIVE Measurement Logic ---
    def measure_tool(self, contour: np.ndarray, tool_type: str, image: np.ndarray) -> Optional[Dict[str, float]]:
        """Measure dimensions for ALL defined tool types."""
        if contour is None or len(contour) < 5: return None
        try:
            rect = cv2.minAreaRect(contour); center, (w_px, h_px), angle = rect
            w_px, h_px = abs(w_px), abs(h_px)
            area_px = cv2.contourArea(contour)
            w_mm, h_mm = w_px * self.pixel_to_mm, h_px * self.pixel_to_mm
            area_mm2 = area_px * self.pixel_to_mm * self.pixel_to_mm
            max_dim_mm, min_dim_mm = max(w_mm, h_mm), min(w_mm, h_mm)

            dimensions = {'area_mm2': round(area_mm2, 2)} # Common dimension

            # --- Tool-Specific Measurements ---
            if tool_type == 'drill_guide':
                 hole_info = self.detect_drill_guide_holes(contour, image) # Specific func might be better
                 dimensions.update({
                    'length': round(max_dim_mm, 1), 'width': round(min_dim_mm, 1),
                    'hole_diameter_est': round(hole_info.get('diameter', 0) * self.pixel_to_mm, 1),
                    'hole_spacing_est': round(hole_info.get('spacing', 0) * self.pixel_to_mm, 1),
                    'num_holes_est': float(hole_info.get('count', 0))})
            elif tool_type == 'depth_gauge':
                 markings = self.detect_markings(contour, image) # Estimate
                 dimensions.update({
                    'length': round(max_dim_mm, 1), 'width': round(min_dim_mm, 1),
                    'markings_est': float(markings), 'max_depth_est': round(max_dim_mm * 0.95, 1)})
            elif tool_type == 'sizing_block':
                 dimensions.update({
                    'width': round(w_mm, 1), 'height': round(h_mm, 1), # Use rect dims directly
                    'thickness_est': round(min_dim_mm * 0.3, 1),
                    'diagonal': round(math.sqrt(w_mm**2 + h_mm**2), 1)})
            elif tool_type == 'alignment_rod':
                 curv_px = self.estimate_curvature(contour)
                 curv_mm = curv_px * self.pixel_to_mm if curv_px < 9999 else float('inf')
                 dimensions.update({
                    'length': round(max_dim_mm, 1), 'diameter': round(min_dim_mm, 1),
                    'radius_of_curvature': round(curv_mm, 1) if curv_mm != float('inf') else curv_mm })
            elif tool_type == 'tunnel_dilator':
                 hole_info = self.detect_holes_detailed(contour, image)
                 dimensions.update({
                    'length': round(max_dim_mm, 1), 'outer_diameter': round(max(w_mm,h_mm), 1), # Outer dim approx
                    'inner_diameter_est': round(hole_info.get('avg_diameter', 0) * self.pixel_to_mm, 1),
                    'taper_angle_est': round(self.calculate_taper(contour), 1),
                    'num_holes': float(hole_info.get('count', 0))})
            elif tool_type == 'femoral_aimer':
                 curv_px = self.estimate_curvature(contour)
                 curv_mm = curv_px * self.pixel_to_mm if curv_px < 9999 else float('inf')
                 dimensions.update({
                    'max_dimension': round(max_dim_mm, 1), 'min_dimension': round(min_dim_mm, 1),
                    'curvature_est': round(curv_mm, 1) if curv_mm != float('inf') else curv_mm })
                 # Add more complex measures here if needed (e.g., arm lengths)
            elif tool_type == 'guide_wire':
                 dimensions.update({
                    'length': round(max_dim_mm, 1), 'diameter': round(min_dim_mm, 2)}) # Higher precision diameter
            elif tool_type == 'cannulated_reamer':
                 hole_info = self.detect_holes_detailed(contour, image)
                 dimensions.update({
                    'length': round(max_dim_mm, 1), 'outer_diameter': round(min_dim_mm, 1),
                    'cannulation_diameter_est': round(hole_info.get('avg_diameter', 0) * self.pixel_to_mm, 1),
                    'taper_angle_est': round(self.calculate_taper(contour), 1)})
            elif tool_type == 'endobutton':
                 hole_info = self.detect_holes_detailed(contour, image)
                 dimensions.update({
                    'length': round(max_dim_mm, 1), 'width': round(min_dim_mm, 1),
                    'num_holes': float(hole_info.get('count', 0)),
                    'avg_hole_diameter': round(hole_info.get('avg_diameter', 0) * self.pixel_to_mm, 2),
                    'hole_spacing_est': round(self.estimate_hole_spacing(hole_info.get('hole_centers_px', []), contour) * self.pixel_to_mm, 1)})
            else: # Unknown
                 dimensions.update({
                    'max_dimension': round(max_dim_mm, 1), 'min_dimension': round(min_dim_mm, 1)})

            return dimensions
        except Exception as e:
             print(f"Error measuring tool ({tool_type}): {e}")
             return None

    # --- Helper Measurement Functions (Keep relevant ones) ---
    def detect_holes_detailed(self, contour: np.ndarray, image: np.ndarray) -> Dict:
        """Detailed hole analysis for Endobutton, Reamer, Dilator etc."""
        # Function remains the same as previously provided.
        details = {'count': 0, 'avg_diameter': 0.0, 'max_diameter': 0.0, 'min_diameter': 0.0, 'hole_centers_px': []}
        if contour is None or len(contour)<3 or image is None: return details
        try:
            mask = np.zeros(image.shape[:2], dtype=np.uint8); cv2.drawContours(mask, [contour], 0, 255, -1)
            cnt_in, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            area = cv2.contourArea(contour)
            if area <= 0 or hier is None or len(hier) == 0: return details
            holes = []
            for i, h in enumerate(hier[0]):
                if h[3] == 0:
                    h_cnt = cnt_in[i]; h_area = cv2.contourArea(h_cnt)
                    if h_area > max(3, area * 0.002) and h_area < area * 0.7:
                        (x, y), r = cv2.minEnclosingCircle(h_cnt)
                        if r > 0.5 and cv2.pointPolygonTest(contour, (int(x), int(y)), False) >= 0:
                            holes.append({'center': (x, y), 'radius': r})
            details['count'] = len(holes)
            if len(holes) > 0:
                diams = [2 * h['radius'] for h in holes]; details['avg_diameter'] = sum(diams) / len(holes)
                details['max_diameter'] = max(diams); details['min_diameter'] = min(diams)
                details['hole_centers_px'] = [h['center'] for h in holes]
        except Exception: pass
        return details

    def estimate_hole_spacing(self, centers_px: list, contour: np.ndarray) -> float:
         """Estimate spacing between holes, e.g., for Endobutton"""
         # Function remains the same as previously provided.
         if not centers_px or len(centers_px) < 2: return 0.0
         if len(centers_px) == 2:
             try: return math.dist(centers_px[0], centers_px[1])
             except Exception: return 0.0
         distances = []
         for i in range(len(centers_px)):
             for j in range(i + 1, len(centers_px)):
                 try: distances.append(math.dist(centers_px[i], centers_px[j]))
                 except Exception: continue
         if distances: return sum(distances) / len(distances) # Simple average for > 2 holes
         else: return 0.0

    def detect_drill_guide_holes(self, contour: np.ndarray, image: np.ndarray) -> Dict:
        """Specific detection for drill guide holes (might be redundant with detailed)."""
        # Function remains the same as previously provided, but could potentially be merged
        # into detect_holes_detailed if needed. Let's keep it separate for now.
        results = {'diameter': 0.0, 'spacing': 0.0, 'count': 0}
        if contour is None or len(contour) < 3 or image is None: return results
        try:
            mask = np.zeros(image.shape[:2], dtype=np.uint8); cv2.drawContours(mask, [contour], 0, 255, -1)
            cnt_in, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            tool_area = cv2.contourArea(contour); valid_holes = []
            if tool_area <=0 or hier is None or len(hier)==0: return results
            for i, h in enumerate(hier[0]):
                if h[3] == 0:
                    h_cnt = cnt_in[i]; area = cv2.contourArea(h_cnt)
                    if area > max(5, tool_area * 0.005) and area < tool_area * 0.5:
                        (x, y), r = cv2.minEnclosingCircle(h_cnt)
                        if r > 1.0: valid_holes.append({'center':(x,y), 'radius':r})
            results['count'] = len(valid_holes)
            if len(valid_holes) > 0:
                diams = [2*h['radius'] for h in valid_holes]; results['diameter'] = sum(diams)/len(diams)
                if len(valid_holes) > 1:
                    centers = [h['center'] for h in valid_holes]; distances = []
                    for i in range(len(centers)):
                        for j in range(i+1, len(centers)):
                            try: distances.append(math.dist(centers[i], centers[j]))
                            except Exception: continue
                    if distances: results['spacing'] = sum(distances)/len(distances)
        except Exception: pass
        return results


    def detect_markings(self, contour: np.ndarray, image: np.ndarray) -> int:
        """Estimate markings on depth gauge."""
        # Function remains the same as previously provided.
        if contour is None or len(contour) < 5: return 0
        try:
            rect = cv2.minAreaRect(contour); _, (w,h), _ = rect; max_dim_px = max(abs(w), abs(h))
            px_per_mark = 7.0 / self.pixel_to_mm if self.pixel_to_mm > 1e-5 else 70
            px_per_mark = max(5, px_per_mark) # Min 5 pixels apart
            num = int(max_dim_px / px_per_mark)
            return max(3, min(num, 30))
        except Exception: return 5

    def estimate_curvature(self, contour: np.ndarray) -> float:
        """Estimate curvature radius in pixels"""
        # Function remains the same as previously provided.
        if contour is None or len(contour) < 10: return 10000.0
        try:
            ellipse = cv2.fitEllipse(contour); _,(min_ax, maj_ax),_ = ellipse; min_ax,maj_ax = abs(min_ax),abs(maj_ax)
            if maj_ax < 1e-5 or min_ax < 1e-5: return 10000.0
            if maj_ax/min_ax > 15: return 10000.0 # Treat highly elliptical as straight
            a = maj_ax/2.0; b = min_ax/2.0
            if a > 1e-5: rad = (b**2)/a; return 10000.0 if rad<5.0 or rad>5000.0 else rad
            else: return 10000.0
        except cv2.error: return 10000.0
        except Exception: return 10000.0

    def calculate_taper(self, contour: np.ndarray) -> float:
        """Estimate taper angle in degrees"""
        # Function remains the same as previously provided.
        if contour is None or len(contour)<5: return 0.0
        try:
            ellipse = cv2.fitEllipse(contour); _,(min_ax, maj_ax),_ = ellipse; min_ax,maj_ax = abs(min_ax),abs(maj_ax)
            if maj_ax < 1e-5 or min_ax < 1e-5 or maj_ax <= min_ax: return 0.0
            if maj_ax > min_ax:
                rad = math.atan((maj_ax-min_ax)/(2*maj_ax)); deg = math.degrees(rad)
                return round(max(0.0, min(20.0, deg)), 1)
            else: return 0.0
        except cv2.error: return 0.0
        except Exception: return 0.0

    # --- COMPREHENSIVE Formatting ---
    def format_dimensions(self, tool_type: str, dimensions: Dict[str, float]) -> str:
        """Format dimensions string for ALL tool types."""
        if not dimensions: return "Dims: N/A"
        s = []
        try:
            # --- Format based on tool type ---
            if tool_type == 'guide_wire':
                 s.append(f"L: {dimensions.get('length', '?'):.1f}mm")
                 s.append(f"Dia: {dimensions.get('diameter', '?'):.2f}mm")
            elif tool_type == 'endobutton':
                 s.append(f"L: {dimensions.get('length', '?'):.1f}mm W: {dimensions.get('width', '?'):.1f}mm")
                 s.append(f"Holes: {int(dimensions.get('num_holes', 0))} Dia: {dimensions.get('avg_hole_diameter', '?'):.2f}mm")
                 s.append(f"Space(est): {dimensions.get('hole_spacing_est', '?'):.1f}mm")
            elif tool_type == 'cannulated_reamer':
                 s.append(f"L: {dimensions.get('length', '?'):.1f}mm ODia: {dimensions.get('outer_diameter', '?'):.1f}mm")
                 cann_d = dimensions.get('cannulation_diameter_est', 0)
                 cann_str = f"CannDia(est): {cann_d:.1f}mm" if cann_d > 0 else "Cann: N/A"
                 s.append(cann_str)
                 s.append(f"Taper(est): {dimensions.get('taper_angle_est', '?'):.1f}deg")
            elif tool_type == 'femoral_aimer':
                 s.append(f"MaxD: {dimensions.get('max_dimension', '?'):.1f}mm MinD: {dimensions.get('min_dimension', '?'):.1f}mm")
                 curv = dimensions.get('curvature_est', '?')
                 curv_str = f"Curv(est): {curv:.0f}mm" if curv != float('inf') and curv != '?' else "Curv: Straight"
                 s.append(curv_str)
            elif tool_type == 'drill_guide':
                 s.append(f"L: {dimensions.get('length', '?'):.1f}mm W: {dimensions.get('width', '?'):.1f}mm")
                 s.append(f"Holes(est): {int(dimensions.get('num_holes_est', 0))} Dia: {dimensions.get('hole_diameter_est', '?'):.1f}mm")
                 s.append(f"Space(est): {dimensions.get('hole_spacing_est', '?'):.1f}mm")
            elif tool_type == 'depth_gauge':
                 s.append(f"L: {dimensions.get('length', '?'):.1f}mm W: {dimensions.get('width', '?'):.1f}mm")
                 s.append(f"Marks(est): {int(dimensions.get('markings_est', 0))} MaxD(est): {dimensions.get('max_depth_est', '?'):.1f}mm")
            elif tool_type == 'sizing_block':
                 s.append(f"W: {dimensions.get('width', '?'):.1f}mm H: {dimensions.get('height', '?'):.1f}mm")
                 s.append(f"Thick(est): {dimensions.get('thickness_est', '?'):.1f}mm Diag: {dimensions.get('diagonal', '?'):.1f}mm")
            elif tool_type == 'alignment_rod':
                 curv = dimensions.get('radius_of_curvature', '?')
                 curv_str = f"Curv: {curv:.0f}mm" if curv != float('inf') and curv != '?' else "Curv: Straight"
                 s.append(f"L: {dimensions.get('length', '?'):.1f}mm Dia: {dimensions.get('diameter', '?'):.1f}mm")
                 s.append(curv_str)
            elif tool_type == 'tunnel_dilator':
                 s.append(f"L: {dimensions.get('length', '?'):.1f}mm ODia: {dimensions.get('outer_diameter', '?'):.1f}mm")
                 id_str = f"IDia(est): {dimensions.get('inner_diameter_est', '?'):.1f}mm"
                 s.append(id_str)
                 s.append(f"Taper(est): {dimensions.get('taper_angle_est', '?'):.1f}deg")
            else: # Unknown
                 s.append(f"MaxD: {dimensions.get('max_dimension', '?'):.1f}mm MinD: {dimensions.get('min_dimension', '?'):.1f}mm")
                 s.append(f"Area: {dimensions.get('area_mm2', '?'):.1f}mm2")

            # Join lines, limit total lines displayed?
            max_lines = 3
            return "\n".join(s[:max_lines])

        except Exception as e:
            print(f"Error formatting dimensions for {tool_type}: {e}")
            return "Dims: Error"


    def draw_detection(self, image: np.ndarray, contour: np.ndarray,
                      tool_type: str, dimensions: Dict[str, float], item_id: int):
        """Draw detection results."""
        # Function remains the same as previously provided.
        if contour is None or len(contour)==0 or dimensions is None: return
        try:
            color = self.colors.get(tool_type, self.colors['unknown'])
            cv2.drawContours(image, [contour], 0, color, self.line_thickness)
            rect = cv2.minAreaRect(contour); box = np.intp(cv2.boxPoints(rect))
            cv2.drawContours(image, [box], 0, color, 1) # Thin box outline
            x, y, w, h = cv2.boundingRect(contour)
            img_h, img_w = image.shape[:2]

            label_y = y - 10 if y > 20 else y + h + 15
            if label_y > img_h - 60 : label_y = y - 10
            if label_y < 10: label_y = 15

            cv2.putText(image, f"#{item_id}: {tool_type.replace('_', ' ').title()}",
                        (x, label_y), self.font, self.font_scale * 1.1, self.text_color, self.font_thickness + 1, lineType=cv2.LINE_AA)

            dim_str = self.format_dimensions(tool_type, dimensions)
            y_offset = label_y + 18; line_height = 15
            lines = dim_str.split('\n')
            for i, line in enumerate(lines):
                 text_y = y_offset + i * line_height
                 if text_y > img_h - 10: text_y = img_h - 10
                 if text_y < 10: text_y = 10
                 text_x = x
                 (text_width, _), _ = cv2.getTextSize(line, self.font, self.font_scale, self.font_thickness)
                 if text_x + text_width > img_w - 10: text_x = max(10, img_w - 10 - text_width)
                 cv2.putText(image, line, (text_x, text_y), self.font, self.font_scale, self.text_color, self.font_thickness, lineType=cv2.LINE_AA)
        except Exception as e: print(f"Error drawing detection for tool #{item_id} ({tool_type}): {e}")


    # --- COMPREHENSIVE Calibration Standards (MUST BE ADJUSTED BY USER) ---
    def auto_calibrate(self, image: np.ndarray, reference_tool_type: str = None) -> bool:
        """Attempt auto-calibration using combined standard dimensions."""
        print("Attempting auto-calibration...")
        # Use a temporary detector with default scale
        temp_detector = OrthopedicToolDetector(model_path=self.model_path, pixel_to_mm=0.1)
        _, tools = temp_detector.detect_tools(image)
        if not tools: print("Auto-calibration failed: No tools detected."); return False

        # --- ADJUST THESE STANDARD DIMENSIONS TO MATCH YOUR TOOLS ---
        standard_dimensions = {
            # Original Set
            'drill_guide': {'length': 120.0, 'width': 10.0, 'hole_diameter': 7.0},
            'depth_gauge': {'length': 150.0, 'width': 6.0},
            'sizing_block': {'width': 50.0, 'height': 50.0}, # Example
            'alignment_rod': {'diameter': 3.0, 'length': 200.0},
            'tunnel_dilator': {'outer_diameter': 10.0, 'length': 130.0},
             # Second Set
            'guide_wire': {'diameter': 1.6, 'length': 280.0}, # e.g., 1.6mm K-wire
            'endobutton': {'length': 12.0, 'width': 4.0},     # e.g., 12mm
            'cannulated_reamer': {'outer_diameter': 8.0, 'length': 100.0}, # e.g., 8mm head
            'femoral_aimer': {'max_dimension': 150.0}         # Less reliable, use overall size
        }
        print("Warning: Auto-calibration relies on standard dimensions defined in code. Verify these match your tools.")

        possible_refs = [t for t in tools if t.get('type') in standard_dimensions]
        if not possible_refs: print("Auto-calibration failed: No reference tool types detected."); return False

        reference_tool = None
        target_type = reference_tool_type if reference_tool_type in standard_dimensions else None
        if target_type:
            for tool in possible_refs:
                if tool.get('type') == target_type: reference_tool = tool; break
            if not reference_tool: print(f"Specified type '{target_type}' not found. Trying largest known type.")
        if not reference_tool: # If no target or target not found
             reference_tool = max(possible_refs, key=lambda t: t.get('area_pixels', 0))

        if not reference_tool: print("Auto-calibration failed: Could not select reference."); return False

        tool_type = reference_tool.get('type'); contour = np.array(reference_tool.get('contour')).astype(np.int32)
        print(f"Using detected '{tool_type}' (ID: {reference_tool.get('id')}) as reference.")

        rect = cv2.minAreaRect(contour); _, (w_px, h_px), _ = rect
        max_dim_px, min_dim_px = max(abs(w_px), abs(h_px)), min(abs(w_px), abs(h_px))
        std_dims = standard_dimensions[tool_type]
        new_pixel_to_mm = None

        # --- Combined Calibration Logic ---
        # Prioritize reliable dimensions like diameter or width/length for simple shapes
        cal_options = []
        if 'diameter' in std_dims and min_dim_px > 0: cal_options.append({'dim':'diameter', 'val':std_dims['diameter'] / min_dim_px})
        if 'outer_diameter' in std_dims: # Use min_dim for reamer OD, max_dim for dilator OD? Be specific.
             if tool_type == 'cannulated_reamer' and min_dim_px > 0: cal_options.append({'dim':'outer_diameter', 'val':std_dims['outer_diameter'] / min_dim_px})
             elif tool_type == 'tunnel_dilator' and max_dim_px > 0: cal_options.append({'dim':'outer_diameter', 'val':std_dims['outer_diameter'] / max_dim_px})
        if 'width' in std_dims and min_dim_px > 0: cal_options.append({'dim':'width', 'val':std_dims['width'] / min_dim_px})
        if 'length' in std_dims and max_dim_px > 0: cal_options.append({'dim':'length', 'val':std_dims['length'] / max_dim_px})
        if 'height' in std_dims and max_dim_px > 0: cal_options.append({'dim':'height', 'val':std_dims['height'] / max_dim_px}) # Often same as length
        if 'max_dimension' in std_dims and max_dim_px > 0: cal_options.append({'dim':'max_dimension', 'val':std_dims['max_dimension'] / max_dim_px})

        if not cal_options: print(f"No suitable dimension found for calibrating {tool_type}."); return False

        # Choose the best calibration option (e.g., prioritize diameter/width over length)
        best_cal = min(cal_options, key=lambda x: {'diameter':0, 'width':1, 'height': 2, 'outer_diameter': 3, 'length':4, 'max_dimension':5}.get(x['dim'], 99))
        new_pixel_to_mm = best_cal['val']
        print(f"  Calibrating using {tool_type} {best_cal['dim']} ({std_dims[best_cal['dim']]}mm / { (min_dim_px if 'diameter' in best_cal['dim'] or 'width' in best_cal['dim'] else max_dim_px):.1f}px)")

        if new_pixel_to_mm is not None and 0.001 < new_pixel_to_mm < 1.0: # Sanity check scale
            self.pixel_to_mm = new_pixel_to_mm
            print(f"Auto-calibration successful: 1 pixel = {self.pixel_to_mm:.4f} mm")
            return True
        else:
            print(f"Auto-calibration failed: Derived scale ({new_pixel_to_mm}) invalid. Using default.")
            self.pixel_to_mm = 0.1
            return False

    # --- Manual Calibrate & Save Results (Unchanged Structurally) ---
    def manual_calibrate(self, image: np.ndarray, reference_length_mm: float) -> bool:
        """Calibrate using simulated manual selection."""
        # Function remains the same as previously provided.
        print(f"Attempting manual calibration with reference length: {reference_length_mm} mm.")
        print("(Simulation: Using longest dimension of largest detected tool as reference)")
        temp_detector = OrthopedicToolDetector(model_path=self.model_path, pixel_to_mm=0.1)
        _, tools = temp_detector.detect_tools(image)
        if not tools: print("Manual calibration failed: No tools detected."); return False
        largest_tool = max(tools, key=lambda t: t.get('area_pixels', 0))
        contour = np.array(largest_tool.get('contour')).astype(np.int32); tool_type = largest_tool.get('type', 'unknown')
        if contour is None or len(contour) < 5: print("Manual calibration failed: Bad contour."); return False
        rect = cv2.minAreaRect(contour); _, (w_px, h_px), _ = rect; max_dim_px = max(abs(w_px), abs(h_px))
        if max_dim_px > 0:
            new_pixel_to_mm = reference_length_mm / max_dim_px
            if 0.001 < new_pixel_to_mm < 1.0:
                self.pixel_to_mm = new_pixel_to_mm
                print(f"Manual calibration successful (using '{tool_type}' max dim {max_dim_px:.1f}px): 1 pixel = {self.pixel_to_mm:.4f} mm")
                return True
            else: print(f"Manual calibration failed: Derived scale ({new_pixel_to_mm:.4f}) out of bounds.")
        else: print("Manual calibration failed: Largest tool has zero max dimension.")
        return False

    def save_results(self, results: List[Dict], output_path: str):
        """Save detection results to JSON."""
        # Function remains the same as previously provided.
        serializable = []
        for res in results:
            res_copy = res.copy()
            if 'contour' in res_copy and isinstance(res_copy['contour'], np.ndarray): res_copy['contour'] = res_copy['contour'].tolist()
            if 'dimensions' in res_copy and isinstance(res_copy['dimensions'], dict):
                 res_copy['dimensions'] = {k: (float(v) if isinstance(v, (np.float32, np.float64, np.floating)) else
                                                int(v) if isinstance(v, (np.int32, np.int64, np.integer)) else
                                                ('Inf' if math.isinf(v) else 'NaN' if math.isnan(v) else v) if isinstance(v, float) else # Handle Inf/NaN for JSON
                                                v)
                                           for k, v in res_copy['dimensions'].items()}
            if 'area_pixels' in res_copy and isinstance(res_copy['area_pixels'], np.floating): res_copy['area_pixels'] = float(res_copy['area_pixels'])
            serializable.append(res_copy)
        try:
            with open(output_path, 'w') as f: json.dump(serializable, f, indent=2)
            print(f"Results saved to {output_path}")
        except TypeError as e: print(f"Error saving results (TypeError): {e}. Check for non-serializable data (e.g., NaN without allow_nan).")
        except Exception as e: print(f"Error saving results: {e}")


# --- Main analysis function wrapper ---
def analyze_orthopedic_tools(image_path: str, model_path: str = None,
                           output_path: str = "result.jpg",
                           json_path: str = "results.json",
                           reference_length_mm: float = None,
                           auto_calibrate: bool = True,
                           debug: bool = False):
    """
    Analyze a comprehensive set of orthopedic tools in an image.
    """
    print(f"--- Starting Analysis for {os.path.basename(image_path)} ---")
    image = cv2.imread(image_path)
    if image is None: print(f"Error: Could not load image from {image_path}"); return None, []
    print(f"Image loaded ({image.shape[1]}x{image.shape[0]})")

    detector = OrthopedicToolDetector(model_path=model_path, pixel_to_mm=0.1, debug_mode=debug) # Start with default scale

    # Calibration
    calibrated = False
    if reference_length_mm is not None:
        print("--- Manual Calibration ---")
        if detector.manual_calibrate(image, reference_length_mm): calibrated = True
        else: print("Manual calibration failed.")
    if not calibrated and auto_calibrate:
        print("--- Auto Calibration ---")
        if detector.auto_calibrate(image): calibrated = True
        else: print("Auto-calibration failed.")
    if not calibrated:
         print("--- Using Default Scale (0.1 px/mm) ---")
         detector.pixel_to_mm = 0.1 # Ensure default if all else fails

    # Detection
    print("\n--- Detecting Tools ---")
    result_image, tools = detector.detect_tools(image)

    # Output
    if result_image is not None:
        try: cv2.imwrite(output_path, result_image); print(f"\nResult image saved to {output_path}")
        except Exception as e: print(f"Error saving result image: {e}")
    else: print("\nWarning: Result image is None.")
    if json_path and tools: detector.save_results(tools, json_path)
    elif not tools: print("No tools detected, JSON file not saved.")

    # Summary
    print(f"\n--- Detection Summary ---")
    print(f"Final Scale Used: 1 pixel = {detector.pixel_to_mm:.4f} mm")
    print(f"Detected {len(tools)} tools:")
    if tools:
        counts = {}
        for tool in tools:
            t_type = tool.get('type', 'unknown')
            counts[t_type] = counts.get(t_type, 0) + 1
            print(f"\nTool #{tool.get('id', '?')}: {t_type.replace('_', ' ').title()}")
            dims = tool.get('dimensions', {})
            if dims:
                for dim, value in dims.items():
                    unit = "mm"; fmt = ".1f"
                    if "area" in dim: unit = "mm2"; fmt=".1f"
                    elif "angle" in dim: unit = "deg"; fmt=".1f"
                    elif "num" in dim or "count" in dim or "markings" in dim : unit = ""; fmt=""
                    elif "diameter" in dim and ("guide_wire" in tool_type or "endobutton" in tool_type): fmt=".2f" # Higher precision for small diams
                    elif isinstance(value, float) and math.isinf(value): value = "Straight"; unit = ""

                    val_str = f"{value:{fmt}}" if fmt and isinstance(value, (float,int)) and not isinstance(value, bool) else str(value)
                    print(f"  {dim.replace('_', ' ').title()}: {val_str}{unit}")
            else: print("  Dimensions: N/A")
        print("\nTool Counts:")
        for t_type, count in counts.items():
            print(f"  - {t_type.replace('_',' ').title()}: {count}")
    print("-" * 30)

    return result_image, tools

# --- Main execution block ---
def main():
    import argparse
    import time

    parser = argparse.ArgumentParser(description='Comprehensive Orthopedic Tool Detector (ACL/PCL Jig Set + More)')
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('-m', '--model', help='Path to trained ML model (Recommended)')
    parser.add_argument('-o', '--output', default='result.jpg', help='Output image path')
    parser.add_argument('-j', '--json', default='results.json', help='Path to save JSON results')
    parser.add_argument('-c', '--calibrate', type=float, help='Reference length in mm for manual calibration simulation')
    parser.add_argument('--no-auto-calibrate', action='store_true', help='Disable automatic calibration')
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--no-display', action='store_true', help='Do not display the result image window')
    args = parser.parse_args()

    start_time = time.time()
    result_image, tools = analyze_orthopedic_tools(
        image_path=args.image_path, model_path=args.model, output_path=args.output,
        json_path=args.json, reference_length_mm=args.calibrate,
        auto_calibrate=not args.no_auto_calibrate, debug=args.debug)
    end_time = time.time()
    print(f"\nTotal analysis time: {end_time - start_time:.2f} seconds")

    if not args.no_display and result_image is not None:
        print("\nDisplaying result image. Press any key to close.")
        try:
            h, w = result_image.shape[:2]; max_h, max_w = 800, 1200
            scale = min(max_h/h, max_w/w, 1.0)
            disp_img = cv2.resize(result_image, None, fx=scale, fy=scale) if scale < 1.0 else result_image
            cv2.imshow("Comprehensive Tool Detection", disp_img)
            cv2.waitKey(0); cv2.destroyAllWindows()
        except Exception as e: print(f"Error displaying image: {e}")
    elif args.no_display: print("Image display skipped.")
    elif result_image is None: print("Result image is None, cannot display.")

if __name__ == "__main__":
    main()
# --- END OF FILE comprehensive_tool_detector.py ---