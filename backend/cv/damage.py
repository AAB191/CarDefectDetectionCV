from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import cv2
import numpy as np
from .dl import detect_with_yolo
from .cardd_detector import CarDDDetector


@dataclass
class DamageResult:
    annotated_bgr: np.ndarray
    severity: str
    num_regions: int
    boxes: List[Tuple[int, int, int, int]]
    damage_types: dict  # New field for specific damage types
    total_damage_area: float  # Total damaged area in pixels


def _classify_damage_types(contours, boxes, edge_map, bgr_image) -> dict:
    """Classify specific types of damage based on shape and texture analysis"""
    damage_types = {
        "cracks": 0,
        "dents": 0, 
        "scratches": 0,
        "paint_damage": 0
    }
    
    for i, contour in enumerate(contours):
        if i >= len(boxes):
            break
            
        x, y, bw, bh = boxes[i]
        area = cv2.contourArea(contour)
        
        # Analyze shape characteristics
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / (hull_area + 1e-6)
        
        # Aspect ratio analysis
        aspect_ratio = max(bw, bh) / min(bw, bh)
        
        # Edge density in the region
        patch_edges = edge_map[y:y+bh, x:x+bw]
        edge_density = float(patch_edges.sum()) / float(bw * bh)
        
        # Color analysis for paint damage
        patch = bgr_image[y:y+bh, x:x+bw]
        if patch.size > 0:
            # Check for color variation (paint damage often shows color changes)
            color_std = np.std(patch.reshape(-1, 3), axis=0)
            avg_color_std = np.mean(color_std)
        else:
            avg_color_std = 0
        
        # Classification logic
        if aspect_ratio > 5 and edge_density > 0.1:
            # Long and thin with high edge density = scratch
            damage_types["scratches"] += 1
        elif solidity > 0.7 and aspect_ratio < 3:
            # Round and solid = dent
            damage_types["dents"] += 1
        elif edge_density > 0.15 and avg_color_std > 30:
            # High edge density with color variation = crack
            damage_types["cracks"] += 1
        elif avg_color_std > 40:
            # High color variation = paint damage
            damage_types["paint_damage"] += 1
        else:
            # Default to scratch for unclear cases
            damage_types["scratches"] += 1
    
    return damage_types


def calculate_damage_area_from_regions(regions: List[Dict]) -> float:
    """Calculate total damage area from CarDD regions"""
    total_area = 0.0
    for region in regions:
        bbox = region['bbox']
        if len(bbox) >= 4:
            x1, y1, x2, y2 = bbox
            area = (x2 - x1) * (y2 - y1)
            total_area += area
    return total_area


def _classify_severity(num_regions: int, damaged_area_ratio: float, edge_density: float) -> str:
    # More conservative thresholds to reduce false positives
    # Only classify as damage if there are clear indicators
    
    # If no regions detected, it's minor/no damage
    if num_regions == 0:
        return "minor"
    
    # Calculate severity score with more conservative thresholds
    score = 0.0
    
    # Number of regions (weighted less heavily to avoid noise)
    region_score = min(1.0, num_regions / 8.0)  # Increased threshold from 5 to 8
    score += region_score * 0.3  # Reduced weight from 0.4 to 0.3
    
    # Damaged area ratio (most important factor)
    area_score = min(1.0, damaged_area_ratio / 0.08)  # Reduced threshold from 0.12 to 0.08
    score += area_score * 0.5  # Increased weight from 0.4 to 0.5
    
    # Edge density (least important, often noisy)
    edge_score = min(1.0, edge_density / 0.4)  # Increased threshold from 0.25 to 0.4
    score += edge_score * 0.2  # Kept same weight
    
    # Much more conservative classification thresholds
    if score >= 0.9:  # Very high threshold for severe
        return "severe"
    elif score >= 0.75:  # High threshold for moderate
        return "moderate"
    else:
        return "minor"


def detect_damage(image_bytes: bytes) -> DamageResult:
    # Try enhanced CarDD detector first (most accurate)
    try:
        cardd_detector = CarDDDetector()
        cardd_result = cardd_detector.detect_damage(image_bytes)
        if cardd_result and cardd_result['total_damage'] > 0:
            annotated_image = cardd_detector.annotate_image(image_bytes, cardd_result)
            return DamageResult(
                annotated_bgr=annotated_image,
                severity=cardd_result['severity'],
                num_regions=cardd_result['total_damage'],
                boxes=[region['bbox'] for region in cardd_result['regions']],
                damage_types=cardd_result['damage_types'],
                total_damage_area=calculate_damage_area_from_regions(cardd_result['regions'])
            )
    except Exception as e:
        pass
    
    # Try original YOLO model if CarDD fails
    try:
        dl = detect_with_yolo(image_bytes)
        if dl is not None and dl.num_regions > 0:
            return DamageResult(
                annotated_bgr=dl.annotated_bgr, 
                severity=dl.severity, 
                num_regions=dl.num_regions, 
                boxes=dl.boxes,
                damage_types={'cracks': 0, 'dents': 0, 'scratches': 0, 'paint_damage': 0},
                total_damage_area=0.0
            )
    except Exception:
        pass

    np_arr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if bgr is None:
        bgr = np.zeros((256, 256, 3), dtype=np.uint8)

    h, w = bgr.shape[:2]

    # Preprocess: denoise while preserving edges, normalize lighting
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, bb = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, bb])
    bgr_eq = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    smooth = cv2.bilateralFilter(bgr_eq, d=7, sigmaColor=60, sigmaSpace=60)

    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)

    # Gradient magnitude using Scharr (more robust than Sobel)
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    grad = cv2.magnitude(gx, gy)
    grad_norm = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Texture anomaly: difference between original and bilateral-smoothed image
    gray_orig = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    residual = cv2.absdiff(gray_orig, cv2.GaussianBlur(gray_orig, (9, 9), 0))
    residual = cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX)

    # MSER regions capture irregular blotches often present in dents/scratches
    mser = cv2.MSER_create()
    try:
        mser.setMinArea(max(60, int(0.0002 * h * w)))
        mser.setMaxArea(int(0.15 * h * w))
    except Exception:
        pass
    regions, _ = mser.detectRegions(gray)
    mser_mask = np.zeros((h, w), dtype=np.uint8)
    for pts in regions[:800]:
        cv2.fillPoly(mser_mask, [pts.reshape(-1, 1, 2)], 255)

    # More sensitive edge detection for scratches
    edges = cv2.Canny(grad_norm, 30, 100)  # Reduced thresholds for better scratch detection
    cues = cv2.addWeighted(residual, 0.5, grad_norm, 0.5, 0)
    _, cues_thr = cv2.threshold(cues, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    union = cv2.bitwise_or(cues_thr, mser_mask)
    union = cv2.bitwise_or(union, edges)
    
    # Additional scratch detection using Hough lines for linear damage
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    scratch_mask = np.zeros((h, w), dtype=np.uint8)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Only consider lines that are roughly horizontal or vertical (typical scratches)
            angle = abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
            if angle < 30 or angle > 150:  # Horizontal-ish
                cv2.line(scratch_mask, (x1, y1), (x2, y2), 255, 2)
            elif 60 < angle < 120:  # Vertical-ish
                cv2.line(scratch_mask, (x1, y1), (x2, y2), 255, 2)
    
    # Add scratch mask to union
    union = cv2.bitwise_or(union, scratch_mask)

    # Morphology to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    union = cv2.morphologyEx(union, cv2.MORPH_CLOSE, kernel, iterations=2)
    union = cv2.dilate(union, kernel, iterations=1)

    # Optional GrabCut refinement using union as probable foreground
    grab = np.where(union > 0, cv2.GC_PR_FGD, cv2.GC_BGD).astype(np.uint8)
    try:
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(bgr, grab, None, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_MASK)
        mask = np.where((grab == cv2.GC_FGD) | (grab == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except Exception:
        mask = union

    # Final components and contour analysis with stricter filtering
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: List[Tuple[int, int, int, int]] = []
    total_area = h * w
    damaged_area = 0.0
    edge_density = 0.0

    annotated = bgr.copy()

    edge_map = edges.astype(bool)
    
    # Sort contours by area (largest first) to prioritize significant damage
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        
        # More sensitive minimum area threshold for scratches
        min_area = max(100, 0.0005 * total_area)  # Reduced from 300 and 0.001
        if area < min_area:
            continue
            
        x, y, bw, bh = cv2.boundingRect(contour)
        
        # More sensitive minimum size requirements for thin scratches
        if bw < 3 or bh < 3:  # Further reduced from 5 to allow very thin scratches
            continue
            
        # More lenient aspect ratio filter for long scratches
        aspect_ratio = max(bw, bh) / min(bw, bh)
        if aspect_ratio > 100:  # Increased from 50 to allow very long scratches
            continue
            
        # More lenient solidity filter for irregular scratch shapes
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull) + 1e-6
        solidity = area / hull_area
        if solidity < 0.2:  # Reduced from 0.4
            continue
            
        # More sensitive edge content check
        patch_edges = edge_map[y:y+bh, x:x+bw]
        edge_ratio = float(patch_edges.sum()) / float(bw * bh)
        if edge_ratio < 0.02:  # Reduced from 0.05
            continue
            
        # More lenient damage region criteria
        if area > min_area * 1.2 or edge_ratio > 0.05:  # Reduced thresholds
            boxes.append((x, y, bw, bh))
            damaged_area += area
            edge_density += edge_ratio
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 0, 255), 2)

    damaged_ratio = float(damaged_area) / float(total_area) if total_area > 0 else 0.0
    avg_edge_density = (edge_density / max(1, len(boxes))) if boxes else 0.0
    
    # Classify specific damage types
    damage_types = _classify_damage_types(contours, boxes, edge_map, bgr)
    
    # Additional confidence check: if damage is very small, classify as minor
    if damaged_ratio < 0.005:  # Less than 0.5% of image
        severity = "minor"
    elif len(boxes) < 2 and damaged_ratio < 0.01:  # Single small region
        severity = "minor"
    else:
        severity = _classify_severity(len(boxes), damaged_ratio, avg_edge_density)

    # Create more detailed summary
    damage_summary = []
    if damage_types["cracks"] > 0:
        damage_summary.append(f"{damage_types['cracks']} crack{'s' if damage_types['cracks'] > 1 else ''}")
    if damage_types["dents"] > 0:
        damage_summary.append(f"{damage_types['dents']} dent{'s' if damage_types['dents'] > 1 else ''}")
    if damage_types["scratches"] > 0:
        damage_summary.append(f"{damage_types['scratches']} scratch{'es' if damage_types['scratches'] > 1 else ''}")
    if damage_types["paint_damage"] > 0:
        damage_summary.append(f"{damage_types['paint_damage']} paint damage area{'s' if damage_types['paint_damage'] > 1 else ''}")
    
    summary_text = f"Damage: {severity} | " + (", ".join(damage_summary) if damage_summary else "No specific damage detected")
    cv2.rectangle(annotated, (0, 0), (w, 40), (0, 0, 0), thickness=-1)
    cv2.putText(annotated, summary_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return DamageResult(
        annotated_bgr=annotated,
        severity=severity,
        num_regions=len(boxes),
        boxes=boxes,
        damage_types=damage_types,
        total_damage_area=damaged_area,
    )


