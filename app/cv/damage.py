from dataclasses import dataclass
from typing import List, Tuple
import cv2
import numpy as np
from .dl import detect_with_yolo


@dataclass
class DamageResult:
    annotated_bgr: np.ndarray
    severity: str
    num_regions: int
    boxes: List[Tuple[int, int, int, int]]


def _classify_severity(num_regions: int, damaged_area_ratio: float, edge_density: float) -> str:
    # Combined cues: number of clusters, total damaged area, edge density inside mask
    score = 0.0
    score += min(1.0, num_regions / 5.0) * 0.4
    score += min(1.0, damaged_area_ratio / 0.12) * 0.4
    score += min(1.0, edge_density / 0.25) * 0.2
    if score >= 0.75:
        return "severe"
    if score >= 0.45:
        return "moderate"
    return "minor"


def detect_damage(image_bytes: bytes) -> DamageResult:
    # Try DL model first if available
    try:
        dl = detect_with_yolo(image_bytes)
        if dl is not None:
            return DamageResult(annotated_bgr=dl.annotated_bgr, severity=dl.severity, num_regions=dl.num_regions, boxes=dl.boxes)
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

    # Combine cues
    edges = cv2.Canny(grad_norm, 60, 160)
    cues = cv2.addWeighted(residual, 0.5, grad_norm, 0.5, 0)
    _, cues_thr = cv2.threshold(cues, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    union = cv2.bitwise_or(cues_thr, mser_mask)
    union = cv2.bitwise_or(union, edges)

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

    # Final components and contour analysis
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: List[Tuple[int, int, int, int]] = []
    total_area = h * w
    damaged_area = 0.0
    edge_density = 0.0

    annotated = bgr.copy()

    edge_map = edges.astype(bool)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < max(150, 0.0005 * total_area):
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < 10 or bh < 10:
            continue
        # Solidity filter to drop very thin false positives
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull) + 1e-6
        solidity = area / hull_area
        if solidity < 0.25:
            continue
        boxes.append((x, y, bw, bh))
        damaged_area += area
        patch_edges = edge_map[y:y+bh, x:x+bw]
        edge_density += float(patch_edges.sum()) / float(bw * bh)
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 0, 255), 2)

    damaged_ratio = float(damaged_area) / float(total_area) if total_area > 0 else 0.0
    avg_edge_density = (edge_density / max(1, len(boxes))) if boxes else 0.0
    severity = _classify_severity(len(boxes), damaged_ratio, avg_edge_density)

    summary = f"Damage: {severity} | regions: {len(boxes)}"
    cv2.rectangle(annotated, (0, 0), (w, 40), (0, 0, 0), thickness=-1)
    cv2.putText(annotated, summary, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return DamageResult(
        annotated_bgr=annotated,
        severity=severity,
        num_regions=len(boxes),
        boxes=boxes,
    )


