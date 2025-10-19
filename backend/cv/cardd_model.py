"""
CarDD Model Integration for Enhanced Damage Detection
Uses trained YOLO model on CarDD dataset for better accuracy
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("WARNING: Ultralytics not available. Install with: pip install ultralytics")

class CarDDModel:
    """Enhanced damage detection using CarDD-trained model"""
    
    def __init__(self, model_path: str = "models/cardd_best.pt"):
        self.model_path = model_path
        self.model = None
        self.class_names = [
            'dent',
            'scratch', 
            'crack',
            'paint_damage',
            'bumper_damage',
            'glass_damage'
        ]
        self.load_model()
    
    def load_model(self) -> bool:
        """Load the trained CarDD model"""
        if not ULTRALYTICS_AVAILABLE:
            print("ERROR: Ultralytics not available. Using fallback detection.")
            return False
            
        if not os.path.exists(self.model_path):
            print(f"ERROR: CarDD model not found at {self.model_path}")
            print("Please train the model first using setup_cardd.py")
            return False
        
        try:
            self.model = YOLO(self.model_path)
            print(f"SUCCESS: CarDD model loaded successfully from {self.model_path}")
            return True
        except Exception as e:
            print(f"ERROR: Failed to load CarDD model: {e}")
            return False
    
    def detect_damage(self, image_bytes: bytes) -> Optional[Dict]:
        """Detect damage using trained CarDD model"""
        if self.model is None:
            return None
        
        try:
            # Convert bytes to image
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return None
            
            # Run inference
            results = self.model(image, conf=0.3, iou=0.5)
            
            # Process results
            damage_regions = []
            damage_types = {
                'dent': 0,
                'scratch': 0,
                'crack': 0,
                'paint_damage': 0,
                'bumper_damage': 0,
                'glass_damage': 0
            }
            
            total_confidence = 0.0
            num_detections = 0
            
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        # Get class, confidence, and coordinates
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = self.class_names[class_id]
                        
                        # Only include high-confidence detections
                        if confidence > 0.3:
                            damage_types[class_name] += 1
                            total_confidence += confidence
                            num_detections += 1
                            
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            damage_regions.append({
                                'class': class_name,
                                'confidence': confidence,
                                'bbox': [int(x1), int(y1), int(x2), int(y2)]
                            })
            
            # Calculate severity based on damage analysis
            severity = self._calculate_severity(damage_types, total_confidence, num_detections)
            
            return {
                'severity': severity,
                'damage_types': damage_types,
                'regions': damage_regions,
                'total_damage': sum(damage_types.values()),
                'avg_confidence': total_confidence / max(num_detections, 1),
                'num_detections': num_detections
            }
            
        except Exception as e:
            print(f"ERROR: Error in CarDD detection: {e}")
            return None
    
    def _calculate_severity(self, damage_types: Dict, total_confidence: float, num_detections: int) -> str:
        """Calculate damage severity based on CarDD analysis"""
        
        # Count different types of damage
        total_damage = sum(damage_types.values())
        
        if total_damage == 0:
            return "minor"
        
        # Weight different damage types
        severity_score = 0
        
        # Critical damage types (higher weight)
        critical_damage = damage_types['crack'] + damage_types['glass_damage']
        severity_score += critical_damage * 3
        
        # Major damage types
        major_damage = damage_types['dent'] + damage_types['bumper_damage']
        severity_score += major_damage * 2
        
        # Minor damage types
        minor_damage = damage_types['scratch'] + damage_types['paint_damage']
        severity_score += minor_damage * 1
        
        # Factor in confidence
        avg_confidence = total_confidence / max(num_detections, 1)
        confidence_factor = avg_confidence * 2
        
        # Final severity calculation
        final_score = severity_score + confidence_factor
        
        if final_score >= 8:
            return "severe"
        elif final_score >= 4:
            return "moderate"
        else:
            return "minor"
    
    def annotate_image(self, image_bytes: bytes, detection_result: Dict) -> np.ndarray:
        """Annotate image with CarDD detection results"""
        try:
            # Convert bytes to image
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return np.zeros((100, 100, 3), dtype=np.uint8)
            
            annotated = image.copy()
            
            # Color mapping for different damage types
            colors = {
                'dent': (0, 0, 255),           # Red
                'scratch': (0, 255, 0),        # Green
                'crack': (255, 0, 0),          # Blue
                'paint_damage': (0, 255, 255), # Yellow
                'bumper_damage': (255, 0, 255), # Magenta
                'glass_damage': (255, 255, 0)  # Cyan
            }
            
            # Draw bounding boxes and labels
            for region in detection_result['regions']:
                x1, y1, x2, y2 = region['bbox']
                class_name = region['class']
                confidence = region['confidence']
                color = colors.get(class_name, (128, 128, 128))
                
                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Background for label
                cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                
                # Text
                cv2.putText(annotated, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add summary text
            summary = f"CarDD: {detection_result['severity']} | {detection_result['total_damage']} regions"
            cv2.rectangle(annotated, (0, 0), (len(summary) * 12, 40), (0, 0, 0), -1)
            cv2.putText(annotated, summary, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return annotated
            
        except Exception as e:
            print(f"ERROR: Error annotating image: {e}")
            return image if 'image' in locals() else np.zeros((100, 100, 3), dtype=np.uint8)

def create_cardd_model(model_path: str = "models/cardd_best.pt") -> CarDDModel:
    """Factory function to create CarDD model instance"""
    return CarDDModel(model_path)

# Example usage
if __name__ == "__main__":
    # Test the CarDD model
    model = CarDDModel()
    
    if model.model is not None:
        print("SUCCESS: CarDD model is ready for damage detection")
    else:
        print("WARNING: CarDD model not available. Using fallback detection.")
