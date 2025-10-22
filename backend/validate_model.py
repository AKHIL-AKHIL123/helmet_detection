#!/usr/bin/env python3
"""
Model Validation and Testing Script
Validates the trained helmet detection model and provides performance metrics
"""

import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
import json
from datetime import datetime

class ModelValidator:
    def __init__(self, model_path, dataset_path):
        self.model_path = model_path
        self.dataset_path = Path(dataset_path)
        self.model = None
        self.results = {}
        
    def load_model(self):
        """Load the trained YOLO model"""
        try:
            self.model = YOLO(self.model_path)
            print(f"✅ Model loaded successfully from {self.model_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def validate_on_test_set(self, confidence_threshold=0.5):
        """Validate model on test images"""
        if not self.model:
            print("❌ Model not loaded!")
            return None
            
        val_images = list((self.dataset_path / 'images' / 'val').glob('*.png'))
        if not val_images:
            print("❌ No validation images found!")
            return None
            
        print(f"🔍 Validating on {len(val_images)} images...")
        
        total_detections = 0
        correct_detections = 0
        false_positives = 0
        false_negatives = 0
        
        detection_results = []
        
        for img_path in val_images[:50]:  # Test on first 50 images
            # Run inference
            results = self.model(str(img_path), conf=confidence_threshold)
            result = results[0]
            
            # Get ground truth
            label_path = self.dataset_path / 'labels' / 'val' / f"{img_path.stem}.txt"
            ground_truth = self._load_ground_truth(label_path, result.orig_shape)
            
            # Get predictions
            predictions = self._extract_predictions(result)
            
            # Calculate metrics
            img_metrics = self._calculate_image_metrics(ground_truth, predictions)
            detection_results.append({
                'image': img_path.name,
                'gt_count': len(ground_truth),
                'pred_count': len(predictions),
                'correct': img_metrics['correct'],
                'fp': img_metrics['fp'],
                'fn': img_metrics['fn']
            })
            
            total_detections += len(predictions)
            correct_detections += img_metrics['correct']
            false_positives += img_metrics['fp']
            false_negatives += img_metrics['fn']
        
        # Calculate overall metrics
        precision = correct_detections / (correct_detections + false_positives) if (correct_detections + false_positives) > 0 else 0
        recall = correct_detections / (correct_detections + false_negatives) if (correct_detections + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        self.results = {
            'total_images': len(val_images[:50]),
            'total_detections': total_detections,
            'correct_detections': correct_detections,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'confidence_threshold': confidence_threshold,
            'detailed_results': detection_results
        }
        
        return self.results
    
    def _load_ground_truth(self, label_path, img_shape):
        """Load ground truth annotations"""
        if not label_path.exists():
            return []
            
        height, width = img_shape[:2]
        ground_truth = []
        
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * width
                    y_center = float(parts[2]) * height
                    w = float(parts[3]) * width
                    h = float(parts[4]) * height
                    
                    x1 = int(x_center - w/2)
                    y1 = int(y_center - h/2)
                    x2 = int(x_center + w/2)
                    y2 = int(y_center + h/2)
                    
                    ground_truth.append({
                        'class': class_id,
                        'bbox': [x1, y1, x2, y2],
                        'label': 'helmet' if class_id == 0 else 'head'
                    })
        
        return ground_truth
    
    def _extract_predictions(self, result):
        """Extract predictions from YOLO result"""
        predictions = []
        
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                predictions.append({
                    'class': cls,
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2],
                    'label': result.names[cls]
                })
        
        return predictions
    
    def _calculate_image_metrics(self, ground_truth, predictions, iou_threshold=0.5):
        """Calculate metrics for a single image"""
        if not ground_truth and not predictions:
            return {'correct': 0, 'fp': 0, 'fn': 0}
        
        if not ground_truth:
            return {'correct': 0, 'fp': len(predictions), 'fn': 0}
        
        if not predictions:
            return {'correct': 0, 'fp': 0, 'fn': len(ground_truth)}
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(ground_truth), len(predictions)))
        for i, gt in enumerate(ground_truth):
            for j, pred in enumerate(predictions):
                if gt['class'] == pred['class']:  # Same class
                    iou_matrix[i, j] = self._calculate_iou(gt['bbox'], pred['bbox'])
        
        # Find matches
        matched_gt = set()
        matched_pred = set()
        correct = 0
        
        # Greedy matching based on highest IoU
        while True:
            max_iou = 0
            max_pos = None
            
            for i in range(len(ground_truth)):
                for j in range(len(predictions)):
                    if i not in matched_gt and j not in matched_pred and iou_matrix[i, j] > max_iou:
                        max_iou = iou_matrix[i, j]
                        max_pos = (i, j)
            
            if max_iou >= iou_threshold and max_pos:
                matched_gt.add(max_pos[0])
                matched_pred.add(max_pos[1])
                correct += 1
            else:
                break
        
        fp = len(predictions) - len(matched_pred)
        fn = len(ground_truth) - len(matched_gt)
        
        return {'correct': correct, 'fp': fp, 'fn': fn}
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def generate_report(self, save_path="validation_report.json"):
        """Generate and save validation report"""
        if not self.results:
            print("❌ No validation results available!")
            return
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_path': str(self.model_path),
            'dataset_path': str(self.dataset_path),
            'metrics': {
                'precision': f"{self.results['precision']:.3f}",
                'recall': f"{self.results['recall']:.3f}",
                'f1_score': f"{self.results['f1_score']:.3f}",
                'total_images': self.results['total_images'],
                'total_detections': self.results['total_detections'],
                'correct_detections': self.results['correct_detections'],
                'false_positives': self.results['false_positives'],
                'false_negatives': self.results['false_negatives']
            },
            'detailed_results': self.results['detailed_results']
        }
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Validation report saved to {save_path}")
        return report
    
    def print_summary(self):
        """Print validation summary"""
        if not self.results:
            print("❌ No validation results available!")
            return
        
        print("\n" + "="*50)
        print("🪖 HELMET DETECTION MODEL VALIDATION RESULTS")
        print("="*50)
        print(f"📊 Total Images Tested: {self.results['total_images']}")
        print(f"🎯 Total Detections: {self.results['total_detections']}")
        print(f"✅ Correct Detections: {self.results['correct_detections']}")
        print(f"❌ False Positives: {self.results['false_positives']}")
        print(f"⚠️  False Negatives: {self.results['false_negatives']}")
        print("-"*50)
        print(f"🎯 Precision: {self.results['precision']:.3f}")
        print(f"📈 Recall: {self.results['recall']:.3f}")
        print(f"⚖️  F1-Score: {self.results['f1_score']:.3f}")
        print("-"*50)
        
        # Performance assessment
        f1 = self.results['f1_score']
        if f1 >= 0.9:
            print("🌟 EXCELLENT: Model performance is outstanding!")
        elif f1 >= 0.8:
            print("✅ GOOD: Model performance is solid!")
        elif f1 >= 0.7:
            print("⚠️  FAIR: Model performance is acceptable but could be improved")
        else:
            print("❌ POOR: Model needs significant improvement")
        
        print("="*50)

def main():
    # Configuration
    model_path = "helmet_detection/yolov8s_helmet_detection/weights/best.pt"
    dataset_path = "yolo_dataset"
    
    print("🪖 Helmet Detection Model Validator")
    print("="*40)
    
    # Initialize validator
    validator = ModelValidator(model_path, dataset_path)
    
    # Load model
    if not validator.load_model():
        return
    
    # Run validation
    print("\n🔍 Running validation...")
    results = validator.validate_on_test_set(confidence_threshold=0.5)
    
    if results:
        # Print summary
        validator.print_summary()
        
        # Generate report
        validator.generate_report("validation_report.json")
        
        print(f"\n💡 Tip: Try different confidence thresholds to optimize performance!")
        print(f"💡 Current threshold: {results['confidence_threshold']}")
    else:
        print("❌ Validation failed!")

if __name__ == "__main__":
    main()