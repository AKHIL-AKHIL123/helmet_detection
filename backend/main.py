#!/usr/bin/env python3
"""
Helmet Detection FastAPI Backend
Automatically finds and loads the latest trained model
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil
import os
import uuid
import glob
from datetime import datetime
from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelSwitchRequest(BaseModel):
    model_name: str

class ModelManager:
    """Manages model loading and selection"""
    
    def __init__(self, model_base_path: str = "helmet_detection"):
        self.model_base_path = Path(model_base_path)
        self.model = None
        self.model_info = {}
        
    def find_latest_model(self) -> Optional[Path]:
        """Find the latest trained model"""
        if not self.model_base_path.exists():
            logger.warning(f"Model directory not found: {self.model_base_path}")
            return None
        
        # Look for model directories
        model_dirs = []
        for item in self.model_base_path.iterdir():
            if item.is_dir() and (item / "weights" / "best.pt").exists():
                model_dirs.append(item)
        
        if not model_dirs:
            logger.warning("No trained models found")
            return None
        
        # Sort by modification time (latest first)
        model_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest_model_dir = model_dirs[0]
        
        return latest_model_dir / "weights" / "best.pt"
    
    def load_model(self) -> bool:
        """Load the latest available model"""
        model_path = self.find_latest_model()
        
        if model_path is None:
            logger.error("No model found to load")
            return False
        
        try:
            logger.info(f"Loading model from: {model_path}")
            self.model = YOLO(str(model_path))
            
            # Store model info
            self.model_info = {
                "path": str(model_path),
                "loaded_at": datetime.now().isoformat(),
                "model_size": model_path.stat().st_size,
                "classes": self.model.names if hasattr(self.model, 'names') else {}
            }
            
            logger.info(f"Model loaded successfully: {self.model_info['classes']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        if self.model is None:
            return {"status": "no_model", "message": "No model loaded"}
        
        return {
            "status": "loaded",
            "info": self.model_info,
            "classes": self.model.names if hasattr(self.model, 'names') else {}
        }

# Initialize FastAPI app
app = FastAPI(
    title="Helmet Detection API",
    description="AI-powered helmet detection for safety compliance",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize model manager and load model
# Check if we're running from project root or backend directory
import os
if os.path.exists("backend/helmet_detection"):
    # Running from project root
    model_manager = ModelManager("backend/helmet_detection")
else:
    # Running from backend directory
    model_manager = ModelManager("helmet_detection")

model_loaded = model_manager.load_model()

@app.get("/")
async def read_root():
    """Root endpoint with API status"""
    model_info = model_manager.get_model_info()
    return {
        "message": "Helmet Detection API is running",
        "version": "2.0.0",
        "model_status": model_info["status"],
        "classes": model_info.get("classes", {})
    }

@app.get("/model/info")
async def get_model_info():
    """Get detailed model information"""
    return model_manager.get_model_info()

@app.get("/model/reload")
async def reload_model():
    """Reload the latest model"""
    success = model_manager.load_model()
    if success:
        return {"status": "success", "message": "Model reloaded successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload model")

@app.get("/models/available")
async def get_available_models():
    """Get list of available models"""
    available_models = []
    
    # Check for pre-trained YOLO models in current directory
    yolo_models = ["yolov8n.pt", "yolov8s.pt", "yolo11n.pt"]
    for model_file in yolo_models:
        if os.path.exists(model_file):
            available_models.append(model_file)
    
    # Check for trained models
    if model_manager.model_base_path.exists():
        for item in model_manager.model_base_path.iterdir():
            if item.is_dir() and (item / "weights" / "best.pt").exists():
                model_name = f"Custom: {item.name}"
                available_models.append(model_name)
    
    return {"models": available_models}

@app.post("/model/switch")
async def switch_model(request: ModelSwitchRequest):
    """Switch to a different model"""
    model_name = request.model_name
    if not model_name:
        raise HTTPException(status_code=400, detail="Model name is required")
    
    try:
        # Determine model path
        if model_name.startswith("Custom:"):
            # Custom trained model
            custom_name = model_name.replace("Custom: ", "")
            model_path = model_manager.model_base_path / custom_name / "weights" / "best.pt"
        else:
            # Pre-trained YOLO model
            model_path = Path(model_name)
        
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")
        
        # Load the new model
        logger.info(f"Switching to model: {model_path}")
        model_manager.model = YOLO(str(model_path))
        
        # Update model info
        model_manager.model_info = {
            "path": str(model_path),
            "loaded_at": datetime.now().isoformat(),
            "model_size": model_path.stat().st_size,
            "classes": model_manager.model.names if hasattr(model_manager.model, 'names') else {}
        }
        
        logger.info(f"Successfully switched to model: {model_name}")
        return {
            "status": "success", 
            "message": f"Successfully switched to {model_name}",
            "model_info": model_manager.model_info
        }
        
    except Exception as e:
        logger.error(f"Failed to switch model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to switch model: {str(e)}")

@app.post("/detect/")
async def detect_helmet(file: UploadFile = File(...)):
    """Detect helmets in uploaded image"""
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Check if model is loaded
    if model_manager.model is None:
        raise HTTPException(
            status_code=500, 
            detail="Model not loaded. Please train a model first or check model directory."
        )
    
    # Save uploaded file
    file_extension = os.path.splitext(file.filename)[1]
    filename = f"{uuid.uuid4()}{file_extension}"
    file_path = UPLOAD_DIR / filename
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Read and validate image
        img = cv2.imread(str(file_path))
        if img is None:
            raise HTTPException(status_code=400, detail="Could not read the image file")
        
        # Resize large images for faster processing
        height, width = img.shape[:2]
        max_size = 1280  # Maximum dimension
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img_resized = cv2.resize(img, (new_width, new_height))
            logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
        else:
            img_resized = img
        
        # Perform detection on resized image
        results = model_manager.model(img_resized, verbose=False)
        result = results[0]
        
        # Scale detection results back to original image size if resized
        if max(height, width) > max_size:
            scale_back = max(height, width) / max_size
            for box in result.boxes if result.boxes is not None else []:
                box.xyxy[0] *= scale_back
        
        # Process detections
        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = result.names[cls]
                
                detections.append({
                    "label": label,
                    "confidence": conf,
                    "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                })
        
        # Create visualization with precise bounding boxes
        img_with_boxes = img.copy()
        for det in detections:
            box = det["box"]
            label_text = f"{det['label']} {det['confidence']:.2f}"
            
            # Color coding: Green for helmet, Red for head/no helmet
            color = (0, 255, 0) if det['label'] == 'helmet' else (0, 0, 255)
            
            # Ensure coordinates are integers for precise drawing
            x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
            
            # Draw precise bounding box with thicker lines
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 3)
            
            # Draw corner markers for better visibility
            corner_size = 10
            # Top-left corner
            cv2.line(img_with_boxes, (x1, y1), (x1 + corner_size, y1), color, 3)
            cv2.line(img_with_boxes, (x1, y1), (x1, y1 + corner_size), color, 3)
            # Top-right corner
            cv2.line(img_with_boxes, (x2, y1), (x2 - corner_size, y1), color, 3)
            cv2.line(img_with_boxes, (x2, y1), (x2, y1 + corner_size), color, 3)
            # Bottom-left corner
            cv2.line(img_with_boxes, (x1, y2), (x1 + corner_size, y2), color, 3)
            cv2.line(img_with_boxes, (x1, y2), (x1, y2 - corner_size), color, 3)
            # Bottom-right corner
            cv2.line(img_with_boxes, (x2, y2), (x2 - corner_size, y2), color, 3)
            cv2.line(img_with_boxes, (x2, y2), (x2, y2 - corner_size), color, 3)
            
            # Draw label background with padding
            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_y = y1 - 10 if y1 - 10 > text_height else y2 + text_height + 10
            
            cv2.rectangle(img_with_boxes, 
                         (x1, label_y - text_height - 5), 
                         (x1 + text_width + 10, label_y + 5), 
                         color, -1)
            
            # Draw label text with better positioning
            cv2.putText(img_with_boxes, label_text, (x1 + 5, label_y - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save result image
        result_path = UPLOAD_DIR / f"result_{filename}"
        cv2.imwrite(str(result_path), img_with_boxes)
        
        # Calculate safety metrics
        helmet_count = sum(1 for det in detections if det['label'] == 'helmet')
        head_count = len(detections) - helmet_count
        compliance_rate = (helmet_count / len(detections) * 100) if detections else 0
        
        return {
            "status": "success",
            "detections": detections,
            "metrics": {
                "total_detections": len(detections),
                "helmet_count": helmet_count,
                "head_count": head_count,
                "compliance_rate": round(compliance_rate, 1)
            },
            "result_image": f"/results/result_{filename}",
            "model_info": {
                "classes": result.names,
                "model_path": model_manager.model_info.get("path", "unknown")
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
    
    finally:
        # Clean up uploaded file
        if file_path.exists():
            try:
                file_path.unlink()
            except:
                pass

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_manager.model is not None,
        "upload_dir": str(UPLOAD_DIR),
        "timestamp": datetime.now().isoformat()
    }

# Mount the uploads directory for serving static files
app.mount("/results", StaticFiles(directory=UPLOAD_DIR), name="results")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("🚀 Helmet Detection API starting up...")
    if model_loaded:
        logger.info("✅ Model loaded successfully")
    else:
        logger.warning("⚠️ No model loaded - train a model first")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )