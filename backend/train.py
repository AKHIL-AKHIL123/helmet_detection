import os
import sys
import xml.etree.ElementTree as ET
import shutil
import yaml
from pathlib import Path
import random
import json
from datetime import datetime
import numpy as np
import torch
from ultralytics import YOLO
import cv2
import argparse
from typing import Dict, List, Tuple, Optional

class GPUChecker:
    """GPU availability and compatibility checker"""
    
    @staticmethod
    def check_gpu_availability() -> bool:
        """Check if GPU is available and compatible"""
        if not torch.cuda.is_available():
            return False
        
        device_count = torch.cuda.device_count()
        if device_count == 0:
            return False
        
        # Check GPU memory
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_gb = gpu_memory / (1024**3)
            return gpu_memory_gb >= 2.0  # Minimum 2GB GPU memory
        except:
            return False
    
    @staticmethod
    def get_gpu_info() -> Dict:
        """Get detailed GPU information"""
        if not torch.cuda.is_available():
            return {"available": False, "reason": "CUDA not available"}
        
        try:
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            device_props = torch.cuda.get_device_properties(current_device)
            
            return {
                "available": True,
                "device_count": device_count,
                "current_device": current_device,
                "device_name": device_name,
                "total_memory_gb": device_props.total_memory / (1024**3),
                "compute_capability": f"{device_props.major}.{device_props.minor}"
            }
        except Exception as e:
            return {"available": False, "reason": f"GPU info error: {str(e)}"}

class DatasetProcessor:
    """Handles dataset conversion and preparation"""
    
    def __init__(self, source_path: str, output_path: str, class_mapping: Dict[str, int]):
        self.source_path = Path(source_path)
        self.output_path = Path(output_path)
        self.class_mapping = class_mapping
        
    def analyze_dataset(self) -> Dict:
        """Analyze dataset and return statistics"""
        print("📊 Analyzing dataset...")
        
        img_dir = self.source_path / 'images'
        ann_dir = self.source_path / 'annotations'
        
        img_files = list(img_dir.glob('*.png'))
        ann_files = list(ann_dir.glob('*.xml'))
        
        if not img_files:
            raise ValueError(f"No images found in {img_dir}")
        
        print(f"📁 Found {len(img_files)} images")
        print(f"📄 Found {len(ann_files)} annotations")
        
        # Analyze annotations
        class_counts = {name: 0 for name in self.class_mapping.keys()}
        bbox_sizes = []
        image_sizes = []
        
        for ann_file in ann_files[:100]:  # Sample for speed
            try:
                stats = self._analyze_annotation(ann_file)
                for cls_name, count in stats['class_counts'].items():
                    if cls_name in class_counts:
                        class_counts[cls_name] += count
                
                bbox_sizes.extend(stats['bbox_sizes'])
                if stats['image_size']:
                    image_sizes.append(stats['image_size'])
                    
            except Exception as e:
                print(f"⚠️ Error processing {ann_file}: {e}")
        
        # Print statistics
        print(f"\n📈 Dataset Statistics:")
        for cls_name, count in class_counts.items():
            print(f"   {cls_name.title()} objects: {count}")
        print(f"   Total objects: {sum(class_counts.values())}")
        
        if image_sizes:
            avg_width = np.mean([s[0] for s in image_sizes])
            avg_height = np.mean([s[1] for s in image_sizes])
            print(f"   Average image size: {avg_width:.0f}x{avg_height:.0f}")
        
        if bbox_sizes:
            avg_bbox_width = np.mean([s[0] for s in bbox_sizes])
            avg_bbox_height = np.mean([s[1] for s in bbox_sizes])
            print(f"   Average bbox size: {avg_bbox_width:.0f}x{avg_bbox_height:.0f}")
        
        return {
            'total_images': len(img_files),
            'total_annotations': len(ann_files),
            'class_counts': class_counts,
            'image_sizes': image_sizes,
            'bbox_sizes': bbox_sizes
        }
    
    def _analyze_annotation(self, xml_path: Path) -> Dict:
        """Analyze single annotation file"""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        class_counts = {name: 0 for name in self.class_mapping.keys()}
        bbox_sizes = []
        image_size = None
        
        # Get image size
        size_elem = root.find('size')
        if size_elem is not None:
            width = int(size_elem.find('width').text)
            height = int(size_elem.find('height').text)
            image_size = (width, height)
        
        # Count objects and bbox sizes
        for obj in root.findall('object'):
            cls_name = self._map_class_name(obj.find('name').text)
            if cls_name in class_counts:
                class_counts[cls_name] += 1
            
            bbox = obj.find('bndbox')
            if bbox is not None:
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                
                bbox_width = xmax - xmin
                bbox_height = ymax - ymin
                if bbox_width > 0 and bbox_height > 0:
                    bbox_sizes.append((bbox_width, bbox_height))
        
        return {
            'class_counts': class_counts,
            'bbox_sizes': bbox_sizes,
            'image_size': image_size
        }
    
    def _map_class_name(self, original_name: str) -> str:
        """Map original class name to standardized name"""
        name_lower = original_name.lower()
        if 'with helmet' in name_lower:
            return 'helmet'
        elif 'without helmet' in name_lower:
            return 'head'
        elif 'helmet' in name_lower:
            return 'helmet'
        else:
            return 'head'
    
    def convert_voc_to_yolo(self, xml_path: Path, img_width: int, img_height: int) -> str:
        """Convert VOC XML to YOLO format"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            boxes = []
            for obj in root.findall('object'):
                cls_name = self._map_class_name(obj.find('name').text)
                class_id = self.class_mapping.get(cls_name, 0)
                
                bbox = obj.find('bndbox')
                if bbox is None:
                    continue
                
                # Extract coordinates
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                
                # Validate bbox
                if xmax <= xmin or ymax <= ymin:
                    continue
                
                # Convert to YOLO format (normalized)
                x_center = ((xmin + xmax) / 2) / img_width
                y_center = ((ymin + ymax) / 2) / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height
                
                # Clamp values to [0, 1]
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                boxes.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            return '\n'.join(boxes)
            
        except Exception as e:
            print(f"⚠️ Error converting {xml_path}: {e}")
            return ""
    
    def prepare_dataset(self, train_split: float = 0.8) -> str:
        """Prepare YOLO format dataset"""
        print("🔄 Preparing YOLO dataset...")
        
        # Create directory structure
        for split in ['train', 'val']:
            (self.output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        img_dir = self.source_path / 'images'
        img_files = list(img_dir.glob('*.png'))
        
        if not img_files:
            raise ValueError(f"No images found in {img_dir}")
        
        # Shuffle and split
        random.seed(42)
        random.shuffle(img_files)
        
        split_idx = int(train_split * len(img_files))
        train_files = img_files[:split_idx]
        val_files = img_files[split_idx:]
        
        print(f"📊 Dataset split: {len(train_files)} train, {len(val_files)} val")
        
        # Process splits
        for split_name, files in [('train', train_files), ('val', val_files)]:
            self._process_split(split_name, files)
        
        # Create dataset YAML
        yaml_path = self._create_dataset_yaml()
        print(f"📝 Dataset YAML saved to {yaml_path}")
        
        return str(yaml_path)
    
    def _process_split(self, split_name: str, files: List[Path]):
        """Process a single dataset split"""
        print(f"Processing {split_name} set...")
        processed = 0
        
        for img_file in files:
            try:
                # Copy image
                dst_img = self.output_path / 'images' / split_name / img_file.name
                shutil.copy2(img_file, dst_img)
                
                # Process annotation
                xml_file = self.source_path / 'annotations' / f"{img_file.stem}.xml"
                if xml_file.exists():
                    # Get image dimensions
                    img = cv2.imread(str(img_file))
                    if img is None:
                        continue
                    height, width = img.shape[:2]
                    
                    # Convert annotation
                    yolo_ann = self.convert_voc_to_yolo(xml_file, width, height)
                    
                    # Save YOLO annotation
                    if yolo_ann.strip():
                        label_file = self.output_path / 'labels' / split_name / f"{img_file.stem}.txt"
                        with open(label_file, 'w') as f:
                            f.write(yolo_ann)
                    
                    processed += 1
                    
            except Exception as e:
                print(f"⚠️ Error processing {img_file}: {e}")
        
        print(f"✅ Processed {processed}/{len(files)} {split_name} files")
    
    def _create_dataset_yaml(self) -> Path:
        """Create dataset YAML configuration"""
        data_yaml = {
            'path': str(self.output_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'names': {v: k for k, v in self.class_mapping.items()},
            'nc': len(self.class_mapping)
        }
        
        yaml_path = self.output_path / 'helmet_dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        return yaml_path

class HelmetTrainer:
    """Main training class with STRICT GPU enforcement"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.training_results = {}
        self.gpu_monitoring = True
        
        # Initialize dataset processor
        self.dataset_processor = DatasetProcessor(
            source_path=config['dataset']['source_path'],
            output_path=config['dataset']['output_dir'],
            class_mapping=config['classes']
        )
    
    def _continuous_gpu_monitor(self):
        """Continuously monitor GPU usage during training"""
        import threading
        import time
        
        def monitor():
            while self.gpu_monitoring:
                try:
                    if torch.cuda.is_available():
                        memory_used = torch.cuda.memory_allocated(0)
                        if memory_used < 50 * 1024**2:  # Less than 50MB
                            print("⚠️ WARNING: Very low GPU memory usage detected!")
                            print(f"   Current GPU memory: {memory_used / 1024**2:.1f} MB")
                    time.sleep(30)  # Check every 30 seconds
                except:
                    break
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
    
    def train(self) -> bool:
        """Main training pipeline with STRICT GPU enforcement"""
        try:
            # STRICT GPU check - NO BYPASS
            if not self._check_gpu_requirements():
                print("🛑 TRAINING ABORTED: GPU requirements not met")
                sys.exit(1)
            
            # Start continuous GPU monitoring
            self._continuous_gpu_monitor()
            
            # Analyze dataset
            dataset_stats = self.dataset_processor.analyze_dataset()
            
            # Prepare dataset
            yaml_path = self.dataset_processor.prepare_dataset(
                train_split=self.config['dataset']['train_split']
            )
            
            # Train model with GPU enforcement
            results = self._train_model(yaml_path)
            
            # Stop GPU monitoring
            self.gpu_monitoring = False
            
            if results:
                # Save training log
                self._save_training_log(dataset_stats, results)
                print("🎉 GPU-ONLY training completed successfully!")
                return True
            else:
                print("❌ Training failed!")
                sys.exit(1)
            
        except KeyboardInterrupt:
            print("\n🛑 Training interrupted by user")
            self.gpu_monitoring = False
            sys.exit(1)
        except Exception as e:
            print(f"❌ Training pipeline failed: {e}")
            self.gpu_monitoring = False
            sys.exit(1)
    
    def _check_gpu_requirements(self) -> bool:
        """Check GPU requirements and exit if not met"""
        print("🔍 Checking GPU requirements...")
        
        gpu_info = GPUChecker.get_gpu_info()
        
        if not gpu_info["available"]:
            print("❌ GPU REQUIREMENT NOT MET!")
            print(f"   Reason: {gpu_info.get('reason', 'Unknown')}")
            print("\n💡 GPU Requirements:")
            print("   - CUDA-compatible GPU")
            print("   - Minimum 2GB GPU memory")
            print("   - CUDA drivers installed")
            print("\n🛑 Training aborted. GPU is required for optimal performance.")
            return False
        
        print("✅ GPU Requirements Met!")
        print(f"   Device: {gpu_info['device_name']}")
        print(f"   Memory: {gpu_info['total_memory_gb']:.1f} GB")
        print(f"   Compute: {gpu_info['compute_capability']}")
        
        return True
    
    def _verify_gpu_usage(self):
        """Verify that GPU is actually being used during training"""
        print("� Vterifying GPU usage...")
        
        try:
            # Test tensor allocation on GPU
            test_tensor = torch.randn(1000, 1000, device='cuda:0')
            
            # Check if CUDA is being used
            if not test_tensor.is_cuda:
                print("❌ CRITICAL: Tensor not on GPU!")
                return False
            
            # Check GPU memory usage
            gpu_memory = torch.cuda.memory_allocated(0)
            if gpu_memory == 0:
                print("❌ CRITICAL: No GPU memory allocated!")
                return False
            
            print(f"✅ GPU verification passed - Memory allocated: {gpu_memory / 1024**2:.1f} MB")
            
            # Clean up test tensor
            del test_tensor
            torch.cuda.empty_cache()
            
            return True
            
        except Exception as e:
            print(f"❌ GPU verification failed: {e}")
            return False
    
    def _load_model_on_gpu(self, model_path: str):
        """Load YOLO model and ensure it's on GPU"""
        print("📥 Loading YOLO model with GPU enforcement...")
        
        try:
            # Load model
            model = YOLO(model_path)
            
            # Force model to GPU immediately
            print("🎯 Moving model to GPU...")
            
            # Method 1: Direct CUDA move
            if hasattr(model, 'model') and model.model is not None:
                model.model = model.model.cuda()
                print("✅ Model moved to GPU via .cuda()")
            
            # Method 2: Use to() method
            if hasattr(model, 'model'):
                model.model = model.model.to('cuda:0')
                print("✅ Model moved to GPU via .to('cuda:0')")
            
            # Verify the move worked
            if hasattr(model, 'model') and model.model is not None:
                model_device = next(model.model.parameters()).device
                print(f"📱 Model device after forcing: {model_device}")
                
                if model_device.type != 'cuda':
                    print("❌ CRITICAL: Failed to move model to GPU!")
                    print(f"   Model device type: {model_device.type}")
                    return None
                
                # Check GPU memory usage
                gpu_memory = torch.cuda.memory_allocated(model_device.index or 0)
                print(f"📊 GPU memory after model load: {gpu_memory / 1024**2:.1f} MB")
                
                if gpu_memory < 50 * 1024**2:  # Less than 50MB
                    print("⚠️ WARNING: Low GPU memory usage - model might not be fully on GPU")
                
                print("✅ Model successfully loaded on GPU")
                return model
            else:
                print("❌ CRITICAL: Cannot access model parameters")
                return None
                
        except Exception as e:
            print(f"❌ Failed to load model on GPU: {e}")
            return None
    
    def _monitor_device_usage(self, model):
        """Monitor actual device usage during model initialization"""
        print("🔍 Final device verification...")
        
        try:
            # Check model device
            model_device = next(model.model.parameters()).device
            print(f"📱 Model device: {model_device}")
            
            if model_device.type != 'cuda':
                print("❌ CRITICAL: Model is not on GPU!")
                print(f"   Model device type: {model_device.type}")
                print("🛑 ABORTING TRAINING - GPU REQUIRED!")
                return False
            
            # Verify GPU memory usage
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated(model_device.index or 0)
                print(f"📊 GPU memory usage: {gpu_memory / 1024**2:.1f} MB")
                
                if gpu_memory < 50 * 1024**2:  # Less than 50MB
                    print("⚠️ WARNING: Very low GPU memory usage")
                else:
                    print("✅ Good GPU memory usage detected")
            
            print("✅ Final device verification passed - GPU is being used")
            return True
            
        except Exception as e:
            print(f"❌ Device monitoring failed: {e}")
            return False
    
    def _train_model(self, yaml_path: str):
        """Train the YOLO model with STRICT GPU enforcement"""
        print("🚀 Starting GPU-ONLY training...")
        
        # Pre-training GPU verification
        if not self._verify_gpu_usage():
            print("🛑 TRAINING ABORTED: GPU verification failed")
            sys.exit(1)
        
        # Load model with GPU enforcement
        model = self._load_model_on_gpu(self.config['training']['model'])
        if model is None:
            print("🛑 TRAINING ABORTED: Failed to load model on GPU")
            sys.exit(1)
        
        # Final device verification
        if not self._monitor_device_usage(model):
            print("🛑 TRAINING ABORTED: Model not on GPU after loading")
            sys.exit(1)
        
        # Force GPU parameters - ABSOLUTELY NO CPU FALLBACK
        train_params = {
            'data': yaml_path,
            'epochs': self.config['training']['epochs'],
            'batch': self.config['training']['batch_size'],
            'imgsz': self.config['training']['image_size'],
            'device': 'cuda:0',  # EXPLICIT CUDA DEVICE - NO FALLBACK
            'patience': self.config['training']['patience'],
            'save_period': self.config['training']['save_period'],
            'project': 'helmet_detection',
            'name': f'gpu_enforced_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'exist_ok': True,
            'verbose': True,
            'workers': 8,  # Optimized for GPU
            'amp': True,   # Automatic Mixed Precision for speed
            'cache': True, # Cache images for faster training
        }
        
        print(f"📋 GPU-ONLY Training Parameters:")
        for key, value in train_params.items():
            print(f"   {key}: {value}")
        
        # Final GPU check before training
        print("🔍 Final GPU check before training...")
        if not torch.cuda.is_available():
            print("❌ CRITICAL: CUDA not available at training time!")
            print("🛑 TRAINING ABORTED")
            sys.exit(1)
        
        # Set CUDA device explicitly and environment
        torch.cuda.set_device(0)
        current_device = torch.cuda.current_device()
        print(f"🎯 CUDA device set to: {current_device}")
        
        # Set environment variables to force GPU usage
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        print("🔧 Environment configured for GPU-only operation")
        
        # Clear any cached models to ensure fresh GPU loading
        torch.cuda.empty_cache()
        print("🧹 GPU cache cleared")
        
        try:
            print("🚀 Starting YOLO training on GPU...")
            
            # Monitor GPU usage during training start
            initial_memory = torch.cuda.memory_allocated(0)
            print(f"📊 Initial GPU memory: {initial_memory / 1024**2:.1f} MB")
            
            results = model.train(**train_params)
            
            # Verify GPU was used during training
            final_memory = torch.cuda.memory_allocated(0)
            print(f"📊 Final GPU memory: {final_memory / 1024**2:.1f} MB")
            
            if final_memory < initial_memory + 100 * 1024**2:  # Less than 100MB increase
                print("⚠️ WARNING: Suspiciously low GPU memory usage during training")
            
            self.training_results = {
                'model_path': str(results.save_dir / 'weights' / 'best.pt'),
                'save_dir': str(results.save_dir),
                'training_time': getattr(results, 'speed', 'N/A'),
                'final_metrics': getattr(results, 'results_dict', {}),
                'gpu_memory_used': f"{final_memory / 1024**2:.1f} MB"
            }
            
            print(f"✅ GPU training completed successfully!")
            print(f"📁 Results: {results.save_dir}")
            print(f"🏆 Best model: {self.training_results['model_path']}")
            print(f"📊 GPU memory used: {self.training_results['gpu_memory_used']}")
            
            return results
            
        except RuntimeError as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                print(f"❌ CUDA/GPU ERROR: {e}")
                print("🛑 TRAINING ABORTED - GPU ERROR")
                sys.exit(1)
            else:
                print(f"❌ Training failed: {e}")
                return None
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return None
    
    def _save_training_log(self, dataset_stats: Dict, results):
        """Save comprehensive training log"""
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'gpu_info': GPUChecker.get_gpu_info(),
            'config': self.config,
            'dataset_stats': dataset_stats,
            'training_results': self.training_results,
            'model_info': {
                'architecture': 'YOLOv8s',
                'classes': list(self.config['classes'].keys()),
                'num_classes': len(self.config['classes'])
            }
        }
        
        log_path = f"gpu_training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"📊 Training log saved: {log_path}")

def get_training_config(args) -> Dict:
    """Get training configuration from arguments"""
    return {
        'dataset': {
            'source_path': '../dataset',
            'output_dir': 'yolo_dataset',
            'train_split': 0.8
        },
        'training': {
            'model': args.model,
            'epochs': args.epochs,
            'batch_size': args.batch,
            'image_size': 640,
            'patience': args.patience,
            'save_period': 10
        },
        'classes': {
            'helmet': 0,
            'head': 1
        }
    }

def main():
    parser = argparse.ArgumentParser(
        description="GPU-Optimized Helmet Detection Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--epochs", type=int, default=50, 
                       help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, 
                       help="Batch size (will be optimized for GPU)")
    parser.add_argument("--model", default="yolov8s.pt", 
                       help="Base YOLO model")
    parser.add_argument("--patience", type=int, default=10,
                       help="Early stopping patience")
    
    args = parser.parse_args()
    
    print("🪖 GPU-Optimized Helmet Detection Training")
    print("=" * 45)
    
    # Get configuration
    config = get_training_config(args)
    
    # Initialize and run trainer
    trainer = HelmetTrainer(config)
    success = trainer.train()
    
    if success:
        print("\n🎉 Training completed successfully!")
        print("💡 Next steps:")
        print("   1. Test model: python test_model.py --interactive")
        print("   2. Run validation: python validate_model.py")
        print("   3. Start system: python ../start_system.py")
    else:
        print("\n❌ Training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()