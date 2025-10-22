# 🪖 Helmet Detection System

A complete AI-powered safety compliance system that detects helmets in construction and industrial environments using YOLOv8 and provides a user-friendly web interface.

## 🌟 Features

- **Real-time Helmet Detection** using YOLOv8
- **Safety Compliance Metrics** with percentage calculations
- **Web Interface** built with Streamlit
- **REST API** powered by FastAPI
- **Batch Processing** for multiple images
- **Interactive Testing** tools
- **Model Training** pipeline

## 🏗️ Architecture

```
📁 helmet_detection/
├── 🧠 backend/           # FastAPI + YOLO model
├── 🎨 frontend/          # Streamlit web interface
├── 📊 dataset/           # Training images & annotations
├── 🚀 start_system.py    # Complete system launcher
└── 📋 requirements.txt   # Dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Complete System

```bash
python start_system.py
```

This will:

- ✅ Check all dependencies
- 🚀 Start FastAPI backend (port 8000)
- 🎨 Launch Streamlit frontend (port 8501)
- 🌐 Open your browser automatically

### 3. Use the Application

1. **Upload an image** in the web interface
2. **Adjust confidence threshold** if needed
3. **Click "Detect Helmets"** to analyze
4. **View results** with safety compliance metrics

## 🛠️ Manual Setup

### Backend Only

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Only

```bash
streamlit run frontend/app.py --server.port 8501
```

## 📊 Model Information

- **Architecture**: YOLOv8 Small
- **Classes**:
  - `helmet` (Class 0) - Person wearing helmet
  - `head` (Class 1) - Person without helmet
- **Training Data**: 766 annotated images
- **Format**: VOC XML → YOLO format conversion

## 🧪 Testing & Validation

### Quick Model Test

```bash
python backend/test_model.py --image path/to/test/image.jpg
```

### Interactive Testing

```bash
python backend/test_model.py --interactive
```

### Test Single Image

```bash
python backend/test_model.py --image path/to/image.jpg
```

### Full Model Validation

```bash
python backend/validate_model.py
```

## 🎯 Training

### Check GPU Availability (REQUIRED)
```bash
python backend/check_gpu.py
```

### Train the Model (GPU ONLY)
```bash
python backend/train.py --epochs 50
```

### Training Options
- `--epochs`: Number of training epochs (default: 50)
- `--batch`: Batch size (default: 16, auto-optimized for GPU)
- `--model`: Base model to use (default: yolov8s.pt)
- `--patience`: Early stopping patience (default: 10)

### STRICT GPU Requirements
- ✅ CUDA-compatible NVIDIA GPU (REQUIRED)
- ✅ Minimum 2GB GPU memory (REQUIRED)
- ✅ CUDA drivers installed (REQUIRED)
- ❌ **CPU training BLOCKED** - Training will abort immediately if GPU not detected
- ⚡ **Fast Training**: 15-20 minutes on GPU vs 10+ hours on CPU

## 📁 Project Structure

```
helmet_detection/
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── train.py               # Complete training pipeline
│   ├── validate_model.py      # Model validation
│   ├── test_model.py          # Interactive testing
│   └── helmet_detection/      # Trained model weights
├── frontend/
│   ├── app.py                 # Streamlit application
│   ├── config.py              # Frontend configuration
│   └── README.md              # Frontend documentation
├── dataset/
│   ├── images/                # Training images
│   └── annotations/           # VOC XML annotations
├── start_system.py            # Complete system launcher
└── requirements.txt           # Dependencies
```

## 🔧 Configuration

### API Configuration (`frontend/config.py`)

```python
API_CONFIG = {
    "base_url": "http://localhost:8000",
    "timeout": 30,
    "max_file_size": 10 * 1024 * 1024,  # 10MB
}
```

### Model Auto-Detection

The FastAPI backend automatically finds and loads the latest trained model:
- Scans `backend/helmet_detection/` for trained models
- Loads the most recently trained model
- Provides model info via `/model/info` endpoint
- Supports model reloading via `/model/reload` endpoint

## 📈 Performance Metrics

The system provides:

- **Precision**: Accuracy of helmet detections
- **Recall**: Coverage of actual helmets
- **F1-Score**: Balanced performance metric
- **Safety Compliance**: Percentage of people wearing helmets

## 🐛 Troubleshooting

### Model Not Found

```bash
# Check if model exists
ls backend/helmet_detection/yolov8s_helmet_detection/weights/

# If missing, train the model
python backend/train.py
```

### API Connection Issues

```bash
# Check if FastAPI is running
curl http://localhost:8000/

# Check firewall settings
# Ensure ports 8000 and 8501 are open
```

### CUDA Issues

```bash
# For CPU-only systems, use:
python backend/train.py --device cpu
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **Ultralytics YOLOv8** for the detection model
- **FastAPI** for the backend framework
- **Streamlit** for the web interface
- **OpenCV** for image processing

## 📞 Support

For issues and questions:

1. Check the troubleshooting section
2. Review the logs in the terminal
3. Test individual components manually
4. Create an issue with detailed error messages

---

**🪖 Stay Safe! Wear Your Helmet!** 🚧
