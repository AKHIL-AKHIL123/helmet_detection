# Helmet Detection Frontend

A Streamlit-based web interface for the Helmet Detection System.

## Features

- 📤 **Image Upload**: Support for JPG, PNG, and other common formats
- 🔍 **Real-time Detection**: Connect to FastAPI backend for helmet detection
- 📊 **Safety Metrics**: Display compliance rates and detection statistics
- ⚙️ **Configurable**: Adjustable confidence thresholds
- 🎨 **Visual Results**: Show detection results with bounding boxes

## Quick Start

### Option 1: Use the System Launcher (Recommended)
```bash
python start_system.py
```

### Option 2: Manual Start
1. Start FastAPI backend:
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

2. Start Streamlit frontend:
```bash
streamlit run frontend/app.py --server.port 8501
```

## Usage

1. **Upload Image**: Click "Browse files" and select an image
2. **Adjust Settings**: Use the sidebar to set confidence threshold
3. **Detect**: Click "🔍 Detect Helmets" button
4. **View Results**: See detection results, metrics, and annotated image

## Configuration

Edit `frontend/config.py` to customize:
- API endpoints
- UI settings
- Detection parameters
- Messages and labels

## API Requirements

The frontend expects a FastAPI backend running on `http://localhost:8000` with:
- `GET /` - Health check endpoint
- `POST /detect/` - Image detection endpoint

## Troubleshooting

### API Connection Issues
- Ensure FastAPI backend is running on port 8000
- Check firewall settings
- Verify API endpoint in config.py

### Image Upload Issues
- Supported formats: JPG, JPEG, PNG, BMP, TIFF
- Maximum file size: 10MB
- Ensure image is not corrupted

### Performance Issues
- Lower confidence threshold for more detections
- Use smaller images for faster processing
- Check system resources (CPU/Memory)