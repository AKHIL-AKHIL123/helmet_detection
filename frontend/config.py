# Configuration for Industrial Safety Monitoring System

# API Configuration
API_CONFIG = {
    "base_url": "http://localhost:8000",
    "timeout": 30,
    "max_file_size": 20 * 1024 * 1024,  # 20MB
    "endpoints": {
        "detect": "/detect/",
        "model_info": "/model/info",
        "model_reload": "/model/reload",
        "health": "/health"
    }
}

# UI Configuration
UI_CONFIG = {
    "page_title": "Industrial Safety Monitoring System",
    "page_icon": "⚡",
    "layout": "wide",
    "default_confidence": 0.5,
    "supported_formats": ['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
    "max_batch_size": 50,
    "theme": {
        "primary_color": "#2a5298",
        "secondary_color": "#1e3c72",
        "success_color": "#28a745",
        "warning_color": "#ffc107",
        "danger_color": "#dc3545",
        "background_color": "#FFFFFF",
        "secondary_background_color": "#f8f9fa",
        "text_color": "#212529"
    }
}

# Detection Configuration
DETECTION_CONFIG = {
    "confidence_range": (0.0, 1.0),
    "confidence_step": 0.05,
    "helmet_color": "#00FF00",      # Green for helmet
    "no_helmet_color": "#FF0000",   # Red for no helmet
    "compliance_thresholds": {
        "excellent": 100,
        "good": 80,
        "moderate": 60,
        "poor": 0
    }
}

# Visualization Configuration
VIZ_CONFIG = {
    "chart_height": 300,
    "colors": {
        "helmet": "#00FF00",
        "no_helmet": "#FF0000",
        "compliance_good": "#28a745",
        "compliance_warning": "#ffc107",
        "compliance_danger": "#dc3545"
    },
    "plotly_config": {
        "displayModeBar": False,
        "staticPlot": False
    }
}

# Messages and Text
MESSAGES = {
    "upload_help": "Upload images to detect helmets and analyze safety compliance",
    "batch_upload_help": "Upload multiple images for batch processing and compliance analysis",
    "api_connecting": "🤖 AI is analyzing the image...",
    "api_success": "✅ Detection completed successfully!",
    "api_error": "❌ Detection failed. Please check the API connection.",
    "no_detections": "🔍 No detections found with current confidence threshold. Try lowering the threshold.",
    "compliance_perfect": "🎉 Perfect Safety Compliance: 100%",
    "compliance_good": "✅ Good Safety Compliance: {:.1f}%",
    "compliance_moderate": "⚠️ Moderate Safety Compliance: {:.1f}%",
    "compliance_poor": "🚨 Poor Safety Compliance: {:.1f}% - Immediate Action Required!",
    "model_loading": "🔄 Loading AI model...",
    "batch_processing": "📊 Processing batch images...",
}

# Feature Flags
FEATURES = {
    "batch_processing": True,
    "model_management": True,
    "advanced_metrics": True,
    "compliance_charts": True,
    "export_results": True,
    "real_time_detection": False  # Future feature
}