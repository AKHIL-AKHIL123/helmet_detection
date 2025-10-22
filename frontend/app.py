import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image
import io
import json
import time
from datetime import datetime
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Page config
st.set_page_config(
    page_title="Industrial Safety Monitoring System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Constants
API_BASE_URL = "http://localhost:8000"

# Initialize theme in session state
if 'dark_theme' not in st.session_state:
    st.session_state.dark_theme = False

def apply_theme():
    """Apply light or dark theme based on session state"""
    if st.session_state.dark_theme:
        # Dark theme
        st.markdown("""
        <style>
            .stApp {
                background-color: #1e1e1e !important;
                color: #ffffff !important;
            }
            
            .stApp > div {
                background-color: #1e1e1e !important;
            }
            
            /* Dark theme text */
            .stMarkdown, .stText, p, div, span, h1, h2, h3, h4, h5, h6 {
                color: #ffffff !important;
            }
            
            /* Sidebar styling - dark */
            .css-1d391kg {
                background-color: #2d2d2d !important;
            }
            
            /* Dataframes - dark */
            .stDataFrame {
                background-color: #2d2d2d !important;
                color: #ffffff !important;
            }
            
            /* File uploader - dark theme */
            .stFileUploader {
                background-color: #2d2d2d !important;
            }
            
            .stFileUploader > div {
                background-color: #2d2d2d !important;
                border: 2px dashed #4a5568 !important;
                border-radius: 8px !important;
            }
            
            .stFileUploader label {
                color: #ffffff !important;
                font-weight: 600 !important;
            }
            
            .stFileUploader div[data-testid="stFileUploaderDropzone"] {
                background-color: #2d2d2d !important;
                color: #ffffff !important;
            }
            
            .stFileUploader div[data-testid="stFileUploaderDropzone"] > div {
                color: #ffffff !important;
            }
            
            .stFileUploader small {
                color: #a0a0a0 !important;
            }
            
            /* File uploader button - dark */
            .stFileUploader button {
                background-color: #4a5568 !important;
                color: #ffffff !important;
                border: 1px solid #4a5568 !important;
            }
            
            .stFileUploader button:hover {
                background-color: #2d3748 !important;
                border-color: #2d3748 !important;
            }
            
            /* Main header - dark */
            .main-header {
                background: linear-gradient(90deg, #2d3748 0%, #4a5568 100%);
                padding: 1rem 2rem;
                border-radius: 10px;
                margin-bottom: 2rem;
                color: white;
                text-align: center;
            }
            
            /* Metric cards - dark */
            .metric-card {
                background: #2d2d2d;
                padding: 1rem;
                border-radius: 8px;
                border-left: 4px solid #4a5568;
                box-shadow: 0 2px 4px rgba(0,0,0,0.3);
                margin-bottom: 1rem;
                color: #ffffff;
            }
            
            .status-success {
                background: #1a2e1a;
                border-left-color: #28a745;
                color: #90ee90;
            }
            
            .status-warning {
                background: #2e2a1a;
                border-left-color: #ffc107;
                color: #ffeb3b;
            }
            
            .status-danger {
                background: #2e1a1a;
                border-left-color: #dc3545;
                color: #ff6b6b;
            }
            
            /* Upload section - dark */
            .upload-section {
                border: 2px dashed #4a5568;
                border-radius: 10px;
                padding: 2rem;
                text-align: center;
                background: #2d2d2d;
                margin-bottom: 2rem;
                color: #ffffff;
            }
            
            /* Results section - dark */
            .results-section {
                background: #2d2d2d;
                border-radius: 10px;
                padding: 1.5rem;
                box-shadow: 0 2px 8px rgba(0,0,0,0.3);
                color: #ffffff;
            }
            
            /* Sidebar section - dark */
            .sidebar-section {
                background: #2d2d2d;
                padding: 1rem;
                border-radius: 8px;
                margin-bottom: 1rem;
            }
            
            /* Theme toggle button */
            .theme-toggle {
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 999;
                background: #4a5568;
                color: white;
                border: none;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                font-size: 20px;
                cursor: pointer;
                box-shadow: 0 2px 8px rgba(0,0,0,0.3);
                transition: all 0.3s ease;
            }
            
            .theme-toggle:hover {
                background: #2d3748;
                transform: scale(1.1);
            }
            
            /* Additional file uploader fixes - dark */
            .stFileUploader div[data-baseweb="file-uploader"] {
                background-color: #2d2d2d !important;
                color: #ffffff !important;
            }
            
            .stFileUploader div[data-baseweb="file-uploader"] span {
                color: #ffffff !important;
            }
            
            .stFileUploader div[data-baseweb="file-uploader"] p {
                color: #ffffff !important;
            }
            
            /* Force all file uploader text to be visible - dark */
            .stFileUploader * {
                color: #ffffff !important;
            }
            
            .stFileUploader section {
                background-color: #2d2d2d !important;
                border: 2px dashed #4a5568 !important;
            }
            
            .stFileUploader section > div {
                color: #ffffff !important;
            }
            
            /* File drag and drop area - dark */
            .stFileUploader [data-testid="stFileUploaderDropzone"] {
                background-color: #2d2d2d !important;
                border: 2px dashed #4a5568 !important;
                color: #ffffff !important;
            }
            
            .stFileUploader [data-testid="stFileUploaderDropzone"] * {
                color: #ffffff !important;
            }
            
            /* Override any inherited styles - dark */
            .stFileUploader div, .stFileUploader span, .stFileUploader p, .stFileUploader label {
                color: #ffffff !important;
                background-color: transparent !important;
            }
            
            /* File uploader instructions text - dark */
            .stFileUploader [data-testid="stFileUploaderInstructions"] {
                color: #ffffff !important;
            }
            
            .stFileUploader [data-testid="stFileUploaderInstructions"] * {
                color: #ffffff !important;
            }
        </style>
        """, unsafe_allow_html=True)
    else:
        # Light theme
        st.markdown("""
        <style>
            .stApp {
                background-color: #ffffff !important;
                color: #212529 !important;
            }
            
            .stApp > div {
                background-color: #ffffff !important;
            }
            
            /* Light theme text */
            .stMarkdown, .stText, p, div, span, h1, h2, h3, h4, h5, h6 {
                color: #212529 !important;
            }
            
            /* Sidebar styling - light */
            .css-1d391kg {
                background-color: #f8f9fa !important;
            }
            
            /* Dataframes - light */
            .stDataFrame {
                background-color: white !important;
                color: #212529 !important;
            }
            
            /* File uploader - light theme */
            .stFileUploader {
                background-color: #ffffff !important;
            }
            
            .stFileUploader > div {
                background-color: #ffffff !important;
                border: 2px dashed #cccccc !important;
                border-radius: 8px !important;
            }
            
            .stFileUploader label {
                color: #212529 !important;
                font-weight: 600 !important;
            }
            
            .stFileUploader div[data-testid="stFileUploaderDropzone"] {
                background-color: #f8f9fa !important;
                color: #212529 !important;
            }
            
            .stFileUploader div[data-testid="stFileUploaderDropzone"] > div {
                color: #212529 !important;
            }
            
            .stFileUploader small {
                color: #6c757d !important;
            }
            
            /* File uploader button - light */
            .stFileUploader button {
                background-color: #2a5298 !important;
                color: #ffffff !important;
                border: 1px solid #2a5298 !important;
            }
            
            .stFileUploader button:hover {
                background-color: #1e3c72 !important;
                border-color: #1e3c72 !important;
            }
            
            /* Main header - light */
            .main-header {
                background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
                padding: 1rem 2rem;
                border-radius: 10px;
                margin-bottom: 2rem;
                color: white;
                text-align: center;
            }
            
            /* Metric cards - light */
            .metric-card {
                background: white;
                padding: 1rem;
                border-radius: 8px;
                border-left: 4px solid #2a5298;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 1rem;
                color: #212529;
            }
            
            .status-success {
                background: #d4edda;
                border-left-color: #28a745;
                color: #155724;
            }
            
            .status-warning {
                background: #fff3cd;
                border-left-color: #ffc107;
                color: #856404;
            }
            
            .status-danger {
                background: #f8d7da;
                border-left-color: #dc3545;
                color: #721c24;
            }
            
            /* Upload section - light */
            .upload-section {
                border: 2px dashed #cccccc;
                border-radius: 10px;
                padding: 2rem;
                text-align: center;
                background: #f8f9fa;
                margin-bottom: 2rem;
                color: #212529;
            }
            
            /* Results section - light */
            .results-section {
                background: white;
                border-radius: 10px;
                padding: 1.5rem;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                color: #212529;
            }
            
            /* Sidebar section - light */
            .sidebar-section {
                background: #f8f9fa;
                padding: 1rem;
                border-radius: 8px;
                margin-bottom: 1rem;
            }
            
            /* Theme toggle button */
            .theme-toggle {
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 999;
                background: #2a5298;
                color: white;
                border: none;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                font-size: 20px;
                cursor: pointer;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                transition: all 0.3s ease;
            }
            
            .theme-toggle:hover {
                background: #1e3c72;
                transform: scale(1.1);
            }
            
            /* Additional file uploader fixes - light */
            .stFileUploader div[data-baseweb="file-uploader"] {
                background-color: #ffffff !important;
                color: #212529 !important;
            }
            
            .stFileUploader div[data-baseweb="file-uploader"] span {
                color: #212529 !important;
            }
            
            .stFileUploader div[data-baseweb="file-uploader"] p {
                color: #212529 !important;
            }
            
            /* Force all file uploader text to be visible - light */
            .stFileUploader * {
                color: #212529 !important;
            }
            
            .stFileUploader section {
                background-color: #f8f9fa !important;
                border: 2px dashed #cccccc !important;
            }
            
            .stFileUploader section > div {
                color: #212529 !important;
            }
            
            /* File drag and drop area - light */
            .stFileUploader [data-testid="stFileUploaderDropzone"] {
                background-color: #f8f9fa !important;
                border: 2px dashed #cccccc !important;
                color: #212529 !important;
            }
            
            .stFileUploader [data-testid="stFileUploaderDropzone"] * {
                color: #212529 !important;
            }
            
            /* Override any inherited styles - light */
            .stFileUploader div, .stFileUploader span, .stFileUploader p, .stFileUploader label {
                color: #212529 !important;
                background-color: transparent !important;
            }
            
            /* File uploader instructions text - light */
            .stFileUploader [data-testid="stFileUploaderInstructions"] {
                color: #212529 !important;
            }
            
            .stFileUploader [data-testid="stFileUploaderInstructions"] * {
                color: #212529 !important;
            }
        </style>
        """, unsafe_allow_html=True)

def toggle_theme():
    """Toggle between light and dark theme"""
    st.session_state.dark_theme = not st.session_state.dark_theme
    st.rerun()

def clear_detection_results():
    """Clear previous detection results from session state"""
    keys_to_clear = ['detection_result', 'detection_timestamp', 'current_file_info']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

def main():
    # Apply current theme
    apply_theme()
    
    # Theme toggle button
    theme_icon = "🌙" if not st.session_state.dark_theme else "☀️"
    theme_text = "Dark Mode" if not st.session_state.dark_theme else "Light Mode"
    
    # Create theme toggle in sidebar
    with st.sidebar:
        if st.button(f"{theme_icon} {theme_text}", use_container_width=True):
            toggle_theme()
    
    # Professional header
    st.markdown("""
    <div class="main-header">
        <h1>Industrial Safety Monitoring System</h1>
        <p>AI-Powered Personal Protective Equipment Detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check API status first
    api_status = check_api_status()
    
    # Create main layout
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("### System Configuration")
        
        # API Status - Professional display
        if api_status['connected']:
            st.markdown("""
            <div class="metric-card status-success">
                <strong>System Status:</strong> Online<br>
                <small>AI Detection Service Active</small>
            </div>
            """, unsafe_allow_html=True)
            
            if api_status.get('model_info'):
                model_info = api_status['model_info']
                if 'classes' in model_info and model_info['classes']:
                    st.markdown("**Detection Classes:**")
                    for class_id, class_name in model_info['classes'].items():
                        st.text(f"• {class_name.title()}")
        else:
            st.markdown("""
            <div class="metric-card status-danger">
                <strong>System Status:</strong> Offline<br>
                <small>Detection service unavailable</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Settings
        st.markdown("### Detection Parameters")
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            0.0, 1.0, 0.5, 0.05,
            help="Minimum confidence level for detections"
        )
        
        # Advanced options in expander
        with st.expander("Advanced Options"):
            batch_mode = st.checkbox("Batch Processing Mode")
            save_results = st.checkbox("Save Results Locally", True)
            show_coordinates = st.checkbox("Show Bounding Box Coordinates")
    
    with col1:
        # Main content area
        if not batch_mode:
            single_image_detection(confidence_threshold, save_results, show_coordinates if 'show_coordinates' in locals() else False)
        else:
            batch_image_detection(confidence_threshold, save_results)

def single_image_detection(confidence_threshold, save_results, show_coordinates=False):
    """Professional single image detection interface"""
    
    st.markdown("### Image Analysis")
    
    # Initialize uploader key counter if not exists
    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0
    
    # Upload section with professional styling
    upload_bg_color = "#2d2d2d" if st.session_state.dark_theme else "#f8f9fa"
    upload_text_color = "#ffffff" if st.session_state.dark_theme else "#212529"
    upload_border_color = "#4a5568" if st.session_state.dark_theme else "#cccccc"
    
    st.markdown(f"""
    <div style="
        border: 2px dashed {upload_border_color};
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: {upload_bg_color};
        margin-bottom: 2rem;
        color: {upload_text_color};
    ">
        <h4 style="color: {upload_text_color}; margin-bottom: 0.5rem;">Upload Image for Analysis</h4>
        <p style="color: {upload_text_color}; margin: 0;">Supported formats: JPG, PNG, BMP, TIFF</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add "Upload New Image" button to reset the uploader
    col_upload_btn, col_spacer = st.columns([1, 3])
    with col_upload_btn:
        if st.button("🔄 Upload New Image", use_container_width=True):
            # Increment key to force file uploader reset
            st.session_state.uploader_key += 1
            # Clear all previous data
            clear_detection_results()
            st.rerun()
    
    uploaded_file = st.file_uploader(
        "Select image file", 
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        label_visibility="collapsed",
        key=f"single_image_uploader_{st.session_state.uploader_key}"
    )
    
    # Clear previous results when a new image is uploaded
    if uploaded_file is not None:
        # Check if this is a new file
        current_file_info = f"{uploaded_file.name}_{len(uploaded_file.getvalue())}"
        if 'current_file_info' not in st.session_state or st.session_state.current_file_info != current_file_info:
            # New file uploaded, clear previous results
            st.session_state.current_file_info = current_file_info
            if 'detection_result' in st.session_state:
                del st.session_state['detection_result']
            if 'detection_timestamp' in st.session_state:
                del st.session_state['detection_timestamp']
    
    if uploaded_file is not None:
        # Create two columns for image and results
        img_col, results_col = st.columns([1, 1])
        
        with img_col:
            # Display original image with error handling
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Source Image", use_container_width=True)
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
                st.stop()
            
            # Image metadata in a clean format
            st.markdown("**Image Information**")
            metadata_df = pd.DataFrame({
                'Property': ['Filename', 'Dimensions', 'Format', 'Size'],
                'Value': [
                    uploaded_file.name,
                    f"{image.size[0]} × {image.size[1]} pixels",
                    image.format,
                    f"{len(uploaded_file.getvalue()) / 1024:.1f} KB"
                ]
            })
            st.dataframe(metadata_df, hide_index=True, use_container_width=True)
            
            # Professional detection button
            if st.button("Start Analysis", type="primary", use_container_width=True):
                # Check image size and warn if large
                file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
                if file_size_mb > 5:
                    st.warning(f"Large image detected ({file_size_mb:.1f}MB). Processing may take longer.")
                
                with st.spinner("Processing image..."):
                    progress_bar = st.progress(0)
                    progress_bar.progress(25)
                    
                    result = detect_helmets(uploaded_file, confidence_threshold)
                    progress_bar.progress(100)
                    
                    if result:
                        st.session_state['detection_result'] = result
                        st.session_state['detection_timestamp'] = datetime.now()
                        st.success("Analysis completed successfully")
                        progress_bar.empty()
                    else:
                        st.error("Analysis failed - Check system connection")
                        progress_bar.empty()
        
        with results_col:
            # Check if we have results for the current image
            current_file_info = st.session_state.get('current_file_info', '')
            if ('detection_result' in st.session_state and 
                'current_file_info' in st.session_state and 
                current_file_info):
                
                result = st.session_state['detection_result']
                timestamp = st.session_state.get('detection_timestamp', datetime.now())
                
                # Professional results display with success indicator
                st.success("✅ Analysis Complete!")
                st.markdown("**Analysis Results**")
                st.caption(f"Completed: {timestamp.strftime('%H:%M:%S on %Y-%m-%d')}")
                st.caption(f"📁 File: {uploaded_file.name}")
                
                # Display metrics professionally
                display_professional_metrics(result)
                
                # Display result image with error handling
                if 'result_image_data' in result:
                    st.markdown("**Detection Visualization**")
                    try:
                        st.image(
                            result['result_image_data'], 
                            caption="Detected Objects with Confidence Scores", 
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Error displaying result image: {str(e)}")
                
                # Display detection details
                display_professional_details(result['detections'], show_coordinates)
                
                # Add button to analyze another image
                st.markdown("---")
                if st.button("📤 Analyze Another Image", use_container_width=True, type="primary"):
                    st.session_state.uploader_key += 1
                    clear_detection_results()
                    st.rerun()
            else:
                st.markdown("""
                <div class="results-section">
                    <h4>Analysis Results</h4>
                    <p>Results will appear here after image analysis</p>
                    <p style="margin-top: 1rem;">👆 Upload an image and click "Start Analysis" to begin</p>
                </div>
                """, unsafe_allow_html=True)

def batch_image_detection(confidence_threshold, save_results):
    """Professional batch image detection interface"""
    st.markdown("### Batch Analysis Mode")
    
    # Initialize batch uploader key if not exists
    if 'batch_uploader_key' not in st.session_state:
        st.session_state.batch_uploader_key = 0
    
    # Upload section with theme-aware styling
    upload_bg_color = "#2d2d2d" if st.session_state.dark_theme else "#f8f9fa"
    upload_text_color = "#ffffff" if st.session_state.dark_theme else "#212529"
    upload_border_color = "#4a5568" if st.session_state.dark_theme else "#cccccc"
    
    st.markdown(f"""
    <div style="
        border: 2px dashed {upload_border_color};
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: {upload_bg_color};
        margin-bottom: 2rem;
        color: {upload_text_color};
    ">
        <h4 style="color: {upload_text_color}; margin-bottom: 0.5rem;">Upload Multiple Images</h4>
        <p style="color: {upload_text_color}; margin: 0;">Process multiple images simultaneously for comprehensive safety analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add "Upload New Batch" button
    col_batch_btn, col_spacer = st.columns([1, 3])
    with col_batch_btn:
        if st.button("🔄 Upload New Batch", use_container_width=True):
            st.session_state.batch_uploader_key += 1
            if 'batch_results' in st.session_state:
                del st.session_state['batch_results']
            st.rerun()
    
    uploaded_files = st.file_uploader(
        "Select multiple image files", 
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"batch_image_uploader_{st.session_state.batch_uploader_key}"
    )
    
    if uploaded_files:
        st.markdown(f"**Selected Files:** {len(uploaded_files)} images ready for processing")
        
        # Show file list
        file_data = []
        total_size = 0
        for i, file in enumerate(uploaded_files, 1):
            size_kb = len(file.getvalue()) / 1024
            total_size += size_kb
            file_data.append({
                'No.': i,
                'Filename': file.name,
                'Size (KB)': f"{size_kb:.1f}",
                'Type': file.type
            })
        
        files_df = pd.DataFrame(file_data)
        st.dataframe(files_df, hide_index=True, use_container_width=True)
        
        st.markdown(f"**Total Size:** {total_size:.1f} KB")
        
        if st.button("Start Batch Analysis", type="primary", use_container_width=True):
            process_batch_images(uploaded_files, confidence_threshold)

def detect_helmets(uploaded_file, confidence_threshold):
    """Send image to FastAPI backend for detection with enhanced error handling"""
    max_retries = 2
    
    for attempt in range(max_retries):
        try:
            # Reset file pointer
            uploaded_file.seek(0)
            
            # Prepare files for API request
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            
            # Show progress for user
            if attempt > 0:
                st.info(f"Retry attempt {attempt + 1}/{max_retries}")
            
            # Make API request with extended timeout
            response = requests.post(
                f"{API_BASE_URL}/detect/", 
                files=files,
                timeout=90  # Extended timeout for large images
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Filter detections by confidence threshold (client-side filtering)
                if 'detections' in result:
                    filtered_detections = [
                        det for det in result['detections'] 
                        if det['confidence'] >= confidence_threshold
                    ]
                    result['detections'] = filtered_detections
                    
                    # Update metrics after filtering
                    if 'metrics' in result:
                        helmet_count = sum(1 for det in filtered_detections if det['label'] == 'helmet')
                        head_count = len(filtered_detections) - helmet_count
                        compliance_rate = (helmet_count / len(filtered_detections) * 100) if filtered_detections else 0
                        
                        result['metrics'] = {
                            'total_detections': len(filtered_detections),
                            'helmet_count': helmet_count,
                            'head_count': head_count,
                            'compliance_rate': round(compliance_rate, 1)
                        }
                
                # Try to get the result image
                if 'result_image' in result:
                    try:
                        img_response = requests.get(f"{API_BASE_URL}{result['result_image']}", timeout=10)
                        if img_response.status_code == 200:
                            result['result_image_data'] = img_response.content
                    except Exception as e:
                        st.warning(f"Could not load result image: {str(e)}")
                
                return result
            else:
                error_detail = "Unknown error"
                try:
                    error_response = response.json()
                    error_detail = error_response.get('detail', response.text)
                except:
                    error_detail = response.text
                
                if attempt == max_retries - 1:
                    st.error(f"API Error ({response.status_code}): {error_detail}")
                    return None
                time.sleep(2)
                continue
                
        except requests.exceptions.ConnectionError:
            if attempt == max_retries - 1:
                st.error("Connection Error: Detection service unavailable. Please check system status.")
                return None
            time.sleep(2)  # Wait before retry
            continue
            
        except requests.exceptions.Timeout:
            if attempt == max_retries - 1:
                st.error("Processing timeout. Try with a smaller image or check system performance.")
                return None
            time.sleep(3)  # Wait before retry
            continue
            
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"System Error: {str(e)}")
                return None
            time.sleep(2)
            continue
    
    return None

def check_api_status():
    """Enhanced API status check with model information"""
    try:
        # Check basic connectivity
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        if response.status_code == 200:
            api_info = response.json()
            
            # Get model information
            model_response = requests.get(f"{API_BASE_URL}/model/info", timeout=5)
            model_info = model_response.json() if model_response.status_code == 200 else {}
            
            return {
                'connected': True,
                'api_info': api_info,
                'model_info': model_info
            }
        else:
            return {'connected': False, 'error': f"HTTP {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {'connected': False, 'error': 'Connection refused'}
    except requests.exceptions.Timeout:
        return {'connected': False, 'error': 'Connection timeout'}
    except Exception as e:
        return {'connected': False, 'error': str(e)}

def process_batch_images(uploaded_files, confidence_threshold):
    """Process multiple images in batch"""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name}...")
        progress_bar.progress((i + 1) / len(uploaded_files))
        
        result = detect_helmets(uploaded_file, confidence_threshold)
        if result:
            result['filename'] = uploaded_file.name
            results.append(result)
        
        time.sleep(0.1)  # Small delay to show progress
    
    status_text.text("✅ Batch processing completed!")
    
    # Display batch results
    display_batch_results(results)

def display_professional_metrics(result):
    """Display professional detection metrics"""
    metrics = result.get('metrics', {})
    detections = result.get('detections', [])
    
    total_detections = metrics.get('total_detections', len(detections))
    helmet_count = metrics.get('helmet_count', sum(1 for det in detections if det['label'] == 'helmet'))
    head_count = metrics.get('head_count', total_detections - helmet_count)
    compliance_rate = metrics.get('compliance_rate', 0)
    
    # Professional metrics display
    metrics_data = {
        'Metric': ['Total Personnel', 'Helmet Detected', 'No Helmet', 'Compliance Rate'],
        'Value': [total_detections, helmet_count, head_count, f"{compliance_rate:.1f}%"],
        'Status': [
            'Detected',
            'Compliant' if helmet_count > 0 else 'None',
            'Non-Compliant' if head_count > 0 else 'None',
            get_compliance_status(compliance_rate)
        ]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, hide_index=True, use_container_width=True)
    
    # Compliance status with professional styling
    if total_detections > 0:
        compliance_html = get_compliance_html(compliance_rate)
        st.markdown(compliance_html, unsafe_allow_html=True)
        
        # Professional compliance chart
        if total_detections > 1:
            create_professional_chart(helmet_count, head_count)
    else:
        st.markdown("""
        <div class="metric-card">
            <strong>Detection Status:</strong> No personnel detected in image
        </div>
        """, unsafe_allow_html=True)

def get_compliance_status(rate):
    """Get compliance status text"""
    if rate == 100:
        return "Excellent"
    elif rate >= 80:
        return "Good"
    elif rate >= 60:
        return "Moderate"
    else:
        return "Poor"

def get_compliance_html(rate):
    """Get professional compliance HTML"""
    if rate == 100:
        return """
        <div class="metric-card status-success">
            <strong>Safety Compliance:</strong> 100% - Excellent<br>
            <small>All personnel wearing required PPE</small>
        </div>
        """
    elif rate >= 80:
        return f"""
        <div class="metric-card status-success">
            <strong>Safety Compliance:</strong> {rate:.1f}% - Good<br>
            <small>Most personnel compliant with PPE requirements</small>
        </div>
        """
    elif rate >= 60:
        return f"""
        <div class="metric-card status-warning">
            <strong>Safety Compliance:</strong> {rate:.1f}% - Moderate<br>
            <small>Some personnel not wearing required PPE</small>
        </div>
        """
    else:
        return f"""
        <div class="metric-card status-danger">
            <strong>Safety Compliance:</strong> {rate:.1f}% - Critical<br>
            <small>Immediate safety intervention required</small>
        </div>
        """

def create_professional_chart(helmet_count, head_count):
    """Create professional compliance visualization"""
    if helmet_count + head_count == 0:
        return
    
    # Professional bar chart with theme-aware colors
    categories = ['Compliant', 'Non-Compliant']
    values = [helmet_count, head_count]
    colors = ['#28a745', '#dc3545']
    
    # Adjust chart colors based on theme
    bg_color = '#2d2d2d' if st.session_state.dark_theme else 'white'
    text_color = '#ffffff' if st.session_state.dark_theme else '#2a5298'
    grid_color = '#4a5568' if st.session_state.dark_theme else '#f0f0f0'
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=values,
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="PPE Compliance Analysis",
        xaxis_title="Compliance Status",
        yaxis_title="Personnel Count",
        height=300,
        showlegend=False,
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(family="Arial, sans-serif", size=12, color=text_color),
        title_font=dict(size=14, color=text_color)
    )
    
    fig.update_xaxis(showgrid=False)
    fig.update_yaxis(showgrid=True, gridcolor=grid_color)
    
    st.plotly_chart(fig, use_container_width=True)

def display_professional_details(detections, show_coordinates=False):
    """Display professional detection details"""
    if not detections:
        st.markdown("""
        <div class="metric-card">
            <strong>Detection Status:</strong> No objects detected<br>
            <small>Try adjusting the confidence threshold</small>
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.markdown("**Detection Summary**")
    
    # Professional detection table
    detection_data = []
    for i, detection in enumerate(detections, 1):
        row_data = {
            'ID': f"#{i:02d}",
            'Object': detection['label'].title(),
            'Confidence': f"{detection['confidence']:.1%}",
            'Status': 'Compliant' if detection['label'] == 'helmet' else 'Non-Compliant'
        }
        
        if show_coordinates:
            box = detection['box']
            row_data['Coordinates'] = f"({box['x1']}, {box['y1']}, {box['x2']}, {box['y2']})"
        
        detection_data.append(row_data)
    
    df = pd.DataFrame(detection_data)
    
    # Style the dataframe
    def style_compliance(val):
        if val == 'Compliant':
            return 'background-color: #d4edda; color: #155724'
        elif val == 'Non-Compliant':
            return 'background-color: #f8d7da; color: #721c24'
        return ''
    
    styled_df = df.style.applymap(style_compliance, subset=['Status'])
    st.dataframe(styled_df, hide_index=True, use_container_width=True)
    
    # Summary statistics
    compliant_count = sum(1 for d in detections if d['label'] == 'helmet')
    avg_confidence = sum(d['confidence'] for d in detections) / len(detections)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Average Confidence", f"{avg_confidence:.1%}")
    with col2:
        st.metric("Compliant Objects", f"{compliant_count}/{len(detections)}")

def display_batch_results(results):
    """Display results from batch processing"""
    if not results:
        st.warning("No successful detections in batch processing.")
        return
    
    st.subheader("📊 Batch Processing Results")
    
    # Summary statistics
    total_images = len(results)
    total_detections = sum(len(r['detections']) for r in results)
    total_helmets = sum(r.get('metrics', {}).get('helmet_count', 0) for r in results)
    total_heads = sum(r.get('metrics', {}).get('head_count', 0) for r in results)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📁 Images Processed", total_images)
    with col2:
        st.metric("👥 Total People", total_detections)
    with col3:
        st.metric("🪖 With Helmets", total_helmets)
    with col4:
        st.metric("👤 Without Helmets", total_heads)
    
    # Overall compliance
    overall_compliance = (total_helmets / total_detections * 100) if total_detections > 0 else 0
    if overall_compliance >= 80:
        st.success(f"🎉 Overall Compliance: {overall_compliance:.1f}%")
    else:
        st.error(f"🚨 Overall Compliance: {overall_compliance:.1f}% - Action Required")
    
    # Detailed results table
    batch_data = []
    for result in results:
        metrics = result.get('metrics', {})
        batch_data.append({
            'Filename': result['filename'],
            'People Detected': metrics.get('total_detections', 0),
            'With Helmet': metrics.get('helmet_count', 0),
            'Without Helmet': metrics.get('head_count', 0),
            'Compliance %': f"{metrics.get('compliance_rate', 0):.1f}%"
        })
    
    batch_df = pd.DataFrame(batch_data)
    st.dataframe(batch_df, use_container_width=True)
    
    # Compliance trend chart
    if len(results) > 1:
        compliance_data = [r.get('metrics', {}).get('compliance_rate', 0) for r in results]
        filenames = [r['filename'] for r in results]
        
        fig = px.bar(
            x=filenames, 
            y=compliance_data,
            title="Compliance Rate by Image",
            labels={'x': 'Image', 'y': 'Compliance Rate (%)'},
            color=compliance_data,
            color_continuous_scale=['red', 'yellow', 'green']
        )
        
        # Theme-aware chart styling
        bg_color = '#2d2d2d' if st.session_state.dark_theme else 'white'
        text_color = '#ffffff' if st.session_state.dark_theme else '#2a5298'
        
        fig.update_layout(
            height=400,
            plot_bgcolor=bg_color,
            paper_bgcolor=bg_color,
            font=dict(color=text_color)
        )
        
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()