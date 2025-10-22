#!/usr/bin/env python3
"""
Complete Helmet Detection System Launcher
Integrates FastAPI backend + Streamlit frontend with health checks
"""

import subprocess
import sys
import time
import threading
import requests
import os
from pathlib import Path
import webbrowser

class SystemLauncher:
    def __init__(self):
        self.fastapi_process = None
        self.streamlit_process = None
        self.api_url = "http://localhost:8000"
        self.frontend_url = "http://localhost:8501"
        
    def check_model_exists(self):
        """Check if trained model exists"""
        model_path = "backend/helmet_detection/yolov8s_helmet_detection/weights/best.pt"
        if not os.path.exists(model_path):
            print("❌ Trained model not found!")
            print(f"Expected: {model_path}")
            print("💡 Run training first: python backend/train.py")
            return False
        
        print(f"✅ Model found: {model_path}")
        return True
    
    def check_dependencies(self):
        """Check if required files exist"""
        required_files = [
            "backend/main.py",
            "frontend/app.py",
            "requirements.txt"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            print("❌ Missing required files:")
            for file in missing_files:
                print(f"   - {file}")
            return False
        
        print("✅ All required files found")
        return True
    
    def wait_for_api(self, timeout=30):
        """Wait for FastAPI to be ready"""
        print("⏳ Waiting for API to start...")
        
        for i in range(timeout):
            try:
                response = requests.get(f"{self.api_url}/", timeout=2)
                if response.status_code == 200:
                    print("✅ API is ready!")
                    return True
            except:
                pass
            
            time.sleep(1)
            if i % 5 == 0:
                print(f"   Still waiting... ({i}/{timeout}s)")
        
        print("❌ API failed to start within timeout")
        return False
    
    def start_fastapi(self):
        """Start FastAPI backend"""
        print("🚀 Starting FastAPI backend...")
        
        try:
            self.fastapi_process = subprocess.Popen([
                sys.executable, "-m", "uvicorn",
                "backend.main:app",
                "--host", "0.0.0.0",
                "--port", "8000",
                "--reload"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            return True
        except Exception as e:
            print(f"❌ Failed to start FastAPI: {e}")
            return False
    
    def start_streamlit(self):
        """Start Streamlit frontend"""
        print("🎨 Starting Streamlit frontend...")
        
        try:
            self.streamlit_process = subprocess.Popen([
                sys.executable, "-m", "streamlit", "run",
                "frontend/app.py",
                "--server.port", "8501",
                "--server.address", "0.0.0.0",
                "--server.headless", "true"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            return True
        except Exception as e:
            print(f"❌ Failed to start Streamlit: {e}")
            return False
    
    def open_browser(self):
        """Open browser to the application"""
        print("🌐 Opening browser...")
        try:
            webbrowser.open(self.frontend_url)
        except:
            print(f"💡 Manually open: {self.frontend_url}")
    
    def print_status(self):
        """Print system status"""
        print("\n" + "="*50)
        print("🪖 HELMET DETECTION SYSTEM - RUNNING")
        print("="*50)
        print(f"📡 API Backend:  {self.api_url}")
        print(f"🌐 Web Frontend: {self.frontend_url}")
        print("="*50)
        print("💡 Usage:")
        print("   1. Open the web interface in your browser")
        print("   2. Upload an image to detect helmets")
        print("   3. View safety compliance results")
        print("\n🛑 Press Ctrl+C to stop the system")
        print("="*50)
    
    def cleanup(self):
        """Clean up processes"""
        print("\n🛑 Shutting down system...")
        
        if self.fastapi_process:
            self.fastapi_process.terminate()
            print("✅ FastAPI stopped")
        
        if self.streamlit_process:
            self.streamlit_process.terminate()
            print("✅ Streamlit stopped")
        
        print("👋 System shutdown complete")
    
    def run(self):
        """Run the complete system"""
        print("🪖 Helmet Detection System Launcher")
        print("="*40)
        
        # Pre-flight checks
        if not self.check_dependencies():
            return False
        
        if not self.check_model_exists():
            return False
        
        try:
            # Start FastAPI
            if not self.start_fastapi():
                return False
            
            # Wait for API to be ready
            if not self.wait_for_api():
                self.cleanup()
                return False
            
            # Start Streamlit
            if not self.start_streamlit():
                self.cleanup()
                return False
            
            # Wait a bit for Streamlit to start
            time.sleep(3)
            
            # Print status and open browser
            self.print_status()
            self.open_browser()
            
            # Keep running until interrupted
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
            
        except Exception as e:
            print(f"❌ System error: {e}")
        finally:
            self.cleanup()
        
        return True

def main():
    launcher = SystemLauncher()
    success = launcher.run()
    
    if not success:
        print("\n❌ System failed to start")
        print("💡 Try running components manually:")
        print("   Backend:  uvicorn backend.main:app --reload")
        print("   Frontend: streamlit run frontend/app.py")
        sys.exit(1)

if __name__ == "__main__":
    main()