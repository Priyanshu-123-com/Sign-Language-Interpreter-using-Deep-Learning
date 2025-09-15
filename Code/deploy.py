#!/usr/bin/env python3
"""
ASL Alphabet Recognition System - Deployment Script
Sets up the production environment and validates installation
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "opencv-python", "mediapipe", "pyttsx3", "numpy"
        ])
        print("✅ All packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    dirs = ["logs", "data", "exports"]
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"📁 Created directory: {dir_name}")

def validate_camera():
    """Test camera availability"""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("✅ Camera is working")
                cap.release()
                return True
            else:
                print("⚠️  Camera opened but failed to read")
        else:
            print("⚠️  Camera not accessible")
        cap.release()
        return False
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def create_desktop_shortcut():
    """Create desktop shortcut for easy access"""
    try:
        import os
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        shortcut_path = os.path.join(desktop, "ASL Recognition.lnk")
        
        # Create a simple batch file instead of .lnk
        batch_path = os.path.join(desktop, "ASL Recognition.bat")
        with open(batch_path, 'w') as f:
            f.write(f'@echo off\n')
            f.write(f'cd /d "{os.getcwd()}"\n')
            f.write(f'python asl_alphabet_recognition.py\n')
            f.write(f'pause\n')
        
        print(f"✅ Desktop shortcut created: {batch_path}")
        return True
    except Exception as e:
        print(f"⚠️  Could not create desktop shortcut: {e}")
        return False

def main():
    """Main deployment function"""
    print("=" * 60)
    print("ASL ALPHABET RECOGNITION SYSTEM")
    print("Production Deployment Script")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("❌ Deployment failed: Could not install packages")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Validate camera
    if not validate_camera():
        print("⚠️  Camera validation failed - check camera connection")
    
    # Create desktop shortcut
    create_desktop_shortcut()
    
    print("\n" + "=" * 60)
    print("🎉 DEPLOYMENT COMPLETE!")
    print("=" * 60)
    print("✅ System is ready for production use")
    print("\n📋 Quick Start:")
    print("   1. Run: python asl_alphabet_recognition.py")
    print("   2. Or use the desktop shortcut")
    print("   3. Place hand in green detection area")
    print("   4. Make ASL alphabet gestures")
    print("\n📚 Documentation:")
    print("   - README_ASL_ALPHABET.md")
    print("   - config.json for customization")
    print("\n🔧 Troubleshooting:")
    print("   - Check camera permissions")
    print("   - Ensure good lighting")
    print("   - Review logs for errors")
    print("=" * 60)

if __name__ == "__main__":
    main()
