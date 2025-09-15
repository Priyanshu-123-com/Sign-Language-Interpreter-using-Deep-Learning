#!/usr/bin/env python3
"""
ASL Alphabet Recognition Launcher
Simple launcher for the industry-ready ASL recognition system
"""

import sys
import os
import subprocess

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'opencv-python',
        'mediapipe', 
        'pyttsx3',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'mediapipe':
                import mediapipe
            elif package == 'pyttsx3':
                import pyttsx3
            elif package == 'numpy':
                import numpy
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

def main():
    """Main launcher function"""
    print("=" * 60)
    print("ASL ALPHABET RECOGNITION SYSTEM")
    print("Industry-Ready Production Application")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    print("✅ All dependencies are installed")
    print("🚀 Starting ASL Alphabet Recognition System...")
    print("\nControls:")
    print("  Q - Quit application")
    print("  V - Toggle voice on/off")
    print("  C - Clear current word")
    print("  S - Save recognition history")
    print("\nInstructions:")
    print("  • Show ASL letters clearly in the green box")
    print("  • Close both hands for SPACE")
    print("  • Make sure you have good lighting")
    print("=" * 60)
    
    try:
        # Import and run the main application
        from asl_alphabet_recognition import main as run_app
        run_app()
    except KeyboardInterrupt:
        print("\n👋 Application closed by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check the logs for more details")

if __name__ == "__main__":
    main()
