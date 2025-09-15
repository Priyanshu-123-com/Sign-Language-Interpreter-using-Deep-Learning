#!/usr/bin/env python3
"""
Test script to verify all required libraries are installed correctly
for the Sign Language Recognition System
"""

def test_imports():
    """Test importing all required libraries"""
    print("Testing library imports...")
    
    try:
        import cv2
        print("✓ OpenCV (cv2) - Version:", cv2.__version__)
    except ImportError as e:
        print("✗ OpenCV import failed:", e)
        return False
    
    try:
        import numpy as np
        print("✓ NumPy - Version:", np.__version__)
    except ImportError as e:
        print("✗ NumPy import failed:", e)
        return False
    
    try:
        import tensorflow as tf
        print("✓ TensorFlow - Version:", tf.__version__)
    except ImportError as e:
        print("✗ TensorFlow import failed:", e)
        return False
    
    try:
        import keras
        print("✓ Keras - Version:", keras.__version__)
    except ImportError as e:
        print("✗ Keras import failed:", e)
        return False
    
    try:
        import h5py
        print("✓ H5py - Version:", h5py.__version__)
    except ImportError as e:
        print("✗ H5py import failed:", e)
        return False
    
    try:
        import sklearn
        print("✓ Scikit-learn - Version:", sklearn.__version__)
    except ImportError as e:
        print("✗ Scikit-learn import failed:", e)
        return False
    
    try:
        import scipy
        print("✓ SciPy - Version:", scipy.__version__)
    except ImportError as e:
        print("✗ SciPy import failed:", e)
        return False
    
    try:
        import pyttsx3
        print("✓ pyttsx3 - Text-to-speech library")
    except ImportError as e:
        print("✗ pyttsx3 import failed:", e)
        return False
    
    try:
        import matplotlib
        print("✓ Matplotlib - Version:", matplotlib.__version__)
    except ImportError as e:
        print("✗ Matplotlib import failed:", e)
        return False
    
    try:
        import sqlite3
        print("✓ SQLite3 - Built-in database library")
    except ImportError as e:
        print("✗ SQLite3 import failed:", e)
        return False
    
    try:
        import pickle
        print("✓ Pickle - Built-in serialization library")
    except ImportError as e:
        print("✗ Pickle import failed:", e)
        return False
    
    try:
        import glob
        print("✓ Glob - Built-in file pattern matching")
    except ImportError as e:
        print("✗ Glob import failed:", e)
        return False
    
    return True

def test_tensorflow_gpu():
    """Test if TensorFlow can detect GPU"""
    try:
        import tensorflow as tf
        print("\nTesting TensorFlow GPU support...")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ GPU detected: {len(gpus)} GPU(s) available")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu}")
        else:
            print("ℹ No GPU detected - will use CPU")
        return True
    except Exception as e:
        print(f"✗ GPU test failed: {e}")
        return False

def test_opencv_camera():
    """Test if OpenCV can access camera"""
    try:
        import cv2
        print("\nTesting camera access...")
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✓ Camera access successful")
            cap.release()
            return True
        else:
            print("ℹ Camera not accessible (may be in use by another application)")
            return True  # Not a critical failure
    except Exception as e:
        print(f"ℹ Camera test failed: {e}")
        return True  # Not a critical failure

def main():
    print("=" * 60)
    print("Sign Language Recognition System - Installation Test")
    print("=" * 60)
    
    # Test all imports
    if test_imports():
        print("\n✓ All required libraries imported successfully!")
    else:
        print("\n✗ Some libraries failed to import. Please check the errors above.")
        return False
    
    # Test additional features
    test_tensorflow_gpu()
    test_opencv_camera()
    
    print("\n" + "=" * 60)
    print("Installation test completed!")
    print("Your sign language recognition system is ready to use.")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    main()
