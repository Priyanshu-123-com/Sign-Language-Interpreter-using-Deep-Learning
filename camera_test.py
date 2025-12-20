#!/usr/bin/env python3
"""
Camera Test Script for Sign Language Recognition System
This will help diagnose camera access issues
"""

import cv2
import sys

def test_camera_access():
    """Test camera access with different camera indices"""
    print("Testing camera access...")
    
    # Try different camera indices
    for camera_index in range(3):  # Try cameras 0, 1, 2
        print(f"\nTrying camera index {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        
        if cap.isOpened():
            print(f"✓ Camera {camera_index} is accessible")
            
            # Try to read a frame
            ret, frame = cap.read()
            if ret:
                print(f"✓ Successfully captured frame from camera {camera_index}")
                print(f"  Frame shape: {frame.shape}")
                
                # Show a preview for 3 seconds
                print(f"  Showing preview from camera {camera_index} for 3 seconds...")
                print("  Press any key to close preview")
                
                for i in range(90):  # 3 seconds at 30fps
                    ret, frame = cap.read()
                    if ret:
                        cv2.imshow(f'Camera {camera_index} Test', frame)
                        if cv2.waitKey(1) & 0xFF != 255:  # If any key pressed
                            break
                    else:
                        break
                
                cv2.destroyAllWindows()
                cap.release()
                return camera_index
            else:
                print(f"✗ Could not read frame from camera {camera_index}")
        else:
            print(f"✗ Camera {camera_index} is not accessible")
        
        cap.release()
    
    return None

def check_camera_permissions():
    """Check if camera permissions are available"""
    print("\nChecking camera permissions...")
    
    try:
        # Try to create VideoCapture object
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✓ Camera permissions appear to be available")
            cap.release()
            return True
        else:
            print("✗ Camera permissions may be denied")
            return False
    except Exception as e:
        print(f"✗ Error accessing camera: {e}")
        return False

def main():
    print("=" * 60)
    print("Camera Access Test for Sign Language Recognition")
    print("=" * 60)
    
    # Check permissions first
    if not check_camera_permissions():
        print("\nTroubleshooting tips:")
        print("1. Make sure no other application is using the camera")
        print("2. Check Windows camera privacy settings")
        print("3. Try running as administrator")
        print("4. Restart your computer")
        return False
    
    # Test camera access
    working_camera = test_camera_access()
    
    if working_camera is not None:
        print(f"\n✓ Camera {working_camera} is working!")
        print("You can now proceed with the sign language recognition setup.")
        return True
    else:
        print("\n✗ No working camera found")
        print("\nTroubleshooting steps:")
        print("1. Close any applications using the camera (Zoom, Teams, Skype, etc.)")
        print("2. Check Windows Camera Privacy Settings:")
        print("   - Go to Settings > Privacy > Camera")
        print("   - Make sure 'Allow apps to access your camera' is ON")
        print("   - Make sure 'Allow desktop apps to access your camera' is ON")
        print("3. Try unplugging and reconnecting external cameras")
        print("4. Restart your computer")
        return False

if __name__ == "__main__":
    main()
