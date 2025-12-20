#!/usr/bin/env python3
"""
Basic Sign Language Recognition Demo
Uses simple computer vision techniques without complex dependencies
"""

import cv2
import numpy as np
import pyttsx3
from threading import Thread

class BasicSignRecognizer:
    def __init__(self):
        # Initialize text-to-speech
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        
        # Simple gesture recognition based on contour analysis
        self.gesture_map = {
            "fist": "Fist",
            "open_hand": "Open Hand", 
            "point": "Point",
            "peace": "Peace",
            "thumbs_up": "Thumbs Up",
            "ok": "OK"
        }
        
        self.last_gesture = ""
        self.gesture_count = 0
        self.min_gesture_frames = 15  # Minimum frames to confirm gesture
        
        # Hand detection parameters
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
    def detect_hand_region(self, frame):
        """Detect hand region using color-based segmentation"""
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for skin color
        mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        
        # Morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour (hand)
            hand_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(hand_contour) > 5000:  # Minimum area threshold
                return hand_contour, mask
                
        return None, mask
    
    def analyze_hand_gesture(self, contour):
        """Analyze hand contour to determine gesture"""
        if contour is None:
            return None, 0.0
            
        # Get contour properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if area < 5000:  # Too small
            return None, 0.0
            
        # Approximate the contour
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Get convex hull
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)
        
        # Count fingers based on convexity defects
        finger_count = 0
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])
                
                # Calculate angle between fingers
                a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                
                if a != 0 and b != 0 and c != 0:
                    angle = np.arccos((b**2 + c**2 - a**2) / (2*b*c))
                    if angle <= np.pi/2:  # 90 degrees
                        finger_count += 1
        
        # Determine gesture based on finger count and contour properties
        if finger_count == 0:
            return "fist", 0.8
        elif finger_count == 1:
            return "point", 0.7
        elif finger_count == 2:
            return "peace", 0.8
        elif finger_count >= 4:
            return "open_hand", 0.8
        else:
            return "ok", 0.6
    
    def detect_special_gestures(self, contour, frame):
        """Detect special gestures using additional analysis"""
        if contour is None:
            return None, 0.0
            
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        
        # Get hand center
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Analyze hand orientation and shape
            if aspect_ratio > 1.2:  # Wide hand
                return "open_hand", 0.9
            elif aspect_ratio < 0.8:  # Tall hand
                return "point", 0.8
            else:  # Square-ish hand
                return "fist", 0.7
                
        return None, 0.0
    
    def say_text(self, text):
        """Convert text to speech"""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except:
            pass  # Ignore TTS errors
    
    def run_demo(self):
        """Run the basic sign language recognition demo"""
        print("Basic Sign Language Recognition Demo")
        print("=" * 50)
        print("Supported gestures:")
        print("- Fist: Closed hand")
        print("- Open Hand: All fingers extended")
        print("- Point: One finger extended")
        print("- Peace: Two fingers extended")
        print("- OK: Thumb and index finger circle")
        print("\nInstructions:")
        print("1. Make sure you have good lighting")
        print("2. Keep your hand in the center of the frame")
        print("3. Make clear, distinct gestures")
        print("4. Press 'q' to quit, 'v' to toggle voice")
        
        # Try different camera indices
        cap = None
        for camera_index in [0, 1, 2]:
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"✓ Using camera {camera_index}")
                    break
                else:
                    cap.release()
            else:
                cap.release()
        
        if cap is None or not cap.isOpened():
            print("✗ No camera found!")
            return
            
        voice_enabled = True
        current_gesture = ""
        gesture_frames = 0
        
        print("\nStarting recognition...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            frame_height, frame_width = frame.shape[:2]
            
            # Detect hand region
            hand_contour, mask = self.detect_hand_region(frame)
            
            # Analyze gesture
            gesture, confidence = self.analyze_hand_gesture(hand_contour)
            
            # If basic analysis fails, try special gesture detection
            if gesture is None or confidence < 0.5:
                gesture, confidence = self.detect_special_gestures(hand_contour, frame)
            
            # Update gesture recognition
            if gesture is not None and confidence > 0.5:
                if gesture == current_gesture:
                    gesture_frames += 1
                else:
                    current_gesture = gesture
                    gesture_frames = 1
            else:
                gesture_frames = 0
                current_gesture = ""
            
            # Display recognized gesture
            if gesture_frames >= self.min_gesture_frames and current_gesture != self.last_gesture:
                gesture_text = self.gesture_map.get(current_gesture, current_gesture)
                print(f"Recognized: {gesture_text} (confidence: {confidence:.2f})")
                
                if voice_enabled:
                    Thread(target=self.say_text, args=(gesture_text,)).start()
                
                self.last_gesture = current_gesture
            
            # Draw hand contour
            if hand_contour is not None:
                cv2.drawContours(frame, [hand_contour], -1, (0, 255, 0), 2)
                
                # Draw bounding rectangle
                x, y, w, h = cv2.boundingRect(hand_contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Draw UI
            cv2.rectangle(frame, (10, 10), (400, 120), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (400, 120), (255, 255, 255), 2)
            
            if current_gesture != "":
                gesture_text = self.gesture_map.get(current_gesture, current_gesture)
                cv2.putText(frame, f"Gesture: {gesture_text}", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {confidence:.2f}", (20, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Frames: {gesture_frames}", (20, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Show a gesture...", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, "Make sure hand is visible", (20, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Voice status
            voice_status = "Voice: ON" if voice_enabled else "Voice: OFF"
            cv2.putText(frame, voice_status, (frame_width - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show both original and mask
            mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            combined = np.hstack((frame, mask_colored))
            
            cv2.imshow("Sign Language Recognition", combined)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('v'):
                voice_enabled = not voice_enabled
                print(f"Voice {'enabled' if voice_enabled else 'disabled'}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("Demo ended!")

def main():
    recognizer = BasicSignRecognizer()
    recognizer.run_demo()

if __name__ == "__main__":
    main()
