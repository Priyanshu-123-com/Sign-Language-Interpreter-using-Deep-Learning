#!/usr/bin/env python3
"""
Simple Sign Language Recognition Demo
Uses pre-trained models and basic computer vision techniques
"""

import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
from threading import Thread

class SimpleSignRecognizer:
    def __init__(self):
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize text-to-speech
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        
        # Simple gesture recognition based on finger counts
        self.gesture_map = {
            0: "Zero",
            1: "One", 
            2: "Two",
            3: "Three",
            4: "Four",
            5: "Five",
            "thumbs_up": "Good",
            "thumbs_down": "Bad",
            "peace": "Peace",
            "ok": "OK"
        }
        
        self.last_gesture = ""
        self.gesture_count = 0
        self.min_gesture_frames = 10  # Minimum frames to confirm gesture
        
    def count_fingers(self, landmarks):
        """Count extended fingers based on hand landmarks"""
        tips = [4, 8, 12, 16, 20]  # Finger tip landmarks
        fingers = []
        
        # Thumb (special case)
        if landmarks[tips[0]].x > landmarks[tips[0] - 1].x:
            fingers.append(1)
        else:
            fingers.append(0)
            
        # Other fingers
        for i in range(1, 5):
            if landmarks[tips[i]].y < landmarks[tips[i] - 2].y:
                fingers.append(1)
            else:
                fingers.append(0)
                
        return sum(fingers)
    
    def detect_special_gestures(self, landmarks):
        """Detect special gestures like thumbs up, peace sign, etc."""
        # Thumbs up detection
        if (landmarks[4].y < landmarks[3].y and  # Thumb up
            landmarks[8].y > landmarks[6].y and  # Index down
            landmarks[12].y > landmarks[10].y and # Middle down
            landmarks[16].y > landmarks[14].y and # Ring down
            landmarks[20].y > landmarks[18].y):   # Pinky down
            return "thumbs_up"
            
        # Thumbs down detection
        if (landmarks[4].y > landmarks[3].y and  # Thumb down
            landmarks[8].y > landmarks[6].y and  # Index down
            landmarks[12].y > landmarks[10].y and # Middle down
            landmarks[16].y > landmarks[14].y and # Ring down
            landmarks[20].y > landmarks[18].y):   # Pinky down
            return "thumbs_down"
            
        # Peace sign (V sign)
        if (landmarks[8].y < landmarks[6].y and  # Index up
            landmarks[12].y < landmarks[10].y and # Middle up
            landmarks[16].y > landmarks[14].y and # Ring down
            landmarks[20].y > landmarks[18].y and # Pinky down
            landmarks[4].y > landmarks[3].y):     # Thumb down
            return "peace"
            
        # OK sign (thumb and index form circle)
        if (landmarks[4].x < landmarks[3].x and  # Thumb position
            landmarks[8].y < landmarks[6].y and  # Index up
            landmarks[12].y > landmarks[10].y and # Middle down
            landmarks[16].y > landmarks[14].y and # Ring down
            landmarks[20].y > landmarks[18].y):   # Pinky down
            return "ok"
            
        return None
    
    def say_text(self, text):
        """Convert text to speech"""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except:
            pass  # Ignore TTS errors
    
    def recognize_gesture(self, image):
        """Recognize gesture from image"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        
        gesture = None
        confidence = 0
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Count fingers
                finger_count = self.count_fingers(hand_landmarks.landmark)
                
                # Check for special gestures first
                special_gesture = self.detect_special_gestures(hand_landmarks.landmark)
                
                if special_gesture:
                    gesture = special_gesture
                    confidence = 0.9
                elif 0 <= finger_count <= 5:
                    gesture = finger_count
                    confidence = 0.8
                    
        return gesture, confidence
    
    def run_demo(self):
        """Run the sign language recognition demo"""
        print("Simple Sign Language Recognition Demo")
        print("=" * 50)
        print("Supported gestures:")
        print("- Numbers: 0, 1, 2, 3, 4, 5 (finger count)")
        print("- Thumbs up: Good")
        print("- Thumbs down: Bad") 
        print("- Peace sign: Peace")
        print("- OK sign: OK")
        print("\nPress 'q' to quit, 'v' to toggle voice")
        
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
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            frame_height, frame_width = frame.shape[:2]
            
            # Recognize gesture
            gesture, confidence = self.recognize_gesture(frame)
            
            # Update gesture recognition
            if gesture is not None and confidence > 0.7:
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
                gesture_text = self.gesture_map.get(current_gesture, str(current_gesture))
                print(f"Recognized: {gesture_text}")
                
                if voice_enabled:
                    Thread(target=self.say_text, args=(gesture_text,)).start()
                
                self.last_gesture = current_gesture
            
            # Draw UI
            cv2.rectangle(frame, (10, 10), (400, 100), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (400, 100), (255, 255, 255), 2)
            
            if current_gesture != "":
                gesture_text = self.gesture_map.get(current_gesture, str(current_gesture))
                cv2.putText(frame, f"Gesture: {gesture_text}", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {confidence:.2f}", (20, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Show a gesture...", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Voice status
            voice_status = "Voice: ON" if voice_enabled else "Voice: OFF"
            cv2.putText(frame, voice_status, (frame_width - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("Sign Language Recognition", frame)
            
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
    recognizer = SimpleSignRecognizer()
    recognizer.run_demo()

if __name__ == "__main__":
    main()
