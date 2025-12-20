#!/usr/bin/env python3
"""
ASL Alphabet Recognition Demo
Focuses specifically on recognizing American Sign Language alphabet gestures
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyttsx3
from threading import Thread
import sys
import os

class AlphabetRecognizer:
    def __init__(self):
        # Initialize MediaPipe Hand Landmarker
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(base_options=base_options,
                                           num_hands=2,
                                           min_hand_detection_confidence=0.7,
                                           min_hand_presence_confidence=0.7,
                                           min_tracking_confidence=0.5)
        self.hands = vision.HandLandmarker.create_from_options(options)
        # Drawing will be handled through the new API
        
        # Initialize text-to-speech
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.voice_enabled = True
        except:
            print("⚠ Text-to-speech not available")
            self.voice_enabled = False
        
        # ASL Alphabet mapping
        self.alphabet_map = {
            'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E', 'F': 'F',
            'G': 'G', 'H': 'H', 'I': 'I', 'J': 'J', 'K': 'K', 'L': 'L',
            'M': 'M', 'N': 'N', 'O': 'O', 'P': 'P', 'Q': 'Q', 'R': 'R',
            'S': 'S', 'T': 'T', 'U': 'U', 'V': 'V', 'W': 'W', 'X': 'X',
            'Y': 'Y', 'Z': 'Z'
        }
        
        self.last_letter = ""
        self.letter_count = 0
        self.min_letter_frames = 8  # Faster recognition for letters
        self.current_word = ""
        
    def detect_hand_closed(self, landmarks):
        """Check if a hand is closed (fist)"""
        thumb_up = landmarks[4].y < landmarks[3].y
        index_up = landmarks[8].y < landmarks[6].y
        middle_up = landmarks[12].y < landmarks[10].y
        ring_up = landmarks[16].y < landmarks[14].y
        pinky_up = landmarks[20].y < landmarks[18].y
        
        # Hand is closed if no fingers are up
        return not (index_up or middle_up or ring_up or pinky_up)
        
    def detect_alphabet_gestures(self, landmarks):
        """Detect ASL alphabet gestures using MediaPipe landmarks"""
        # Get finger states
        thumb_up = landmarks[4].y < landmarks[3].y
        index_up = landmarks[8].y < landmarks[6].y
        middle_up = landmarks[12].y < landmarks[10].y
        ring_up = landmarks[16].y < landmarks[14].y
        pinky_up = landmarks[20].y < landmarks[18].y
        
        # Get finger positions for more complex gestures
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        # A - Fist (all fingers down, thumb across palm)
        if not index_up and not middle_up and not ring_up and not pinky_up:
            if landmarks[4].x < landmarks[3].x:  # Thumb across
                return "A"
        
        # B - All fingers up, thumb down
        if index_up and middle_up and ring_up and pinky_up and not thumb_up:
            return "B"
        
        # C - Curved hand (thumb and index form C shape)
        if (landmarks[4].x > landmarks[3].x and  # Thumb out
            landmarks[8].y < landmarks[6].y and  # Index up
            landmarks[12].y > landmarks[10].y and # Middle down
            landmarks[16].y > landmarks[14].y and # Ring down
            landmarks[20].y > landmarks[18].y):   # Pinky down
            return "C"
        
        # D - Only index finger up
        if index_up and not middle_up and not ring_up and not pinky_up and not thumb_up:
            return "D"
        
        # E - All fingers down, thumb up
        if not index_up and not middle_up and not ring_up and not pinky_up and thumb_up:
            return "E"
        
        # F - Thumb and index touching, other fingers up
        if (landmarks[4].x < landmarks[3].x and  # Thumb position
            landmarks[8].y < landmarks[6].y and  # Index up
            landmarks[12].y < landmarks[10].y and # Middle up
            landmarks[16].y > landmarks[14].y and # Ring down
            landmarks[20].y > landmarks[18].y):   # Pinky down
            return "F"
        
        # G - Index and thumb pointing (like gun)
        if (landmarks[4].x > landmarks[3].x and  # Thumb out
            landmarks[8].y < landmarks[6].y and  # Index up
            landmarks[12].y > landmarks[10].y and # Middle down
            landmarks[16].y > landmarks[14].y and # Ring down
            landmarks[20].y > landmarks[18].y):   # Pinky down
            return "G"
        
        # H - Index and middle finger up, others down
        if index_up and middle_up and not ring_up and not pinky_up and not thumb_up:
            return "H"
        
        # I - Only pinky up
        if not index_up and not middle_up and not ring_up and pinky_up and not thumb_up:
            return "I"
        
        # J - I gesture with movement (simplified as I)
        if not index_up and not middle_up and not ring_up and pinky_up and not thumb_up:
            return "J"
        
        # K - Index and middle finger up, thumb between them
        if (index_up and middle_up and not ring_up and not pinky_up and 
            landmarks[4].y < landmarks[3].y):  # Thumb up
            return "K"
        
        # L - Index and thumb up, others down
        if (index_up and not middle_up and not ring_up and not pinky_up and 
            landmarks[4].y < landmarks[3].y):  # Thumb up
            return "L"
        
        # M - Thumb between ring and pinky, others down
        if (not index_up and not middle_up and not ring_up and not pinky_up and 
            landmarks[4].y < landmarks[3].y):  # Thumb up
            return "M"
        
        # N - Thumb between middle and ring, others down
        if (not index_up and not middle_up and not ring_up and not pinky_up and 
            landmarks[4].y < landmarks[3].y):  # Thumb up
            return "N"
        
        # O - Thumb and fingers form circle
        if (landmarks[4].x < landmarks[3].x and  # Thumb position
            landmarks[8].y < landmarks[6].y and  # Index up
            landmarks[12].y > landmarks[10].y and # Middle down
            landmarks[16].y > landmarks[14].y and # Ring down
            landmarks[20].y > landmarks[18].y):   # Pinky down
            return "O"
        
        # P - Thumb and index down, others up
        if not index_up and middle_up and ring_up and pinky_up and not thumb_up:
            return "P"
        
        # Q - Thumb and pinky up, others down
        if (not index_up and not middle_up and not ring_up and pinky_up and 
            landmarks[4].y < landmarks[3].y):  # Thumb up
            return "Q"
        
        # R - Index and middle crossed, others down
        if (index_up and middle_up and not ring_up and not pinky_up and not thumb_up and
            landmarks[8].x > landmarks[12].x):  # Index over middle
            return "R"
        
        # S - Fist with thumb over fingers
        if not index_up and not middle_up and not ring_up and not pinky_up and thumb_up:
            return "S"
        
        # T - Thumb between index and middle, others down
        if (not index_up and not middle_up and not ring_up and not pinky_up and 
            landmarks[4].y < landmarks[3].y):  # Thumb up
            return "T"
        
        # U - Index and middle up, others down
        if index_up and middle_up and not ring_up and not pinky_up and not thumb_up:
            return "U"
        
        # V - Index and middle up spread apart, others down
        if (index_up and middle_up and not ring_up and not pinky_up and not thumb_up and
            abs(landmarks[8].x - landmarks[12].x) > 0.05):  # Fingers spread
            return "V"
        
        # W - Index, middle, ring up, others down
        if index_up and middle_up and ring_up and not pinky_up and not thumb_up:
            return "W"
        
        # X - Index finger bent, others down
        if (not index_up and not middle_up and not ring_up and not pinky_up and not thumb_up and
            landmarks[8].y > landmarks[6].y and landmarks[8].y < landmarks[5].y):  # Index bent
            return "X"
        
        # Y - Thumb and pinky up, others down
        if (not index_up and not middle_up and not ring_up and pinky_up and 
            landmarks[4].y < landmarks[3].y):  # Thumb up
            return "Y"
        
        # Z - Index finger draws Z shape (simplified as index up)
        if index_up and not middle_up and not ring_up and not pinky_up and not thumb_up:
            return "Z"
        
        return None
    
    def say_text(self, text):
        """Convert text to speech"""
        if not self.voice_enabled:
            return
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except:
            pass  # Ignore TTS errors
    
    def recognize_letter(self, image):
        """Recognize letter from image"""
        # Convert the frame to MediaPipe Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # Process the image
        detection_result = self.hands.detect(mp_image)
        
        letter = None
        confidence = 0
        
        if detection_result.hand_landmarks:
            # Check for two hands closed (space functionality)
            if len(detection_result.hand_landmarks) == 2:
                left_hand_closed = self.detect_hand_closed(detection_result.hand_landmarks[0])
                right_hand_closed = self.detect_hand_closed(detection_result.hand_landmarks[1])
                
                if left_hand_closed and right_hand_closed:
                    return "SPACE", 0.95
            
            # Process each hand
            for hand_landmarks in detection_result.hand_landmarks:
                # Draw hand landmarks
                # Detect alphabet gesture
                letter = self.detect_alphabet_gestures(hand_landmarks)
                if letter:
                    confidence = 0.9
                    
        return letter, confidence
    
    def run_demo(self):
        """Run the alphabet recognition demo"""
        print("=" * 60)
        print("ASL Alphabet Recognition Demo")
        print("=" * 60)
        print("Supported letters: A-Z")
        print("Make clear ASL alphabet gestures")
        print("Close both hands for SPACE")
        print("\nControls:")
        print("- Press 'q' to quit")
        print("- Press 'v' to toggle voice")
        print("- Press 'c' to clear current word")
        print("- Press 'h' to show this help")
        print("=" * 60)
        
        # Try different camera indices
        cap = None
        for camera_index in [0, 1, 2]:
            print(f"Trying camera {camera_index}...")
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
        current_letter = ""
        letter_frames = 0
        show_help = False
        
        print("\nStarting alphabet recognition...")
        print("Make sure your hand is visible in the camera!")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            frame_height, frame_width = frame.shape[:2]
            
            # Recognize letter
            letter, confidence = self.recognize_letter(frame)
            
            # Update letter recognition
            if letter is not None and confidence > 0.7:
                if letter == current_letter:
                    letter_frames += 1
                else:
                    current_letter = letter
                    letter_frames = 1
            else:
                letter_frames = 0
                current_letter = ""
            
            # Add letter to word when recognized
            if letter_frames >= self.min_letter_frames and current_letter != self.last_letter:
                if current_letter == "SPACE":
                    print(f"Recognized: SPACE (confidence: {confidence:.2f})")
                    if voice_enabled:
                        Thread(target=self.say_text, args=("space",)).start()
                    self.current_word += " "
                else:
                    print(f"Recognized: {current_letter} (confidence: {confidence:.2f})")
                    if voice_enabled:
                        Thread(target=self.say_text, args=(current_letter,)).start()
                    self.current_word += current_letter
                
                self.last_letter = current_letter
            
            # Enhanced UI Design with improved visuals
            display_height = frame_height
            display_width = frame_width
            panel_width = 450  # Increased panel width for better layout
            
            # Create main panel with enhanced styling
            main_panel = np.zeros((display_height, display_width + panel_width, 3), dtype=np.uint8)
            
            # Add gradient background with better colors
            for i in range(display_height):
                # Create a smooth gradient from dark to light
                intensity = int(15 + (i / display_height) * 25)
                main_panel[i, :display_width] = [intensity, intensity, intensity]
            
            # Copy camera feed
            main_panel[:display_height, :display_width] = frame
            
            # Create info panel with gradient background
            info_panel = np.zeros((display_height, panel_width, 3), dtype=np.uint8)
            # Create vertical gradient for info panel
            for i in range(display_height):
                intensity = int(20 + (i / display_height) * 15)
                info_panel[i, :] = [15 + intensity//3, 15 + intensity//2, 25 + intensity]
            
            # Add decorative border to info panel
            cv2.rectangle(info_panel, (0, 0), (panel_width-1, display_height-1), (80, 120, 180), 2)
            cv2.rectangle(info_panel, (5, 5), (panel_width-6, display_height-6), (60, 100, 160), 1)
            
            # Header with improved styling
            header_height = 90
            cv2.rectangle(info_panel, (0, 0), (panel_width, header_height), (30, 50, 90), -1)
            cv2.line(info_panel, (0, header_height), (panel_width, header_height), (100, 180, 255), 2)
            
            # Title with shadow effect
            cv2.putText(info_panel, "ASL ALPHABET", (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3)
            cv2.putText(info_panel, "ASL ALPHABET", (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 180), 2)
            cv2.putText(info_panel, "RECOGNITION", (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3)
            cv2.putText(info_panel, "RECOGNITION", (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (100, 255, 200), 2)
            
            # Status indicator
            status_text = "RUNNING" if letter_frames > 0 else "WAITING"
            status_color = (0, 255, 0) if letter_frames > 0 else (0, 200, 255)
            cv2.circle(info_panel, (panel_width - 40, 40), 10, status_color, -1)
            cv2.circle(info_panel, (panel_width - 40, 40), 10, (255, 255, 255), 1)
            cv2.putText(info_panel, status_text, (panel_width - 120, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Current letter display with larger font and better positioning
            y_offset = 120
            cv2.putText(info_panel, "CURRENT LETTER:", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 230, 255), 2)
            if current_letter:
                letter_color = (0, 255, 0) if current_letter != "SPACE" else (255, 165, 0)
                # Larger letter display with background
                cv2.rectangle(info_panel, (20, y_offset + 10), (120, y_offset + 80), (30, 30, 50), -1)
                cv2.rectangle(info_panel, (20, y_offset + 10), (120, y_offset + 80), (100, 150, 200), 1)
                cv2.putText(info_panel, current_letter, (50, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, letter_color, 3)
            else:
                cv2.rectangle(info_panel, (20, y_offset + 10), (120, y_offset + 80), (30, 30, 50), -1)
                cv2.rectangle(info_panel, (20, y_offset + 10), (120, y_offset + 80), (100, 150, 200), 1)
                cv2.putText(info_panel, "--", (50, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (150, 150, 150), 3)
            
            # Frame counter for stability indication
            cv2.putText(info_panel, f"Stability: {letter_frames}/{self.min_letter_frames}", (140, y_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Confidence bar with gradient
            y_offset += 90
            cv2.putText(info_panel, "CONFIDENCE:", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 230, 255), 2)
            conf_width = int(350 * confidence)
            # Background bar
            cv2.rectangle(info_panel, (20, y_offset + 10), (370, y_offset + 30), (50, 50, 70), -1)
            cv2.rectangle(info_panel, (20, y_offset + 10), (370, y_offset + 30), (100, 100, 150), 1)
            # Progress bar with color gradient based on confidence
            if conf_width > 0:
                conf_color = (0, 255, 0) if confidence > 0.8 else (255, 255, 0) if confidence > 0.6 else (255, 0, 0)
                cv2.rectangle(info_panel, (20, y_offset + 10), (20 + conf_width, y_offset + 30), conf_color, -1)
            cv2.putText(info_panel, f"{confidence:.1%}", (380, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Word display with scrollable area simulation
            y_offset += 60
            cv2.putText(info_panel, "RECOGNIZED WORD:", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 230, 255), 2)
            word_display = self.current_word if self.current_word else "Start typing..."
            word_color = (255, 255, 255) if self.current_word else (150, 150, 150)
            # Word background
            word_bg_width = min(400, len(word_display) * 15 + 20)
            cv2.rectangle(info_panel, (20, y_offset + 10), (20 + word_bg_width, y_offset + 50), (30, 40, 60), -1)
            cv2.rectangle(info_panel, (20, y_offset + 10), (20 + word_bg_width, y_offset + 50), (100, 150, 200), 1)
            # Truncate long words for display
            display_word = word_display if len(word_display) <= 25 else word_display[-25:]
            cv2.putText(info_panel, display_word, (30, y_offset + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, word_color, 1)
            
            # Character count
            if self.current_word:
                char_count = len(self.current_word)
                cv2.putText(info_panel, f"({char_count} chars)", (20 + word_bg_width + 10, y_offset + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Instructions section with better organization
            y_offset += 80
            cv2.putText(info_panel, "INSTRUCTIONS:", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 100), 2)
            
            instructions = [
                "• Show clear ASL letters in the camera frame",
                "• Hold gesture steady for recognition (~1 sec)",
                "• Close both hands to add SPACE to word",
                "• Keep hand centered in green box (if visible)"
            ]
            
            for i, instruction in enumerate(instructions):
                cv2.putText(info_panel, instruction, (20, y_offset + 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
            
            # Controls section
            y_offset += 150
            cv2.putText(info_panel, "CONTROLS:", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 100), 2)
            
            controls = [
                "Q - Quit Application",
                "V - Toggle Voice Output",
                "C - Clear Current Word",
                "H - Toggle Help Panel"
            ]
            
            for i, control in enumerate(controls):
                cv2.putText(info_panel, control, (20, y_offset + 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
            
            # Voice status indicator with better design
            y_offset += 140
            voice_color = (0, 255, 0) if voice_enabled else (255, 0, 0)
            voice_text = "VOICE ON" if voice_enabled else "VOICE OFF"
            cv2.circle(info_panel, (40, y_offset), 12, voice_color, -1)
            cv2.circle(info_panel, (40, y_offset), 12, (255, 255, 255), 2)
            cv2.putText(info_panel, voice_text, (60, y_offset + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, voice_color, 2)
            
            # Footer with timestamp
            timestamp = cv2.getTextSize("ASL Recognition v1.0", cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            cv2.putText(info_panel, "ASL Recognition v1.0", (panel_width - timestamp[0] - 20, display_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 200), 1)
            
            # Help text overlay with improved design
            if show_help:
                # Semi-transparent overlay
                overlay = info_panel.copy()
                cv2.rectangle(overlay, (50, 50), (panel_width - 50, display_height - 50), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, info_panel, 0.3, 0, info_panel)
                
                # Help panel border
                cv2.rectangle(info_panel, (50, 50), (panel_width - 50, display_height - 50), (100, 180, 255), 2)
                cv2.rectangle(info_panel, (55, 55), (panel_width - 55, display_height - 55), (80, 150, 220), 1)
                
                # Help title
                cv2.putText(info_panel, "HELP PANEL", (panel_width//2 - 70, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 255, 200), 2)
                
                # Help content
                help_sections = [
                    ("Navigation Keys:", ["Q - Quit application", "V - Toggle voice output", "C - Clear current word", "H - Toggle this help"]),
                    ("Recognition Guide:", ["Show clear ASL letters", "Hold gesture for 1-2 seconds", "Two closed hands = SPACE", "Ensure good lighting"]),
                    ("Troubleshooting:", ["If no detection, check lighting", "Move hand closer to camera", "Try different background", "Restart if issues persist"])
                ]
                
                y_pos = 120
                for section_title, section_items in help_sections:
                    cv2.putText(info_panel, section_title, (80, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 100), 2)
                    y_pos += 30
                    for item in section_items:
                        cv2.putText(info_panel, "• " + item, (100, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
                        y_pos += 25
                    y_pos += 10
            
            # Combine panels
            main_panel[:display_height, display_width:] = info_panel
            
            # Add a subtle border between camera and info panel
            cv2.line(main_panel, (display_width, 0), (display_width, display_height), (100, 150, 200), 1)
            
            cv2.imshow("ASL Alphabet Recognition - Enhanced UI", main_panel)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('v'):
                voice_enabled = not voice_enabled
                print(f"Voice {'enabled' if voice_enabled else 'disabled'}")
            elif key == ord('c'):
                self.current_word = ""
                print("Word cleared")
            elif key == ord('h'):
                show_help = not show_help
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nFinal word: {self.current_word}")
        print("Demo ended!")

def main():
    try:
        recognizer = AlphabetRecognizer()
        recognizer.run_demo()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have installed all required packages:")
        print("pip install opencv-python mediapipe pyttsx3 numpy")

if __name__ == "__main__":
    main()
